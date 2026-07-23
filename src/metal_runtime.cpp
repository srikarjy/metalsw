#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "metal_runtime.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <stdexcept>

#include "blosum62.hpp"

namespace metalsw {

namespace {

constexpr uint32_t kMaxQueryLen = 512;

struct Params {
    uint32_t queryLen;
    uint32_t dbCount;
    int32_t gapOpen;
    int32_t gapExtend;
};

}  // namespace

std::vector<int> runSmithWatermanGpu(const std::string &metallibPath, const std::string &query,
                                      const std::vector<FastaRecord> &dbRecords, int gapOpen,
                                      int gapExtend) {
    if (query.size() > kMaxQueryLen) {
        throw std::runtime_error("query longer than MAX_QUERY_LEN (512) in naive v0.2 kernel");
    }

    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device *device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        pool->release();
        throw std::runtime_error("no Metal device available");
    }

    NS::Error *error = nullptr;
    NS::String *pathStr =
        NS::String::string(metallibPath.c_str(), NS::StringEncoding::UTF8StringEncoding);
    NS::URL *url = NS::URL::fileURLWithPath(pathStr);
    MTL::Library *library = device->newLibrary(url, &error);
    if (!library) {
        std::string msg = "failed to load metallib: " + metallibPath;
        if (error) msg += std::string(" (") + error->localizedDescription()->utf8String() + ")";
        device->release();
        pool->release();
        throw std::runtime_error(msg);
    }

    MTL::Function *fn = library->newFunction(
        NS::String::string("smith_waterman_score", NS::StringEncoding::UTF8StringEncoding));
    MTL::ComputePipelineState *pso = device->newComputePipelineState(fn, &error);
    if (!pso) {
        std::string msg = "failed to create compute pipeline state";
        if (error) msg += std::string(" (") + error->localizedDescription()->utf8String() + ")";
        fn->release();
        library->release();
        device->release();
        pool->release();
        throw std::runtime_error(msg);
    }

    // Pack DB sequences into one contiguous buffer + offsets/lengths.
    std::vector<char> dbConcat;
    std::vector<uint32_t> offsets(dbRecords.size());
    std::vector<uint32_t> lengths(dbRecords.size());
    uint32_t cursor = 0;
    for (size_t i = 0; i < dbRecords.size(); ++i) {
        offsets[i] = cursor;
        lengths[i] = static_cast<uint32_t>(dbRecords[i].sequence.size());
        dbConcat.insert(dbConcat.end(), dbRecords[i].sequence.begin(),
                         dbRecords[i].sequence.end());
        cursor += lengths[i];
    }
    if (dbConcat.empty()) dbConcat.push_back('\0');  // avoid zero-length buffer

    std::vector<int16_t> blosumFlat(kAlphabetSize * kAlphabetSize);
    for (int i = 0; i < kAlphabetSize; ++i)
        for (int j = 0; j < kAlphabetSize; ++j) blosumFlat[i * kAlphabetSize + j] = kBlosum62[i][j];

    Params params{static_cast<uint32_t>(query.size()), static_cast<uint32_t>(dbRecords.size()),
                   gapOpen, gapExtend};

    auto makeBuffer = [&](const void *data, size_t length) {
        return device->newBuffer(data, length, MTL::ResourceStorageModeShared);
    };

    MTL::Buffer *queryBuf = makeBuffer(query.data(), query.size());
    MTL::Buffer *paramsBuf = makeBuffer(&params, sizeof(Params));
    MTL::Buffer *blosumBuf = makeBuffer(blosumFlat.data(), blosumFlat.size() * sizeof(int16_t));
    MTL::Buffer *residueIndexBuf = makeBuffer(kResidueIndex.data(), kResidueIndex.size());
    MTL::Buffer *dbSeqsBuf = makeBuffer(dbConcat.data(), dbConcat.size());
    MTL::Buffer *dbOffsetsBuf = makeBuffer(offsets.data(), offsets.size() * sizeof(uint32_t));
    MTL::Buffer *dbLengthsBuf = makeBuffer(lengths.data(), lengths.size() * sizeof(uint32_t));
    MTL::Buffer *outBuf = device->newBuffer(dbRecords.size() * sizeof(int32_t),
                                             MTL::ResourceStorageModeShared);

    MTL::CommandQueue *queue = device->newCommandQueue();
    MTL::CommandBuffer *cmdBuf = queue->commandBuffer();
    MTL::ComputeCommandEncoder *encoder = cmdBuf->computeCommandEncoder();
    encoder->setComputePipelineState(pso);
    encoder->setBuffer(queryBuf, 0, 0);
    encoder->setBuffer(paramsBuf, 0, 1);
    encoder->setBuffer(blosumBuf, 0, 2);
    encoder->setBuffer(residueIndexBuf, 0, 3);
    encoder->setBuffer(dbSeqsBuf, 0, 4);
    encoder->setBuffer(dbOffsetsBuf, 0, 5);
    encoder->setBuffer(dbLengthsBuf, 0, 6);
    encoder->setBuffer(outBuf, 0, 7);

    const NS::UInteger dbCount = dbRecords.size();
    const NS::UInteger maxThreads = pso->maxTotalThreadsPerThreadgroup();
    MTL::Size gridSize = MTL::Size::Make(dbCount, 1, 1);
    MTL::Size groupSize = MTL::Size::Make(std::min<NS::UInteger>(maxThreads, dbCount), 1, 1);
    encoder->dispatchThreads(gridSize, groupSize);
    encoder->endEncoding();

    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();

    std::vector<int> scores(dbRecords.size());
    const int32_t *outData = static_cast<const int32_t *>(outBuf->contents());
    for (size_t i = 0; i < scores.size(); ++i) scores[i] = outData[i];

    queryBuf->release();
    paramsBuf->release();
    blosumBuf->release();
    residueIndexBuf->release();
    dbSeqsBuf->release();
    dbOffsetsBuf->release();
    dbLengthsBuf->release();
    outBuf->release();
    queue->release();
    pso->release();
    fn->release();
    library->release();
    device->release();
    pool->release();

    return scores;
}

}  // namespace metalsw
