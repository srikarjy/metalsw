#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include "metal_runtime.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <algorithm>
#include <cstdio>
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

MTL::ComputePipelineState *makePipeline(MTL::Device *device, MTL::Library *library,
                                         const char *fnName) {
    MTL::Function *fn =
        library->newFunction(NS::String::string(fnName, NS::StringEncoding::UTF8StringEncoding));
    if (!fn) throw std::runtime_error(std::string("kernel function not found: ") + fnName);
    NS::Error *error = nullptr;
    MTL::ComputePipelineState *pso = device->newComputePipelineState(fn, &error);
    fn->release();
    if (!pso) {
        std::string msg = std::string("failed to create pipeline for ") + fnName;
        if (error) msg += std::string(" (") + error->localizedDescription()->utf8String() + ")";
        throw std::runtime_error(msg);
    }
    return pso;
}

void logPipelineStats(const char *label, MTL::ComputePipelineState *pso) {
    std::fprintf(stderr,
                 "%s: maxTotalThreadsPerThreadgroup=%llu threadExecutionWidth=%llu "
                 "staticThreadgroupMemoryLength=%llu\n",
                 label, (unsigned long long)pso->maxTotalThreadsPerThreadgroup(),
                 (unsigned long long)pso->threadExecutionWidth(),
                 (unsigned long long)pso->staticThreadgroupMemoryLength());
}

MTL::Size threadgroupSizeFor(MTL::ComputePipelineState *pso, NS::UInteger count) {
    // Round down to a multiple of the SIMD width where possible, capped by both
    // the pipeline's max and the actual dispatch count.
    const NS::UInteger width = pso->threadExecutionWidth();
    const NS::UInteger maxThreads = pso->maxTotalThreadsPerThreadgroup();
    NS::UInteger size = std::min<NS::UInteger>(maxThreads, count);
    if (width > 0 && size > width) size = (size / width) * width;
    if (size == 0) size = std::min<NS::UInteger>(maxThreads, count);
    return MTL::Size::Make(size, 1, 1);
}

}  // namespace

std::vector<int> runSmithWatermanGpu(const std::string &metallibPath, const std::string &query,
                                      const std::vector<FastaRecord> &dbRecords, int gapOpen,
                                      int gapExtend) {
    if (query.size() > kMaxQueryLen) {
        throw std::runtime_error("query longer than MAX_QUERY_LEN (512) in kernel");
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

    MTL::ComputePipelineState *psoInt8 = makePipeline(device, library, "smith_waterman_score_int8");
    MTL::ComputePipelineState *psoInt16 = makePipeline(device, library, "smith_waterman_score");
    logPipelineStats("int8 kernel ", psoInt8);
    logPipelineStats("int16 kernel", psoInt16);

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

    std::vector<int16_t> profile = buildQueryProfile(query);

    auto makeBuffer = [&](const void *data, size_t length) {
        return device->newBuffer(data, length, MTL::ResourceStorageModeShared);
    };

    MTL::Buffer *profileBuf = makeBuffer(profile.data(), profile.size() * sizeof(int16_t));
    MTL::Buffer *residueIndexBuf = makeBuffer(kResidueIndex.data(), kResidueIndex.size());
    MTL::Buffer *dbSeqsBuf = makeBuffer(dbConcat.data(), dbConcat.size());

    MTL::CommandQueue *queue = device->newCommandQueue();

    // --- Pass 1: fast INT8 kernel over all sequences. ---
    Params paramsAll{static_cast<uint32_t>(query.size()), static_cast<uint32_t>(dbRecords.size()),
                      gapOpen, gapExtend};
    MTL::Buffer *paramsAllBuf = makeBuffer(&paramsAll, sizeof(Params));
    MTL::Buffer *dbOffsetsBuf = makeBuffer(offsets.data(), offsets.size() * sizeof(uint32_t));
    MTL::Buffer *dbLengthsBuf = makeBuffer(lengths.data(), lengths.size() * sizeof(uint32_t));
    MTL::Buffer *outInt8Buf = device->newBuffer(dbRecords.size() * sizeof(int32_t),
                                                 MTL::ResourceStorageModeShared);
    MTL::Buffer *overflowBuf =
        device->newBuffer(dbRecords.size() * sizeof(uint8_t), MTL::ResourceStorageModeShared);

    {
        MTL::CommandBuffer *cmdBuf = queue->commandBuffer();
        MTL::ComputeCommandEncoder *encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(psoInt8);
        encoder->setBuffer(paramsAllBuf, 0, 0);
        encoder->setBuffer(profileBuf, 0, 1);
        encoder->setBuffer(residueIndexBuf, 0, 2);
        encoder->setBuffer(dbSeqsBuf, 0, 3);
        encoder->setBuffer(dbOffsetsBuf, 0, 4);
        encoder->setBuffer(dbLengthsBuf, 0, 5);
        encoder->setBuffer(outInt8Buf, 0, 6);
        encoder->setBuffer(overflowBuf, 0, 7);

        const NS::UInteger dbCount = dbRecords.size();
        MTL::Size gridSize = MTL::Size::Make(dbCount, 1, 1);
        MTL::Size groupSize = threadgroupSizeFor(psoInt8, dbCount);
        encoder->dispatchThreads(gridSize, groupSize);
        encoder->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
    }

    std::vector<int> scores(dbRecords.size());
    const int32_t *int8Scores = static_cast<const int32_t *>(outInt8Buf->contents());
    const uint8_t *overflowFlags = static_cast<const uint8_t *>(overflowBuf->contents());

    std::vector<uint32_t> fallbackIdx;
    for (size_t i = 0; i < dbRecords.size(); ++i) {
        if (overflowFlags[i]) {
            fallbackIdx.push_back(static_cast<uint32_t>(i));
        } else {
            scores[i] = int8Scores[i];
        }
    }
    std::fprintf(stderr, "int8 pass: %zu/%zu sequences overflowed, falling back to int16\n",
                 fallbackIdx.size(), dbRecords.size());

    // --- Pass 2: INT16 fallback kernel, only over the sequences that overflowed. ---
    if (!fallbackIdx.empty()) {
        std::vector<uint32_t> fbOffsets(fallbackIdx.size());
        std::vector<uint32_t> fbLengths(fallbackIdx.size());
        for (size_t k = 0; k < fallbackIdx.size(); ++k) {
            fbOffsets[k] = offsets[fallbackIdx[k]];
            fbLengths[k] = lengths[fallbackIdx[k]];
        }

        Params paramsFb{static_cast<uint32_t>(query.size()),
                         static_cast<uint32_t>(fallbackIdx.size()), gapOpen, gapExtend};
        MTL::Buffer *paramsFbBuf = makeBuffer(&paramsFb, sizeof(Params));
        MTL::Buffer *fbOffsetsBuf = makeBuffer(fbOffsets.data(), fbOffsets.size() * sizeof(uint32_t));
        MTL::Buffer *fbLengthsBuf = makeBuffer(fbLengths.data(), fbLengths.size() * sizeof(uint32_t));
        MTL::Buffer *outInt16Buf = device->newBuffer(fallbackIdx.size() * sizeof(int32_t),
                                                      MTL::ResourceStorageModeShared);

        MTL::CommandBuffer *cmdBuf = queue->commandBuffer();
        MTL::ComputeCommandEncoder *encoder = cmdBuf->computeCommandEncoder();
        encoder->setComputePipelineState(psoInt16);
        encoder->setBuffer(paramsFbBuf, 0, 0);
        encoder->setBuffer(profileBuf, 0, 1);
        encoder->setBuffer(residueIndexBuf, 0, 2);
        encoder->setBuffer(dbSeqsBuf, 0, 3);
        encoder->setBuffer(fbOffsetsBuf, 0, 4);
        encoder->setBuffer(fbLengthsBuf, 0, 5);
        encoder->setBuffer(outInt16Buf, 0, 6);

        const NS::UInteger fbCount = fallbackIdx.size();
        MTL::Size gridSize = MTL::Size::Make(fbCount, 1, 1);
        MTL::Size groupSize = threadgroupSizeFor(psoInt16, fbCount);
        encoder->dispatchThreads(gridSize, groupSize);
        encoder->endEncoding();
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();

        const int32_t *fbScores = static_cast<const int32_t *>(outInt16Buf->contents());
        for (size_t k = 0; k < fallbackIdx.size(); ++k) {
            scores[fallbackIdx[k]] = fbScores[k];
        }

        paramsFbBuf->release();
        fbOffsetsBuf->release();
        fbLengthsBuf->release();
        outInt16Buf->release();
    }

    profileBuf->release();
    residueIndexBuf->release();
    dbSeqsBuf->release();
    paramsAllBuf->release();
    dbOffsetsBuf->release();
    dbLengthsBuf->release();
    outInt8Buf->release();
    overflowBuf->release();
    queue->release();
    psoInt8->release();
    psoInt16->release();
    library->release();
    device->release();
    pool->release();

    return scores;
}

}  // namespace metalsw
