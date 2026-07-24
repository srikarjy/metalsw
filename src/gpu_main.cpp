#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "metal_runtime.hpp"
#include "sw_reference.hpp"
#include "timing.hpp"

namespace {

struct ScoredHit {
    std::string id;
    int score;
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <query.fasta> <db.fasta> [topN] [metallib_path]\n",
                     argv[0]);
        return 1;
    }

    const std::string queryPath = argv[1];
    const std::string dbPath = argv[2];
    const int topN = argc > 3 ? std::atoi(argv[3]) : 10;
    const std::string metallibPath = argc > 4 ? argv[4] : "smith_waterman.metallib";
    const int gapOpen = 11;
    const int gapExtend = 1;

    auto queryRecords = metalsw::parseFasta(queryPath);
    auto dbRecords = metalsw::parseFasta(dbPath);
    const std::string &query = queryRecords.front().sequence;

    metalsw::GpuRunner runner(metallibPath);
    metalsw::Timer timer;
    std::vector<int> gpuScores = runner.run(query, dbRecords, gapOpen, gapExtend);
    const double elapsed = timer.elapsedSeconds();

    uint64_t totalDbResidues = 0;
    int mismatches = 0;
    std::vector<ScoredHit> hits;
    hits.reserve(dbRecords.size());
    for (size_t i = 0; i < dbRecords.size(); ++i) {
        const auto &rec = dbRecords[i];
        const int oracleScore = metalsw::smithWatermanScore(query, rec.sequence, gapOpen, gapExtend);
        const bool pass = oracleScore == gpuScores[i];
        std::fprintf(stderr, "%-20s gpu=%-6d oracle=%-6d %s\n", rec.id.c_str(), gpuScores[i],
                     oracleScore, pass ? "PASS" : "FAIL");
        if (!pass) ++mismatches;
        hits.push_back({rec.id, gpuScores[i]});
        totalDbResidues += rec.sequence.size();
    }

    std::sort(hits.begin(), hits.end(),
              [](const ScoredHit &a, const ScoredHit &b) { return a.score > b.score; });
    const int printCount = std::min<int>(topN, static_cast<int>(hits.size()));
    for (int i = 0; i < printCount; ++i) {
        std::printf("%-20s %d\n", hits[i].id.c_str(), hits[i].score);
    }

    std::fprintf(stderr, "\n%zu sequences, %.4fs, %.4f GCUPS\n", dbRecords.size(), elapsed,
                 metalsw::gcups(query.size(), totalDbResidues, elapsed));
    std::fprintf(stderr, "%d/%zu mismatches vs CPU oracle\n", mismatches, dbRecords.size());

    return mismatches > 0 ? 1 : 0;
}
