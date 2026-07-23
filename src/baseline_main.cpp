#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "sw_reference.hpp"
#include "timing.hpp"

#ifdef METALSW_USE_PARASAIL
#include <parasail.h>
#include <parasail/matrices/blosum62.h>
#endif

namespace {

struct ScoredHit {
    std::string id;
    int score;
};

#ifdef METALSW_USE_PARASAIL
int parasailScore(const std::string &query, const std::string &db, int gapOpen, int gapExtend) {
    parasail_result_t *result = parasail_sw(query.c_str(), static_cast<int>(query.size()),
                                             db.c_str(), static_cast<int>(db.size()), gapOpen,
                                             gapExtend, &parasail_blosum62);
    int score = result->score;
    parasail_result_free(result);
    return score;
}
#endif

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <query.fasta> <db.fasta> [topN]\n", argv[0]);
        return 1;
    }

    const std::string queryPath = argv[1];
    const std::string dbPath = argv[2];
    const int topN = argc > 3 ? std::atoi(argv[3]) : 10;
    const int gapOpen = 11;
    const int gapExtend = 1;

    auto queryRecords = metalsw::parseFasta(queryPath);
    auto dbRecords = metalsw::parseFasta(dbPath);
    const std::string &query = queryRecords.front().sequence;

    std::vector<ScoredHit> hits;
    hits.reserve(dbRecords.size());

    uint64_t totalDbResidues = 0;
    metalsw::Timer timer;
    int mismatches = 0;
    for (const auto &rec : dbRecords) {
        const int score = metalsw::smithWatermanScore(query, rec.sequence, gapOpen, gapExtend);
        hits.push_back({rec.id, score});
        totalDbResidues += rec.sequence.size();

#ifdef METALSW_USE_PARASAIL
        const int refScore = parasailScore(query, rec.sequence, gapOpen, gapExtend);
        if (refScore != score) {
            std::fprintf(stderr, "MISMATCH %s: oracle=%d parasail=%d\n", rec.id.c_str(), score,
                         refScore);
            ++mismatches;
        }
#endif
    }
    const double elapsed = timer.elapsedSeconds();

    std::sort(hits.begin(), hits.end(),
              [](const ScoredHit &a, const ScoredHit &b) { return a.score > b.score; });

    const int printCount = std::min<int>(topN, static_cast<int>(hits.size()));
    for (int i = 0; i < printCount; ++i) {
        std::printf("%-20s %d\n", hits[i].id.c_str(), hits[i].score);
    }

    std::fprintf(stderr, "\n%zu sequences, %.4fs, %.4f GCUPS\n", dbRecords.size(), elapsed,
                 metalsw::gcups(query.size(), totalDbResidues, elapsed));

#ifdef METALSW_USE_PARASAIL
    std::fprintf(stderr, "%d/%zu mismatches vs parasail\n", mismatches, dbRecords.size());
    if (mismatches > 0) return 1;
#endif

    return 0;
}
