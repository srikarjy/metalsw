// Parasail CPU baseline benchmark (v0.4). Same corpus, same gap penalties,
// same warm-up/measured-iteration/mean+-stddev methodology as
// bench_main.cpp's GPU harness, so the two results/*.txt reports are a
// same-machine, same-methodology comparison point per
// PROJECT_BLUEPRINT.md's Stage 2 gate.
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fasta.hpp"
#include "parasail_score.hpp"
#include "stats.hpp"
#include "timing.hpp"

#ifndef METALSW_USE_PARASAIL
#    error "metalsw_parasail_bench requires -DMETALSW_USE_PARASAIL=ON"
#endif

namespace
{

constexpr int kWarmupIterations = 3;  // CPU has no cold-start clock ramp like the GPU; a few
                                      // iterations just settle cache/branch-predictor state.
constexpr int kDefaultMeasuredIterations = 15;  // >= 10 per the blueprint's gate

std::string
timestamp()
{
    std::time_t t = std::time(nullptr);
    char        buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    return buf;
}

double
runOnce(const std::string                       &query,
        const std::vector<metalsw::FastaRecord> &dbRecords,
        int                                      gapOpen,
        int                                      gapExtend)
{
    metalsw::Timer timer;
    for (const auto &rec : dbRecords)
    {
        metalsw::parasailScore(query, rec.sequence, gapOpen, gapExtend);
    }
    return timer.elapsedSeconds();
}

}  // namespace

int
main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: %s <query.fasta> <db.fasta> [measured_iters]\n", argv[0]);
        return 1;
    }

    const std::string queryPath     = argv[1];
    const std::string dbPath        = argv[2];
    const int         measuredIters = argc > 3 ? std::atoi(argv[3]) : kDefaultMeasuredIterations;
    const int         gapOpen       = 11;
    const int         gapExtend     = 1;

    auto               queryRecords = metalsw::parseFasta(queryPath);
    auto               dbRecords    = metalsw::parseFasta(dbPath);
    const std::string &query        = queryRecords.front().sequence;

    uint64_t totalDbResidues = 0;
    for (const auto &rec : dbRecords)
        totalDbResidues += rec.sequence.size();

    std::fprintf(stderr, "warm-up: %d iterations (discarded)\n", kWarmupIterations);
    double coldStartGcups = 0.0;
    for (int i = 0; i < kWarmupIterations; ++i)
    {
        const double elapsed = runOnce(query, dbRecords, gapOpen, gapExtend);
        const double gcups   = metalsw::gcups(query.size(), totalDbResidues, elapsed);
        if (i == 0)
            coldStartGcups = gcups;
    }

    std::fprintf(stderr, "measured: %d iterations\n", measuredIters);
    std::filesystem::create_directories("results");
    const std::string tag     = timestamp();
    const std::string csvPath = "results/parasail_benchmark_" + tag + "_iterations.csv";
    std::ofstream     csv(csvPath);
    if (!csv)
        throw std::runtime_error("failed to open " + csvPath + " for writing");
    csv << "iteration,elapsed_since_start_s,iter_seconds,gcups\n";

    metalsw::Timer        wallClock;
    metalsw::RunningStats gcupsStats;
    for (int i = 0; i < measuredIters; ++i)
    {
        const double iterSeconds = runOnce(query, dbRecords, gapOpen, gapExtend);
        const double gcups       = metalsw::gcups(query.size(), totalDbResidues, iterSeconds);
        gcupsStats.add(gcups);
        csv << i << "," << wallClock.elapsedSeconds() << "," << iterSeconds << "," << gcups << "\n";
    }
    csv.close();

    const std::string resultsPath = "results/parasail_benchmark_" + tag + ".txt";
    std::ofstream     out(resultsPath);
    if (!out)
        throw std::runtime_error("failed to open " + resultsPath + " for writing");
    auto report = [&](std::ostream &os)
    {
        os << "metalsw Parasail CPU baseline benchmark\n";
        os << "query: " << queryPath << " (" << query.size() << " residues)\n";
        os << "corpus: " << dbPath << " (" << dbRecords.size() << " sequences, " << totalDbResidues
           << " total residues)\n";
        os << "gap penalties: open=" << gapOpen << " extend=" << gapExtend << "\n";
        os << "warm-up iterations (discarded): " << kWarmupIterations << "\n";
        os << "cold-start GCUPS (1st warm-up iteration): " << coldStartGcups << "\n";
        os << "measured iterations: " << measuredIters << "\n";
        os << "steady-state GCUPS: mean=" << gcupsStats.mean() << " stddev=" << gcupsStats.stddev()
           << "\n";
        os << "per-iteration log: " << csvPath << "\n";
    };
    report(out);
    report(std::cerr);

    std::fprintf(stderr, "\nwrote %s and %s\n", resultsPath.c_str(), csvPath.c_str());
    return 0;
}
