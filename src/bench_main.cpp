// GPU benchmark harness (v0.4). Constructs one GpuRunner (one-time device/
// pipeline setup, excluded from all timing), runs a fixed warm-up phase
// (discarded, reported separately as "cold start"), then >=10 measured
// iterations whose GCUPS are fed to RunningStats for mean+-stddev — per
// PROJECT_BLUEPRINT.md's v0.4 requirement to report steady-state separately
// from cold-start rather than a single one-off number.
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
#include "metal_runtime.hpp"
#include "stats.hpp"
#include "timing.hpp"

namespace
{

constexpr int kWarmupIterations          = 20;
constexpr int kDefaultMeasuredIterations = 15;  // >= 10 per the blueprint's gate

std::string
timestamp()
{
    std::time_t t = std::time(nullptr);
    char        buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t));
    return buf;
}

}  // namespace

int
main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr,
                     "usage: %s <query.fasta> <db.fasta> [measured_iters] [metallib_path]\n",
                     argv[0]);
        return 1;
    }

    const std::string queryPath     = argv[1];
    const std::string dbPath        = argv[2];
    const int         measuredIters = argc > 3 ? std::atoi(argv[3]) : kDefaultMeasuredIterations;
    const std::string metallibPath  = argc > 4 ? argv[4] : "smith_waterman.metallib";
    const int         gapOpen       = 11;
    const int         gapExtend     = 1;

    auto               queryRecords = metalsw::parseFasta(queryPath);
    auto               dbRecords    = metalsw::parseFasta(dbPath);
    const std::string &query        = queryRecords.front().sequence;

    uint64_t totalDbResidues = 0;
    for (const auto &rec : dbRecords)
        totalDbResidues += rec.sequence.size();

    metalsw::GpuRunner runner(metallibPath);  // one-time setup, not timed

    std::fprintf(stderr,
                 "warm-up: %d iterations (discarded, settles thermal/clock state)\n",
                 kWarmupIterations);
    double coldStartGcups = 0.0;
    for (int i = 0; i < kWarmupIterations; ++i)
    {
        metalsw::Timer timer;
        runner.run(query, dbRecords, gapOpen, gapExtend);
        const double gcups = metalsw::gcups(query.size(), totalDbResidues, timer.elapsedSeconds());
        if (i == 0)
            coldStartGcups = gcups;  // first iteration = cold start
    }

    std::fprintf(stderr, "measured: %d iterations\n", measuredIters);
    std::filesystem::create_directories("results");
    const std::string tag     = timestamp();
    const std::string csvPath = "results/benchmark_" + tag + "_iterations.csv";
    std::ofstream     csv(csvPath);
    if (!csv)
        throw std::runtime_error("failed to open " + csvPath + " for writing");
    csv << "iteration,elapsed_since_start_s,iter_seconds,gcups\n";

    metalsw::Timer wallClock;  // elapsed since the measured phase began, for the thermal curve
    metalsw::RunningStats gcupsStats;
    for (int i = 0; i < measuredIters; ++i)
    {
        metalsw::Timer timer;
        runner.run(query, dbRecords, gapOpen, gapExtend);
        const double iterSeconds = timer.elapsedSeconds();
        const double gcups       = metalsw::gcups(query.size(), totalDbResidues, iterSeconds);
        gcupsStats.add(gcups);
        csv << i << "," << wallClock.elapsedSeconds() << "," << iterSeconds << "," << gcups << "\n";
    }
    csv.close();

    const std::string resultsPath = "results/benchmark_" + tag + ".txt";
    std::ofstream     out(resultsPath);
    if (!out)
        throw std::runtime_error("failed to open " + resultsPath + " for writing");
    auto report = [&](std::ostream &os)
    {
        os << "metalsw GPU benchmark\n";
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
