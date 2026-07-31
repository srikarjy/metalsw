#pragma once

#include <memory>
#include <string>
#include <vector>

#include "fasta.hpp"

namespace metalsw {

// Owns the one-time GPU setup cost (device, compiled library, both compute
// pipelines) so repeated benchmark iterations only pay for buffer
// packing/dispatch/readback, not device/pipeline creation each time.
class GpuRunner {
public:
    explicit GpuRunner(const std::string &metallibPath);
    ~GpuRunner();
    GpuRunner(const GpuRunner &) = delete;
    GpuRunner &operator=(const GpuRunner &) = delete;

    // Runs the inter-sequence Smith-Waterman kernel (one thread per DB
    // sequence, INT8 fast path with INT16 overflow fallback). Returns one
    // score per entry in dbRecords, same order.
    std::vector<int> run(const std::string &query, const std::vector<FastaRecord> &dbRecords,
                          int gapOpen, int gapExtend);

    // Runs the wavefront (anti-diagonal) Smith-Waterman kernel.
    // One threadgroup per database sequence, 32 threads cooperating.
    // Returns one score per entry in dbRecords, same order.
    std::vector<int> runWavefront(const std::string &query, const std::vector<FastaRecord> &dbRecords,
                                   int gapOpen, int gapExtend);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace metalsw
