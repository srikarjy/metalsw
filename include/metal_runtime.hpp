#pragma once

#include <string>
#include <vector>

#include "fasta.hpp"

namespace metalsw {

// Runs the naive inter-sequence Smith-Waterman kernel on the GPU: one thread
// per database sequence. Returns one score per entry in dbRecords, same
// order. metallibPath must point to the compiled smith_waterman.metallib.
std::vector<int> runSmithWatermanGpu(const std::string &metallibPath, const std::string &query,
                                      const std::vector<FastaRecord> &dbRecords, int gapOpen,
                                      int gapExtend);

}  // namespace metalsw
