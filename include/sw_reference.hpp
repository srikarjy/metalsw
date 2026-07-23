#pragma once

#include <string>

namespace metalsw {

// Scalar Gotoh affine-gap Smith-Waterman, score-only, O(min(m,n)) space.
// Gap cost of length L = gapOpen + (L - 1) * gapExtend (first-position-inclusive convention).
int smithWatermanScore(const std::string &seqA, const std::string &seqB,
                        int gapOpen = 11, int gapExtend = 1);

}  // namespace metalsw
