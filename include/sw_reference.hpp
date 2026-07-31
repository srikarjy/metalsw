#pragma once

#include <string>
#include <vector>

namespace metalsw
{

struct TracebackResult
{
    std::string aligned_query;
    std::string aligned_target;
    int         score;
    int         query_start;   // 0-based start in query
    int         query_end;     // 0-based end in query (inclusive)
    int         target_start;  // 0-based start in target
    int         target_end;    // 0-based end in target (inclusive)
};

// Scalar Gotoh affine-gap Smith-Waterman, score-only, O(min(m,n)) space.
// Gap cost of length L = gapOpen + (L - 1) * gapExtend (first-position-inclusive convention).
int
smithWatermanScore(const std::string &seqA,
                   const std::string &seqB,
                   int                gapOpen   = 11,
                   int                gapExtend = 1);

// Full traceback for a single alignment. Returns the optimal local alignment
// with aligned sequences and coordinates. Uses O(m*n) space for traceback.
TracebackResult
smithWatermanTraceback(const std::string &query,
                       const std::string &target,
                       int                gapOpen   = 11,
                       int                gapExtend = 1);

}  // namespace metalsw
