#pragma once

// Only declared when the build is configured with METALSW_USE_PARASAIL (see
// CMakeLists.txt) — Parasail is an optional cross-check/baseline dependency,
// not a hard requirement to build metalsw.
#ifdef METALSW_USE_PARASAIL

#    include <string>

namespace metalsw
{

// Local Smith-Waterman score via Parasail's parasail_sw, BLOSUM62. Same gap
// convention as the metalsw oracle (open inclusive of the first gap
// position, extend per additional position) — verified 0/6 mismatches
// against the CPU oracle (see PROJECT_BLUEPRINT.md, Stage 0).
int
parasailScore(const std::string &query, const std::string &db, int gapOpen, int gapExtend);

}  // namespace metalsw

#endif  // METALSW_USE_PARASAIL
