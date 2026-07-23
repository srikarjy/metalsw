#include "sw_reference.hpp"

#include <algorithm>
#include <vector>

#include "blosum62.hpp"

namespace metalsw {

int smithWatermanScore(const std::string &seqA, const std::string &seqB,
                        int gapOpen, int gapExtend) {
    const size_t n = seqA.size();
    const size_t m = seqB.size();

    std::vector<int> prevH(m + 1, 0), prevF(m + 1, 0);
    std::vector<int> curH(m + 1, 0), curE(m + 1, 0), curF(m + 1, 0);

    int best = 0;
    for (size_t i = 1; i <= n; ++i) {
        curH[0] = 0;
        curE[0] = 0;
        curF[0] = 0;
        int diag = prevH[0];
        for (size_t j = 1; j <= m; ++j) {
            const int matchScore = diag + blosumScore(seqA[i - 1], seqB[j - 1]);
            curE[j] = std::max(curH[j - 1] - gapOpen, curE[j - 1] - gapExtend);
            curF[j] = std::max(prevH[j] - gapOpen, prevF[j] - gapExtend);
            curH[j] = std::max({0, matchScore, curE[j], curF[j]});
            best = std::max(best, curH[j]);
            diag = prevH[j];
        }
        std::swap(prevH, curH);
        std::swap(prevF, curF);
    }
    return best;
}

}  // namespace metalsw
