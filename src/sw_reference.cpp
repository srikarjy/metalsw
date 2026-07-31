#include "sw_reference.hpp"

#include <algorithm>
#include <vector>

#include "blosum62.hpp"

namespace metalsw
{

int
smithWatermanScore(const std::string &seqA, const std::string &seqB, int gapOpen, int gapExtend)
{
    const size_t n = seqA.size();
    const size_t m = seqB.size();

    std::vector<int> prevH(m + 1, 0), prevF(m + 1, 0);
    std::vector<int> curH(m + 1, 0), curE(m + 1, 0), curF(m + 1, 0);

    int best = 0;
    for (size_t i = 1; i <= n; ++i)
    {
        curH[0]  = 0;
        curE[0]  = 0;
        curF[0]  = 0;
        int diag = prevH[0];
        for (size_t j = 1; j <= m; ++j)
        {
            const int matchScore = diag + blosumScore(seqA[i - 1], seqB[j - 1]);
            curE[j]              = std::max(curH[j - 1] - gapOpen, curE[j - 1] - gapExtend);
            curF[j]              = std::max(prevH[j] - gapOpen, prevF[j] - gapExtend);
            curH[j]              = std::max({0, matchScore, curE[j], curF[j]});
            best                 = std::max(best, curH[j]);
            diag                 = prevH[j];
        }
        std::swap(prevH, curH);
        std::swap(prevF, curF);
    }
    return best;
}

TracebackResult
smithWatermanTraceback(const std::string &query,
                       const std::string &target,
                       int                gapOpen,
                       int                gapExtend)
{
    const size_t n = query.size();
    const size_t m = target.size();

    // Full DP matrices for traceback
    std::vector<std::vector<int>> H(n + 1, std::vector<int>(m + 1, 0));
    std::vector<std::vector<int>> E(n + 1, std::vector<int>(m + 1, 0));
    std::vector<std::vector<int>> F(n + 1, std::vector<int>(m + 1, 0));
    // Traceback pointers: 0=stop, 1=diag, 2=up (F), 3=left (E)
    std::vector<std::vector<uint8_t>> trace(n + 1, std::vector<uint8_t>(m + 1, 0));

    int    best   = 0;
    size_t best_i = 0, best_j = 0;

    for (size_t i = 1; i <= n; ++i)
    {
        H[i][0]     = 0;
        E[i][0]     = 0;
        F[i][0]     = 0;
        trace[i][0] = 0;
        for (size_t j = 1; j <= m; ++j)
        {
            const int matchScore = H[i - 1][j - 1] + blosumScore(query[i - 1], target[j - 1]);
            const int eVal       = std::max(H[i][j - 1] - gapOpen, E[i][j - 1] - gapExtend);
            const int fVal       = std::max(H[i - 1][j] - gapOpen, F[i - 1][j] - gapExtend);

            int     h   = matchScore;
            uint8_t dir = 1;  // 1=diag (match)

            if (h < eVal)
            {
                h   = eVal;
                dir = 3;
            }  // left (E)
            if (h < fVal)
            {
                h   = fVal;
                dir = 2;
            }  // up (F)
            if (h < 0)
            {
                h   = 0;
                dir = 0;
            }  // stop

            H[i][j]     = h;
            E[i][j]     = eVal;
            F[i][j]     = fVal;
            trace[i][j] = dir;

            if (h > H[best_i][best_j])
            {
                best_i = i;
                best_j = j;
            }
        }
    }

    // Traceback from best position
    std::string aligned_query, aligned_target;
    size_t      i = best_i, j = best_j;

    while (i > 0 && j > 0 && H[i][j] > 0)
    {
        uint8_t dir = trace[i][j];
        if (dir == 1)
        {  // diagonal (match)
            aligned_query.push_back(query[i - 1]);
            aligned_target.push_back(target[j - 1]);
            --i;
            --j;
        }
        else if (dir == 2)
        {  // up (gap in query / deletion in target)
            aligned_query.push_back('-');
            aligned_target.push_back(target[j - 1]);
            --i;
        }
        else if (dir == 3)
        {  // left (gap in target / insertion in query)
            aligned_query.push_back(query[i - 1]);
            aligned_target.push_back('-');
            --j;
        }
        else
        {  // stop (0)
            break;
        }
    }

    std::reverse(aligned_query.begin(), aligned_query.end());
    std::reverse(aligned_target.begin(), aligned_target.end());

    // Find start positions (0-based)
    size_t query_start  = i;
    size_t query_end    = best_i - 1;
    size_t target_start = j;
    size_t target_end   = best_j - 1;

    return {aligned_query,
            aligned_target,
            H[best_i][best_j],
            static_cast<int>(query_start),
            static_cast<int>(query_end),
            static_cast<int>(target_start),
            static_cast<int>(target_end)};
}

}  // namespace metalsw