// Validates buildQueryProfile() against blosumScore() before any Metal kernel
// consumes it. Pure C++, no Metal/Parasail dependency — runs in the Linux
// devcontainer as well as on macOS.
#include <cstdio>
#include <string>

#include "blosum62.hpp"

int
main()
{
    using namespace metalsw;

    const std::string queries[] = {
        "ARNDCQEGHILKMFPSTWYV",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG",
        "A",
        "WWWWWWWWWW",
    };

    int failures = 0;
    int checks   = 0;
    for (const auto &query : queries)
    {
        auto profile = buildQueryProfile(query);
        for (int r = 0; r < kAlphabetSize; ++r)
        {
            const char residueChar = kAlphabet[r];
            for (size_t j = 0; j < query.size(); ++j)
            {
                ++checks;
                const int16_t expected = static_cast<int16_t>(blosumScore(residueChar, query[j]));
                const int16_t actual   = profile[static_cast<size_t>(r) * query.size() + j];
                if (expected != actual)
                {
                    std::fprintf(stderr,
                                 "MISMATCH query=\"%s\" residue=%c pos=%zu expected=%d actual=%d\n",
                                 query.c_str(),
                                 residueChar,
                                 j,
                                 expected,
                                 actual);
                    ++failures;
                }
            }
        }
    }

    std::fprintf(stderr, "%d checks, %d failures\n", checks, failures);
    if (failures > 0)
        return 1;
    std::printf("PASS\n");
    return 0;
}
