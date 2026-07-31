#include <iostream>
#include <string>
#include "sw_reference.hpp"

int main() {
    using namespace metalsw;

    std::string query = "PLEKHS";
    std::string target1 = "PLEKHS";   // identical
    std::string target2 = "PLEKHA";   // mutation
    std::string target3 = "PLEK";     // partial
    std::string target4 = "ABCDEF";   // unrelated

    int gapOpen = 11, gapExtend = 1;

    std::cout << "Testing traceback...\n\n";

    auto test = [&](const std::string& q, const std::string& t, const char* name) {
        std::cout << "=== " << name << " ===\n";
        auto result = smithWatermanTraceback(q, t, 11, 1);
        std::cout << "Score: " << result.score << "\n";
        std::cout << "Query:   " << result.aligned_query << "\n";
        std::cout << "Target:  " << result.aligned_target << "\n";
        std::cout << "Query coords: [" << result.query_start << ", " << result.query_end << "]\n";
        std::cout << "Target coords: [" << result.target_start << ", " << result.target_end << "]\n";
        std::cout << "\n";

        // Verify score matches score-only function
        int score_only = smithWatermanScore(q, t, 11, 1);
        if (score_only != result.score) {
            std::cerr << "MISMATCH: score-only=" << score_only << " traceback=" << result.score << "\n";
        }
    };

    test(query, target1, "Identical");
    test(query, target2, "Mutation");
    test(query, target3, "Partial");
    test(query, target4, "Unrelated");

    return 0;
}