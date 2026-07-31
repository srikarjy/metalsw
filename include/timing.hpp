#pragma once

#include <chrono>
#include <cstdint>

namespace metalsw
{

class Timer
{
  public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    void reset() { start_ = std::chrono::steady_clock::now(); }

    double elapsedSeconds() const
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }

  private:
    std::chrono::steady_clock::time_point start_;
};

// GCUPS = (queryLen * totalDbResidues) / elapsedSeconds / 1e9
inline double
gcups(uint64_t queryLen, uint64_t totalDbResidues, double elapsedSeconds)
{
    if (elapsedSeconds <= 0.0)
        return 0.0;
    return (static_cast<double>(queryLen) * static_cast<double>(totalDbResidues)) / elapsedSeconds
           / 1e9;
}

}  // namespace metalsw
