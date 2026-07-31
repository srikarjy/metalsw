#pragma once

#include <cmath>
#include <cstddef>

namespace metalsw
{

// Running mean/variance via Welford's algorithm — numerically stable,
// doesn't require storing all samples.
class RunningStats
{
  public:
    void add(double value)
    {
        ++count_;
        const double delta = value - mean_;
        mean_ += delta / static_cast<double>(count_);
        const double delta2 = value - mean_;
        m2_ += delta * delta2;
    }

    size_t count() const { return count_; }
    double mean() const { return mean_; }

    // Sample standard deviation (Bessel's correction). 0 when fewer than 2 samples.
    double stddev() const
    {
        if (count_ < 2)
            return 0.0;
        return std::sqrt(m2_ / static_cast<double>(count_ - 1));
    }

  private:
    size_t count_ = 0;
    double mean_  = 0.0;
    double m2_    = 0.0;
};

}  // namespace metalsw
