#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nr {

struct Percentiles {
  size_t count = 0;
  double p50 = 0, p95 = 0, p99 = 0, max = 0;
};

// 长时间运行用的固定内存直方图：0.01 ms 分辨率，200 ms 以上并入最后一格（最大值单独记录）。
class LatencyHistogram {
 public:
  static constexpr double kBinMs = 0.01;
  static constexpr size_t kBins = 20000;

  LatencyHistogram() : bins_(kBins, 0) {}

  void Add(double ms) {
    const size_t bin = ms <= 0 ? 0 : std::min(kBins - 1, static_cast<size_t>(ms / kBinMs));
    ++bins_[bin];
    ++count_;
    max_ = std::max(max_, ms);
  }

  Percentiles Get() const {
    Percentiles r;
    r.count = count_;
    if (count_ == 0) return r;
    auto at = [&](double q) {
      const uint64_t rank = static_cast<uint64_t>(q * (count_ - 1) + 0.5);
      uint64_t seen = 0;
      for (size_t i = 0; i < kBins; ++i) {
        seen += bins_[i];
        if (seen > rank) return std::min((i + 0.5) * kBinMs, max_);
      }
      return max_;
    };
    r.p50 = at(0.50);
    r.p95 = at(0.95);
    r.p99 = at(0.99);
    r.max = max_;
    return r;
  }

 private:
  std::vector<uint32_t> bins_;
  uint64_t count_ = 0;
  double max_ = 0;
};

inline Percentiles ComputePercentiles(std::vector<double> values) {
  Percentiles r;
  r.count = values.size();
  if (values.empty()) return r;
  std::sort(values.begin(), values.end());
  auto at = [&](double q) {
    return values[std::min(values.size() - 1, static_cast<size_t>(q * (values.size() - 1) + 0.5))];
  };
  r.p50 = at(0.50);
  r.p95 = at(0.95);
  r.p99 = at(0.99);
  r.max = values.back();
  return r;
}

}  // namespace nr
