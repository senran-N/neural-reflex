#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace nr {

struct Percentiles {
  size_t count = 0;
  double p50 = 0, p95 = 0, p99 = 0, max = 0;
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
