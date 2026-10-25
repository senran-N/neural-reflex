#pragma once

#include <windows.h>

#include <cstdint>

namespace nr {

// 全程统一使用 QPC 作为主机单调时钟（设计方案 7.1）。
inline int64_t QpcNow() {
  LARGE_INTEGER v;
  QueryPerformanceCounter(&v);
  return v.QuadPart;
}

inline int64_t QpcFrequency() {
  static const int64_t freq = [] {
    LARGE_INTEGER v;
    QueryPerformanceFrequency(&v);
    return v.QuadPart;
  }();
  return freq;
}

inline double QpcToMs(int64_t ticks) {
  return static_cast<double>(ticks) * 1000.0 / static_cast<double>(QpcFrequency());
}

// WinRT 的 SystemRelativeTime 以 100ns 为单位，由 QPC 换算而来。
// 拆成整秒与余数两部分换算，避免开机时间较长时 hns * freq 溢出。
inline int64_t Hns100ToQpc(int64_t hns) {
  constexpr int64_t kHnsPerSecond = 10'000'000;
  const int64_t freq = QpcFrequency();
  if (freq == kHnsPerSecond) return hns;
  return (hns / kHnsPerSecond) * freq + (hns % kHnsPerSecond) * freq / kHnsPerSecond;
}

}  // namespace nr
