#pragma once

#include <cstdint>
#include <mutex>

namespace nr {

// 只保留最新值的交接位置：新值直接替换未被取走的旧值，不排队（设计方案 6.1）。
template <class T>
class LatestSlot {
 public:
  void Publish(const T& value) {
    std::lock_guard lock(mutex_);
    value_ = value;
    ++sequence_;
  }

  // 有比 lastSeen 更新的值时取出并返回 true。
  bool TakeNewer(uint64_t& lastSeen, T& out) const {
    std::lock_guard lock(mutex_);
    if (sequence_ == lastSeen) return false;
    lastSeen = sequence_;
    out = value_;
    return true;
  }

 private:
  mutable std::mutex mutex_;
  T value_{};
  uint64_t sequence_ = 0;
};

}  // namespace nr
