#pragma once

// 一次检测会话：工作线程完成准备（设备、引擎、采集）后进入实时循环，覆盖层在独立线程呈现。
// 界面和命令行都通过 Snapshot() 轮询状态与每秒统计，不进入实时路径（设计方案 6.2）。

#include "common/stats.h"
#include "inference/detector.h"

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nr {

enum class CaptureMode { kWgc, kDuplication };

struct SessionConfig {
  HWND window = nullptr;       // 设置时以该窗口客户区为检测区域
  HMONITOR monitor = nullptr;  // window 为空时检测整个显示器
  CaptureMode mode = CaptureMode::kWgc;
  DetectorOptions detector;
  bool overlay = true;
  double staleMs = 100;  // 超过该时间没有新结果，覆盖层清除旧框
};

enum class SessionPhase { kIdle, kPreparing, kRunning, kFailed };

// 一秒内的汇总
struct SessionSecond {
  double t = 0;  // 会话开始后的秒数
  double detectionsPerSec = 0;
  double overlayPerSec = 0;
  Percentiles latency;  // 取得画面 → 结果可用
  double preMs = 0, inferMs = 0, postMs = 0;  // GPU 阶段中位数
  double boxes = 0;  // 平均每次检测的框数
  uint32_t overflowFrames = 0;
};

struct SessionSnapshot {
  SessionPhase phase = SessionPhase::kIdle;
  std::string error;  // UTF-8
  double elapsed = 0;
  double prepareMs = 0;
  std::string adapter;  // UTF-8
  int inputSize = 0;
  int numClasses = 0;
  uint64_t detections = 0;
  Percentiles latency, pre, infer, post;  // 整个会话
  uint64_t overflowFrames = 0;
  uint64_t vramUsedBytes = 0, vramBudgetBytes = 0;  // 本进程本地显存占用与预算
  std::vector<SessionSecond> seconds;  // 最近的若干秒，旧在前
  std::map<int, int> lastClasses;      // 最近一次检测的类别 → 框数
};

class Session {
 public:
  Session() = default;
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  // 已在运行时先停止。准备在工作线程进行，进度见 Snapshot().phase。
  void Start(const SessionConfig& config);
  void Stop();
  bool Active() const;

  SessionSnapshot Snapshot(size_t maxSeconds = 120) const;

  // 运行中调整阈值，下一帧起生效，无需重新准备
  void SetThresholds(float conf, float iou);

 private:
  void Run(SessionConfig config);
  void Fail(const std::string& message);

  std::thread worker_;
  std::atomic<bool> stop_{false};
  std::atomic<float> conf_{0.25f}, iou_{0.7f};

  mutable std::mutex mutex_;
  SessionSnapshot state_;
  std::deque<SessionSecond> seconds_;
  LatencyHistogram latency_, pre_, infer_, post_;
};

const char* PhaseName(SessionPhase phase);

}  // namespace nr
