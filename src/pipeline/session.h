#pragma once

// 一次检测会话：工作线程完成准备（设备、引擎、采集）后进入实时循环，覆盖层在独立线程呈现。
// 界面和命令行都通过 Snapshot() 轮询状态与每秒统计，不进入实时路径（设计方案 6.2）。

#include "common/stats.h"
#include "common/win_util.h"
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

  // 以下为调度实验项（设计方案 6.2、6.4、6.5），默认值由实测决定
  int inflight = 1;  // 同时在 GPU 上的帧数，1 或 2
  GpuPriority gpuPriority = GpuPriority::kNormal;
  bool mmcss = false;  // 采集与完成线程注册为 MMCSS "Games" 任务

  bool publishResults = false;  // 每帧结果写入共享内存供其他进程读取（扩展机制方案第 5 节）
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
  long gpuPriorityStatus = 0;  // 设置 GPU 调度优先级的 NTSTATUS
  int gpuPriorityEffective = -1;  // 读回的实际级别（GpuPriority 数值），-1 表示无法读取
  double elapsed = 0;
  double prepareMs = 0;
  std::string adapter;  // UTF-8
  int inputSize = 0;
  int numClasses = 0;
  int numKeypoints = 0;
  std::string decoder;  // 解码器名称（扩展机制方案 4.4）
  bool publishing = false;
  std::string publishError;  // 请求发布但未能发布的原因，UTF-8
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
