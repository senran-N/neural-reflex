#pragma once

// 结果发布（扩展机制方案第 5 节、7.5）：每帧结果写入命名共享内存，供任意语言的其他进程读取。
// 布局固定，字段偏移由 static_assert 锁定；不兼容的修改必须提升 kResultsLayoutVersion 并更新方案 5.2。
// 环形缓冲是布局 1 的兼容扩展：旧读取方按头部给出的偏移读取最新帧，不受影响。

#include "common/detection.h"

#include <windows.h>

#include <cstdint>
#include <string>

namespace nr {

inline constexpr wchar_t kResultsMappingName[] = L"Local\\NeuralReflex.Results.v1";
inline constexpr uint32_t kResultsMagic = 0x5352524E;  // 字节序列 "NRRS"
inline constexpr uint32_t kResultsLayoutVersion = 1;
inline constexpr uint32_t kResultsMetadataCapacity = 64 * 1024;
inline constexpr uint32_t kResultsRingSlots = 128;  // 144 帧/秒时约 0.9 s

struct ResultsHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t totalBytes;
  uint32_t frameOffset;
  volatile int64_t frameSeq;  // 序号锁：奇数表示正在写入；已发布帧数 = frameSeq / 2
  volatile int64_t metaSeq;
  uint32_t metaOffset;
  uint32_t metaCapacity;
  uint32_t metaLength;
  uint32_t hostPid;
  int64_t qpcFrequency;
  uint32_t ringOffset;  // 环形区偏移，0 表示没有（7.5）
  uint32_t reserved;
};
static_assert(offsetof(ResultsHeader, frameSeq) == 16 && offsetof(ResultsHeader, metaSeq) == 24 &&
              offsetof(ResultsHeader, metaOffset) == 32 && offsetof(ResultsHeader, hostPid) == 44 &&
              offsetof(ResultsHeader, qpcFrequency) == 48 && offsetof(ResultsHeader, ringOffset) == 56 &&
              sizeof(ResultsHeader) == 64);

struct ResultsFrame {
  uint64_t frameIndex;  // 会话内帧号，从 0 开始
  int64_t acquireQpc;
  int64_t resultQpc;
  int32_t regionX, regionY, regionW, regionH;  // 检测区域，屏幕物理像素
  int32_t count;
  int32_t numKeypoints;
  int32_t candidates;
  int32_t overflow;
  Detection detections[kMaxDetections];
  Keypoint keypoints[kMaxDetections * kMaxKeypoints];
};
static_assert(sizeof(Detection) == 24 && sizeof(Keypoint) == 12);
static_assert(offsetof(ResultsFrame, count) == 40 && offsetof(ResultsFrame, detections) == 56);

// 环形区开头 64 字节
struct ResultsRingHeader {
  uint32_t slots;
  uint32_t stride;        // 槽跨度（字节）
  uint32_t recordOffset;  // 槽内帧记录偏移
  uint32_t reserved[13];
};
static_assert(sizeof(ResultsRingHeader) == 64);

// 槽：第 k 次发布写入槽 (k - 1) % slots，写入期间 seq = 2k - 1，写完 seq = 2k
struct ResultsRingSlot {
  volatile int64_t seq;
  int64_t session;  // 帧所属会话号，与元数据中的 session 相同
  uint8_t reserved[48];
  ResultsFrame record;
};
static_assert(offsetof(ResultsRingSlot, record) == 64);

struct PublishedFrame {
  int64_t session = 0;  // 会话号，与元数据中的 session 相同
  uint64_t frameIndex = 0;
  int64_t acquireQpc = 0, resultQpc = 0;
  RECT region{};
};

class ResultPublisher {
 public:
  // 创建或接管共享内存与事件（名为 mapping、mapping.Event0、mapping.Event1）。
  // 另一个仍在运行的进程正在发布时抛出异常。测试工具可用其他名字，以免与正在运行的宿主冲突。
  explicit ResultPublisher(const std::wstring& mapping = kResultsMappingName);
  ~ResultPublisher();
  ResultPublisher(const ResultPublisher&) = delete;
  ResultPublisher& operator=(const ResultPublisher&) = delete;

  void PublishMetadata(const std::string& utf8Json);
  // 完成线程调用：两次复制（最新帧与环形槽，只复制有效部分）、一次事件操作，不等待读取方
  void Publish(const PublishedFrame& frame, const DetectionOutput& result);

 private:
  void Close();

  HANDLE mapping_ = nullptr;
  HANDLE events_[2] = {};
  ResultsHeader* header_ = nullptr;
  ResultsFrame* frame_ = nullptr;
  char* meta_ = nullptr;
  ResultsRingHeader* ring_ = nullptr;
};

}  // namespace nr
