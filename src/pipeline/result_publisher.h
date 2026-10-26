#pragma once

// 结果发布（扩展机制方案第 5 节）：每帧结果写入命名共享内存，供任意语言的其他进程读取。
// 布局固定，字段偏移由 static_assert 锁定；修改布局必须提升 kResultsLayoutVersion 并更新方案 5.2。

#include "inference/postprocess.h"

#include <windows.h>

#include <cstdint>
#include <string>

namespace nr {

inline constexpr wchar_t kResultsMappingName[] = L"Local\\NeuralReflex.Results.v1";
inline constexpr wchar_t kResultsEventName0[] = L"Local\\NeuralReflex.Results.v1.Event0";
inline constexpr wchar_t kResultsEventName1[] = L"Local\\NeuralReflex.Results.v1.Event1";
inline constexpr uint32_t kResultsMagic = 0x5352524E;  // 字节序列 "NRRS"
inline constexpr uint32_t kResultsLayoutVersion = 1;
inline constexpr uint32_t kResultsMetadataCapacity = 64 * 1024;

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
};
static_assert(offsetof(ResultsHeader, frameSeq) == 16 && offsetof(ResultsHeader, metaSeq) == 24 &&
              offsetof(ResultsHeader, metaOffset) == 32 && offsetof(ResultsHeader, hostPid) == 44 &&
              offsetof(ResultsHeader, qpcFrequency) == 48 && sizeof(ResultsHeader) == 56);

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

struct PublishedFrame {
  uint64_t frameIndex = 0;
  int64_t acquireQpc = 0, resultQpc = 0;
  RECT region{};
};

class ResultPublisher {
 public:
  // 创建或接管共享内存与事件。另一个仍在运行的进程正在发布时抛出异常。
  ResultPublisher();
  ~ResultPublisher();
  ResultPublisher(const ResultPublisher&) = delete;
  ResultPublisher& operator=(const ResultPublisher&) = delete;

  void PublishMetadata(const std::string& utf8Json);
  // 完成线程调用：一次复制、一次事件操作，不等待读取方
  void Publish(const PublishedFrame& frame, const DetectionOutput& result);

 private:
  void Close();

  HANDLE mapping_ = nullptr;
  HANDLE events_[2] = {};
  ResultsHeader* header_ = nullptr;
  ResultsFrame* frame_ = nullptr;
  char* meta_ = nullptr;
};

}  // namespace nr
