#include "pipeline/result_publisher.h"

#include "common/qpc.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace nr {
namespace {

constexpr uint32_t Align64(size_t v) { return static_cast<uint32_t>((v + 63) & ~size_t{63}); }
constexpr uint32_t kFrameOffset = Align64(sizeof(ResultsHeader));
constexpr uint32_t kMetaOffset = Align64(kFrameOffset + sizeof(ResultsFrame));
constexpr uint32_t kTotalBytes = kMetaOffset + kResultsMetadataCapacity;

bool ProcessAlive(DWORD pid) {
  if (pid == 0) return false;
  HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  if (!h) return false;
  DWORD code = 0;
  const bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
  CloseHandle(h);
  return alive;
}

}  // namespace

ResultPublisher::ResultPublisher() {
  mapping_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, kTotalBytes, kResultsMappingName);
  if (!mapping_) throw std::runtime_error("创建结果共享内存失败");
  const bool existed = GetLastError() == ERROR_ALREADY_EXISTS;
  header_ = static_cast<ResultsHeader*>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, 0));
  if (!header_) {
    CloseHandle(mapping_);
    throw std::runtime_error("映射结果共享内存失败");
  }
  // 已存在：读取方仍持有上一次会话的映射，或另一个进程正在发布。只接管已退出进程留下的映射。
  MEMORY_BASIC_INFORMATION info{};
  VirtualQuery(header_, &info, sizeof(info));
  if (existed && (info.RegionSize < kTotalBytes ||
                  (header_->magic == kResultsMagic && ProcessAlive(header_->hostPid) &&
                   header_->hostPid != GetCurrentProcessId()))) {
    const DWORD other = header_->magic == kResultsMagic ? header_->hostPid : 0;
    UnmapViewOfFile(header_);
    CloseHandle(mapping_);
    throw std::runtime_error(other ? "进程 " + std::to_string(other) + " 正在发布结果，本会话不发布"
                                   : std::string("结果共享内存已被占用且布局不兼容，本会话不发布"));
  }
  for (int i = 0; i < 2; ++i) {
    events_[i] = CreateEventW(nullptr, TRUE, FALSE, i ? kResultsEventName1 : kResultsEventName0);
    if (!events_[i]) {
      Close();
      throw std::runtime_error("创建结果通知事件失败");
    }
    ResetEvent(events_[i]);
  }

  // 接管时保留序号与帧记录（读取方以元数据中的会话号区分新旧会话）；
  // 上一个写入者在写入中途退出时序号停在奇数，补到下一个偶数
  if (!existed || header_->magic != kResultsMagic) {
    header_->frameSeq = 0;
    header_->metaSeq = 0;
    header_->metaLength = 0;
  }
  if (header_->frameSeq & 1) InterlockedIncrement64(const_cast<int64_t*>(&header_->frameSeq));
  if (header_->metaSeq & 1) InterlockedIncrement64(const_cast<int64_t*>(&header_->metaSeq));
  header_->magic = kResultsMagic;
  header_->version = kResultsLayoutVersion;
  header_->totalBytes = kTotalBytes;
  header_->frameOffset = kFrameOffset;
  header_->metaOffset = kMetaOffset;
  header_->metaCapacity = kResultsMetadataCapacity;
  header_->hostPid = GetCurrentProcessId();
  header_->qpcFrequency = QpcFrequency();
  frame_ = reinterpret_cast<ResultsFrame*>(reinterpret_cast<char*>(header_) + kFrameOffset);
  meta_ = reinterpret_cast<char*>(header_) + kMetaOffset;
}

ResultPublisher::~ResultPublisher() { Close(); }

void ResultPublisher::Close() {
  if (header_) {
    header_->hostPid = 0;  // 标记为无人发布，下一个会话可直接接管
    UnmapViewOfFile(header_);
  }
  for (HANDLE& e : events_) {
    if (e) CloseHandle(e);
    e = nullptr;
  }
  if (mapping_) CloseHandle(mapping_);
  header_ = nullptr;
  mapping_ = nullptr;
}

void ResultPublisher::PublishMetadata(const std::string& utf8Json) {
  const size_t n = std::min<size_t>(utf8Json.size(), kResultsMetadataCapacity);
  int64_t* seq = const_cast<int64_t*>(&header_->metaSeq);
  const int64_t s = *seq;
  InterlockedExchange64(seq, s + 1);
  std::memcpy(meta_, utf8Json.data(), n);
  header_->metaLength = static_cast<uint32_t>(n);
  InterlockedExchange64(seq, s + 2);
}

void ResultPublisher::Publish(const PublishedFrame& f, const DetectionOutput& r) {
  int64_t* seq = const_cast<int64_t*>(&header_->frameSeq);
  const int64_t s = *seq;
  const int64_t k = s / 2 + 1;  // 本次发布后的已发布帧数
  ResetEvent(events_[(k + 1) % 2]);
  InterlockedExchange64(seq, s + 1);

  frame_->frameIndex = f.frameIndex;
  frame_->acquireQpc = f.acquireQpc;
  frame_->resultQpc = f.resultQpc;
  frame_->regionX = f.region.left;
  frame_->regionY = f.region.top;
  frame_->regionW = f.region.right - f.region.left;
  frame_->regionH = f.region.bottom - f.region.top;
  frame_->count = r.count;
  frame_->numKeypoints = r.numKeypoints;
  frame_->candidates = r.candidates;
  frame_->overflow = r.overflow;
  // 只复制有效部分；读取方按 count 与 numKeypoints 取用
  std::memcpy(frame_->detections, r.detections, sizeof(Detection) * static_cast<size_t>(r.count));
  if (r.numKeypoints > 0) {
    std::memcpy(frame_->keypoints, r.keypoints,
                sizeof(Keypoint) * static_cast<size_t>(r.count) * static_cast<size_t>(r.numKeypoints));
  }

  InterlockedExchange64(seq, s + 2);
  SetEvent(events_[k % 2]);
}

}  // namespace nr
