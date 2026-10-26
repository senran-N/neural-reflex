#include "results/result_publisher.h"

#include "common/qpc.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace nr {
namespace {

constexpr uint32_t Align64(size_t v) { return static_cast<uint32_t>((v + 63) & ~size_t{63}); }
constexpr uint32_t kFrameOffset = 128;  // 头部之后留出扩展空间
constexpr uint32_t kMetaOffset = Align64(kFrameOffset + sizeof(ResultsFrame));
constexpr uint32_t kRingOffset = Align64(kMetaOffset + kResultsMetadataCapacity);
constexpr uint32_t kRingStride = Align64(sizeof(ResultsRingSlot));
constexpr uint32_t kTotalBytes = kRingOffset + sizeof(ResultsRingHeader) + kResultsRingSlots * kRingStride;
static_assert(sizeof(ResultsHeader) <= kFrameOffset);

ResultsRingSlot* SlotAt(ResultsRingHeader* ring, uint64_t index) {
  return reinterpret_cast<ResultsRingSlot*>(reinterpret_cast<char*>(ring) + sizeof(ResultsRingHeader) +
                                            index * kRingStride);
}

// 只复制有效部分；读取方按 count 与 numKeypoints 取用
void WriteRecord(ResultsFrame* dst, const PublishedFrame& f, const DetectionOutput& r) {
  dst->frameIndex = f.frameIndex;
  dst->acquireQpc = f.acquireQpc;
  dst->resultQpc = f.resultQpc;
  dst->regionX = f.region.left;
  dst->regionY = f.region.top;
  dst->regionW = f.region.right - f.region.left;
  dst->regionH = f.region.bottom - f.region.top;
  dst->count = r.count;
  dst->numKeypoints = r.numKeypoints;
  dst->candidates = r.candidates;
  dst->overflow = r.overflow;
  std::memcpy(dst->detections, r.detections, sizeof(Detection) * static_cast<size_t>(r.count));
  if (r.numKeypoints > 0) {
    std::memcpy(dst->keypoints, r.keypoints,
                sizeof(Keypoint) * static_cast<size_t>(r.count) * static_cast<size_t>(r.numKeypoints));
  }
}

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

ResultPublisher::ResultPublisher(const std::wstring& mapping) {
  mapping_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, kTotalBytes, mapping.c_str());
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
    events_[i] = CreateEventW(nullptr, TRUE, FALSE, (mapping + (i ? L".Event1" : L".Event0")).c_str());
    if (!events_[i]) {
      Close();
      throw std::runtime_error("创建结果通知事件失败");
    }
    ResetEvent(events_[i]);
  }

  // 接管时保留序号与帧记录（读取方以元数据中的会话号区分新旧会话）；
  // 上一个写入者在写入中途退出时序号停在奇数，补到下一个偶数
  ring_ = reinterpret_cast<ResultsRingHeader*>(reinterpret_cast<char*>(header_) + kRingOffset);
  if (!existed || header_->magic != kResultsMagic) {
    header_->frameSeq = 0;
    header_->metaSeq = 0;
    header_->metaLength = 0;
    for (uint32_t i = 0; i < kResultsRingSlots; ++i) SlotAt(ring_, i)->seq = 0;
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
  ring_->slots = kResultsRingSlots;
  ring_->stride = kRingStride;
  ring_->recordOffset = offsetof(ResultsRingSlot, record);
  header_->ringOffset = kRingOffset;
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
  ring_ = nullptr;
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

  WriteRecord(frame_, f, r);
  // 环形槽在帧序号锁的奇数期间写完：读取方看到已发布帧数 >= k 时第 k 帧已完整
  ResultsRingSlot* slot = SlotAt(ring_, static_cast<uint64_t>(k - 1) % kResultsRingSlots);
  int64_t* slotSeq = const_cast<int64_t*>(&slot->seq);
  InterlockedExchange64(slotSeq, 2 * k - 1);
  slot->session = f.session;
  WriteRecord(&slot->record, f, r);
  InterlockedExchange64(slotSeq, 2 * k);

  InterlockedExchange64(seq, s + 2);
  SetEvent(events_[k % 2]);
}

}  // namespace nr
