#include "pipeline/session.h"

#include "capture/dd_capture.h"
#include "capture/wgc_capture.h"
#include "common/d3d_util.h"
#include "common/latest_slot.h"
#include "common/qpc.h"
#include "common/win_util.h"
#include "inference/d3d_cuda_interop.h"
#include "inference/cuda_check.h"
#include "overlay/overlay.h"
#include "results/result_publisher.h"

#include <avrt.h>
#include <nlohmann/json.hpp>

#include <unknwn.h>
#include <winrt/base.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>

namespace nr {
namespace {

constexpr size_t kMaxSeconds = 600;
constexpr float kKeypointMinScore = 0.25f;  // 关键点显示阈值，与 Ultralytics 绘图缺省值一致
constexpr float kKeypointSize = 5.0f;
// 每个框 4 个矩形，每个关键点 1 个
constexpr size_t kOverlayMaxRects = static_cast<size_t>(kMaxDetections) * (4 + kMaxKeypoints);

struct OverlayPoint {
  float x, y;
  int cls;
};

struct OverlayFrame {
  std::vector<RECT> boxes;
  std::vector<int> classes;
  std::vector<OverlayPoint> points;
  int64_t acquireQpc = 0;
};

std::string Utf8(const std::wstring& w) {
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
  std::string s(static_cast<size_t>(n > 0 ? n - 1 : 0), '\0');
  if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
  return s;
}

double Median(std::vector<double>& v) {
  if (v.empty()) return 0;
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// 覆盖层线程：在呈现时机取最新结果画框，不等待未就绪的结果（设计方案 5.2、6.2）
void OverlayLoop(IDXGIAdapter1* adapter, RECT screenRect, const LatestSlot<OverlayFrame>& slot,
                 const std::atomic<bool>& stop, double staleMs, std::atomic<uint64_t>& presents,
                 std::atomic<bool>& failed, std::string& error) {
  SetCurrentThreadName(L"nr-overlay");
  try {
    Overlay overlay(adapter, screenRect, kOverlayMaxRects);
    uint64_t seen = 0;
    bool haveFrame = false, showing = false;
    OverlayFrame current;
    std::vector<ColorRect> rects;
    // 与界面强调色一致的首色，其余按类别循环
    static constexpr float kPalette[][3] = {{0.24f, 0.86f, 0.52f}, {1.0f, 0.36f, 0.48f}, {0.29f, 0.60f, 1.0f},
                                            {1.0f, 0.80f, 0.20f}, {1.0f, 0.55f, 0.20f}, {0.40f, 0.90f, 0.95f}};
    while (!stop) {
      // 等待消息并低频检查最新结果；不等待交换链，避免无新结果时形成满刷新率空转。
      const DWORD r = MsgWaitForMultipleObjectsEx(0, nullptr, 4, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
      if (r == WAIT_OBJECT_0) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) DispatchMessageW(&msg);
      }

      const bool updated = slot.TakeNewer(seen, current);
      haveFrame = haveFrame || updated;
      const bool fresh = haveFrame && QpcToMs(QpcNow() - current.acquireQpc) <= staleMs;
      if (!updated && fresh == showing) continue;  // 画面没有变化，不提交任何 GPU 绘制或 Present

      rects.clear();
      if (fresh) {
        for (size_t i = 0; i < current.boxes.size(); ++i) {
          const RECT& b = current.boxes[i];
          const float* c = kPalette[static_cast<size_t>(current.classes[i]) % std::size(kPalette)];
          AppendOutline(rects, static_cast<float>(b.left), static_cast<float>(b.top),
                        static_cast<float>(b.right - b.left), static_cast<float>(b.bottom - b.top), 2.0f, c[0],
                        c[1], c[2], 1.0f);
        }
        for (const OverlayPoint& p : current.points) {
          const float* c = kPalette[static_cast<size_t>(p.cls) % std::size(kPalette)];
          rects.push_back({p.x - kKeypointSize / 2, p.y - kKeypointSize / 2, kKeypointSize, kKeypointSize, c[0], c[1],
                           c[2], 1.0f});
        }
      }
      overlay.Present(rects);
      showing = fresh;
      presents++;
    }
  } catch (const std::exception& e) {
    error = e.what();
    failed = true;
  }
}

}  // namespace

const char* PhaseName(SessionPhase phase) {
  switch (phase) {
    case SessionPhase::kIdle: return "idle";
    case SessionPhase::kPreparing: return "preparing";
    case SessionPhase::kRunning: return "running";
    case SessionPhase::kFailed: return "failed";
  }
  return "idle";
}

Session::~Session() { Stop(); }

void Session::Start(const SessionConfig& config) {
  Stop();
  {
    std::lock_guard lock(mutex_);
    state_ = SessionSnapshot{};
    state_.phase = SessionPhase::kPreparing;
    seconds_.clear();
    latency_ = pre_ = infer_ = post_ = LatencyHistogram{};
  }
  stop_ = false;
  conf_ = config.detector.confThreshold;
  iou_ = config.detector.iouThreshold;
  worker_ = std::thread(&Session::Run, this, config);
}

void Session::SetThresholds(float conf, float iou) {
  conf_ = conf;
  iou_ = iou;
}

void Session::Stop() {
  stop_ = true;
  if (worker_.joinable()) worker_.join();
}

bool Session::Active() const {
  std::lock_guard lock(mutex_);
  return state_.phase == SessionPhase::kPreparing || state_.phase == SessionPhase::kRunning;
}

SessionSnapshot Session::Snapshot(size_t maxSeconds) const {
  std::lock_guard lock(mutex_);
  SessionSnapshot s = state_;
  s.latency = latency_.Get();
  s.pre = pre_.Get();
  s.infer = infer_.Get();
  s.post = post_.Get();
  const size_t n = std::min(maxSeconds, seconds_.size());
  s.seconds.assign(seconds_.end() - static_cast<std::ptrdiff_t>(n), seconds_.end());
  return s;
}

void Session::Fail(const std::string& message) {
  std::lock_guard lock(mutex_);
  state_.phase = SessionPhase::kFailed;
  state_.error = message;
}

namespace {

// 采集线程与完成线程之间的在途项
struct InFlight {
  int slot = 0;
  int64_t acquireQpc = 0;
  RECT region{};  // 检测区域，屏幕物理像素
};

// 发布给读取方的会话元数据（扩展机制方案 5.2）
std::string SessionMetadata(const Detector& detector, const SessionConfig& config, int64_t sessionId,
                            const char* state) {
  const ModelManifest& m = detector.Manifest();
  nlohmann::json classes = nlohmann::json::object();
  for (const auto& [k, v] : m.classNames) classes[std::to_string(k)] = v;
  const std::string roiModeStr = config.roi.mode == RoiMode::kCenter   ? "center"
                                 : config.roi.mode == RoiMode::kCustom ? "custom"
                                                                       : "full";
  nlohmann::json j = {{"layout", kResultsLayoutVersion},
                      {"session", sessionId},
                      {"state", state},
                      {"pid", GetCurrentProcessId()},
                      {"model", Utf8(config.detector.engine.onnx.stem().wstring())},
                      {"decoder", detector.DecoderName()},
                      {"inputSize", detector.InputSize()},
                      {"numClasses", detector.NumClasses()},
                      {"coordinates", "screen-physical"},  // 扩展机制方案 7.5
                      {"clock", "qpc"},
                      {"roi",
                       {{"mode", roiModeStr},
                        {"width", config.roi.width},
                        {"height", config.roi.height},
                        {"x", config.roi.x},
                        {"y", config.roi.y}}},
                      {"classNames", classes},
                      {"keypoints",
                       {{"count", detector.NumKeypoints()}, {"names", m.keypointNames}, {"skeleton", m.skeleton}}}};
  return j.dump();
}

class MmcssScope {
 public:
  explicit MmcssScope(bool enable) {
    if (enable) handle_ = AvSetMmThreadCharacteristicsW(L"Games", &index_);
  }
  ~MmcssScope() {
    if (handle_) AvRevertMmThreadCharacteristics(handle_);
  }

 private:
  HANDLE handle_ = nullptr;
  DWORD index_ = 0;
};

}  // namespace

void Session::Run(SessionConfig config) {
  SetCurrentThreadName(L"nr-capture");
  MmcssScope mmcss(config.mmcss);
  std::atomic<bool> overlayStop{false}, overlayFailed{false};
  std::atomic<uint64_t> presents{0};
  std::string overlayError;
  std::thread overlayThread, completionThread;

  // 在途队列
  std::mutex queueMutex;
  std::condition_variable queueCv;
  std::deque<InFlight> queue;
  int inflight = 0;
  bool producerDone = false;
  std::exception_ptr completionError;
  const int maxInflight = std::clamp(config.inflight, 1, Detector::kSlots);

  auto shutdown = [&] {
    {
      std::lock_guard lock(queueMutex);
      producerDone = true;
    }
    queueCv.notify_all();
    if (completionThread.joinable()) completionThread.join();
    overlayStop = true;
    if (overlayThread.joinable()) overlayThread.join();
  };

  try {
    try {
      winrt::init_apartment(winrt::apartment_type::multi_threaded);
    } catch (const winrt::hresult_error&) {
      // 线程已在其他套间中初始化
    }

    const long priorityStatus = SetGpuPriority(config.gpuPriority);
    {
      std::lock_guard lock(mutex_);
      state_.gpuPriorityStatus = priorityStatus;
      state_.gpuPriorityEffective = GetGpuPriority();
    }

    HWND window = config.window;
    if (window && !IsWindow(window)) throw std::runtime_error("目标窗口已关闭");
    HMONITOR monitor = window ? MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST) : config.monitor;
    if (!monitor) monitor = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO mi{sizeof(mi)};
    GetMonitorInfoW(monitor, &mi);

    // 采集、计算与呈现使用驱动该显示器的同一块 GPU（设计方案第 2 节）
    AdapterInfo adapter = FindAdapterForMonitor(monitor);
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Device> device = CreateDevice(adapter.adapter.Get(), &context);
    ComPtr<IDXGIAdapter3> adapter3;
    adapter.adapter.As(&adapter3);

    const int64_t prepareStart = QpcNow();
    const int cudaDevice = CudaDeviceForLuid(adapter.luid);
    Detector detector(device.Get(), cudaDevice, config.detector);

    std::unique_ptr<CaptureSource> source;
    if (config.mode == CaptureMode::kDuplication) {
      source = CreateDuplicationCapture(device.Get(), adapter.output.Get());
    } else if (window) {
      source = CreateWgcWindowCapture(device.Get(), window);
    } else {
      source = CreateWgcMonitorCapture(device.Get(), monitor);
    }

    // 结果发布失败不影响检测，原因显示在界面中
    std::unique_ptr<ResultPublisher> publisher;
    const int64_t sessionId = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
    if (config.publishResults) {
      try {
        publisher = std::make_unique<ResultPublisher>();
        publisher->PublishMetadata(SessionMetadata(detector, config, sessionId, "running"));
      } catch (const std::exception& e) {
        publisher.reset();
        std::lock_guard lock(mutex_);
        state_.publishError = e.what();
      }
      std::lock_guard lock(mutex_);
      state_.publishing = publisher != nullptr;
      state_.sessionId = publisher ? sessionId : 0;
    }

    LatestSlot<OverlayFrame> overlaySlot;
    if (config.overlay) {
      overlayThread = std::thread(OverlayLoop, adapter.adapter.Get(), mi.rcMonitor, std::cref(overlaySlot),
                                  std::cref(overlayStop), config.staleMs, std::ref(presents),
                                  std::ref(overlayFailed), std::ref(overlayError));
    }

    // 本秒统计：完成线程写入，采集线程每秒汇总一次（均在 mutex_ 下）
    uint32_t secOverflow = 0;
    std::vector<double> secLatency, secPre, secInfer, secPost;
    double secBoxes = 0;

    // 两个线程按引用使用本作用域内的对象（检测器、覆盖层交接、统计、发布）。异常离开本作用域时，
    // 这些对象在进入 catch 之前就已析构，因此在这里先结束线程，再标记发布结束。
    struct ScopeExit {
      std::function<void()> f;
      ~ScopeExit() { f(); }
    } joinThreads{[&] {
      shutdown();
      try {
        if (publisher) publisher->PublishMetadata(SessionMetadata(detector, config, sessionId, "stopped"));
      } catch (const std::exception&) {
        // 析构路径上不能抛出；元数据保持 running，读取方可由 pid 判断宿主已退出
      }
    }};

    // 完成线程：按提交顺序等待结果并发布，不阻塞采集
    completionThread = std::thread([&] {
      SetCurrentThreadName(L"nr-complete");
      MmcssScope completionMmcss(config.mmcss);
      try {
        CudaCheck(cudaSetDevice(cudaDevice), "cudaSetDevice");
        OverlayFrame frameOut;
        uint64_t frameIndex = 0;
        for (;;) {
          InFlight item;
          {
            std::unique_lock lock(queueMutex);
            queueCv.wait(lock, [&] { return !queue.empty() || producerDone; });
            if (queue.empty()) return;
            item = queue.front();
            queue.pop_front();
          }
          StageTimes t;
          const DetectionOutput& r = detector.Wait(item.slot, &t);
          const int64_t doneQpc = QpcNow();
          const double ms = QpcToMs(doneQpc - item.acquireQpc);
          if (publisher) publisher->Publish({sessionId, frameIndex, item.acquireQpc, doneQpc, item.region}, r);
          ++frameIndex;

          frameOut.boxes.clear();
          frameOut.classes.clear();
          frameOut.points.clear();
          std::map<int, int> classes;
          for (int i = 0; i < r.count; ++i) {
            const Detection& d = r.detections[i];
            frameOut.boxes.push_back({static_cast<LONG>(d.x1), static_cast<LONG>(d.y1), static_cast<LONG>(d.x2),
                                      static_cast<LONG>(d.y2)});
            frameOut.classes.push_back(d.cls);
            ++classes[d.cls];
            for (int j = 0; j < r.numKeypoints; ++j) {
              const Keypoint& k = r.keypoints[i * r.numKeypoints + j];
              if (k.score >= kKeypointMinScore) frameOut.points.push_back({k.x, k.y, d.cls});
            }
          }
          frameOut.acquireQpc = item.acquireQpc;
          if (config.overlay) overlaySlot.Publish(frameOut);
          const bool overflow = r.overflow > 0;
          const int count = r.count;

          {
            std::lock_guard lock(queueMutex);
            --inflight;
          }
          queueCv.notify_all();

          std::lock_guard lock(mutex_);
          secLatency.push_back(ms);
          secPre.push_back(t.preprocessMs);
          secInfer.push_back(t.inferenceMs);
          secPost.push_back(t.postprocessMs);
          secBoxes += count;
          if (overflow) ++secOverflow;
          latency_.Add(ms);
          pre_.Add(t.preprocessMs);
          infer_.Add(t.inferenceMs);
          post_.Add(t.postprocessMs);
          ++state_.detections;
          if (overflow) ++state_.overflowFrames;
          state_.lastClasses = std::move(classes);
        }
      } catch (...) {
        std::lock_guard lock(queueMutex);
        completionError = std::current_exception();
        inflight = 0;
        queueCv.notify_all();
      }
    });

    const int64_t start = QpcNow();
    {
      std::lock_guard lock(mutex_);
      state_.phase = SessionPhase::kRunning;
      state_.prepareMs = QpcToMs(start - prepareStart);
      state_.adapter = Utf8(adapter.name);
      state_.inputSize = detector.InputSize();
      state_.numClasses = detector.NumClasses();
      state_.numKeypoints = detector.NumKeypoints();
      state_.decoder = detector.DecoderName();
    }
    float conf = conf_, iou = iou_;
    const int64_t freq = QpcFrequency();
    int64_t secStart = start;
    uint64_t secPresentsStart = 0;
    int nextSlot = 0;

    while (!stop_) {
      if (overlayFailed) throw std::runtime_error("覆盖层失败：" + overlayError);
      {
        std::lock_guard lock(queueMutex);
        if (completionError) std::rethrow_exception(completionError);
      }
      if (window && !IsWindow(window)) throw std::runtime_error("目标窗口已关闭");

      // 在途帧达到上限时先等最早一帧完成，再取画面：等待期间到达的画面由采集接口合并为最新一帧，
      // 提交的总是等待结束时的最新画面（设计方案 6.1）
      bool slotFree;
      {
        std::unique_lock lock(queueMutex);
        slotFree = queueCv.wait_for(lock, std::chrono::milliseconds(100),
                                    [&] { return inflight < maxInflight || completionError; });
        if (completionError) std::rethrow_exception(completionError);
      }

      CapturedFrame frame;
      const AcquireStatus status = slotFree ? source->AcquireLatest(100, frame) : AcquireStatus::kTimeout;
      if (status == AcquireStatus::kLost) {
        if (!source->Recreate()) throw std::runtime_error("采集中断且无法恢复");
        continue;
      }

      if (status == AcquireStatus::kNewFrame) {
        // 检测区域：窗口客户区，或整个采集画面
        D3D11_BOX region{0, 0, 0, frame.width, frame.height, 1};
        POINT origin = frame.screenOrigin;
        bool inFrame = true;
        if (window) {
          RECT client{};
          GetClientRect(window, &client);
          origin = {0, 0};
          ClientToScreen(window, &origin);
          const LONG x = origin.x - frame.screenOrigin.x, y = origin.y - frame.screenOrigin.y;
          inFrame = client.right > 0 && client.bottom > 0 && x >= 0 && y >= 0 &&
                    x + client.right <= static_cast<LONG>(frame.width) &&
                    y + client.bottom <= static_cast<LONG>(frame.height);
          region = {static_cast<UINT>(x), static_cast<UINT>(y), 0, static_cast<UINT>(x + client.right),
                    static_cast<UINT>(y + client.bottom), 1};
        }

        if (inFrame) {
          // 局部感兴趣区域（RoI）裁剪计算
          if (config.roi.mode != RoiMode::kFull) {
            const LONG baseW = static_cast<LONG>(region.right - region.left);
            const LONG baseH = static_cast<LONG>(region.bottom - region.top);
            const LONG baseX = static_cast<LONG>(region.left);
            const LONG baseY = static_cast<LONG>(region.top);
            const int targetW = config.roi.width > 0 ? config.roi.width : detector.InputSize();
            const int targetH = config.roi.height > 0 ? config.roi.height : detector.InputSize();
            const LONG roiW = std::clamp(static_cast<LONG>(targetW), 1L, baseW);
            const LONG roiH = std::clamp(static_cast<LONG>(targetH), 1L, baseH);
            LONG roiOffsetX = 0, roiOffsetY = 0;
            if (config.roi.mode == RoiMode::kCenter) {
              roiOffsetX = (baseW - roiW) / 2;
              roiOffsetY = (baseH - roiH) / 2;
            } else if (config.roi.mode == RoiMode::kCustom) {
              roiOffsetX = std::clamp(static_cast<LONG>(config.roi.x), 0L, std::max(0L, baseW - roiW));
              roiOffsetY = std::clamp(static_cast<LONG>(config.roi.y), 0L, std::max(0L, baseH - roiH));
            }
            region = {static_cast<UINT>(baseX + roiOffsetX), static_cast<UINT>(baseY + roiOffsetY), 0,
                      static_cast<UINT>(baseX + roiOffsetX + roiW), static_cast<UINT>(baseY + roiOffsetY + roiH), 1};
            origin.x += roiOffsetX;
            origin.y += roiOffsetY;
          }

          if (conf != conf_ || iou != iou_) {
            conf = conf_;
            iou = iou_;
            detector.SetThresholds(conf, iou);
          }
          detector.Submit(context.Get(), frame.texture, region, static_cast<float>(origin.x),
                          static_cast<float>(origin.y), nextSlot);
          source->ReleaseFrame();
          {
            std::lock_guard lock(queueMutex);
            const RECT screenRegion{origin.x, origin.y, origin.x + static_cast<LONG>(region.right - region.left),
                                    origin.y + static_cast<LONG>(region.bottom - region.top)};
            queue.push_back({nextSlot, frame.acquireQpc, screenRegion});
            ++inflight;
          }
          queueCv.notify_all();
          nextSlot = (nextSlot + 1) % Detector::kSlots;
        } else {
          source->ReleaseFrame();
        }
      }

      const int64_t now = QpcNow();
      if (now - secStart >= freq) {
        const double secs = static_cast<double>(now - secStart) / freq;
        DXGI_QUERY_VIDEO_MEMORY_INFO vram{};
        const bool haveVram =
            adapter3 && SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vram));
        const uint64_t p = presents.load();

        std::lock_guard lock(mutex_);
        SessionSecond s;
        s.t = QpcToMs(now - start) / 1000.0;
        s.detectionsPerSec = secLatency.size() / secs;
        s.overlayPerSec = (p - secPresentsStart) / secs;
        s.latency = ComputePercentiles(secLatency);
        s.preMs = Median(secPre);
        s.inferMs = Median(secInfer);
        s.postMs = Median(secPost);
        s.boxes = secLatency.empty() ? 0 : secBoxes / secLatency.size();
        s.overflowFrames = secOverflow;
        seconds_.push_back(s);
        if (seconds_.size() > kMaxSeconds) seconds_.pop_front();
        state_.elapsed = s.t;
        if (haveVram) {
          state_.vramUsedBytes = vram.CurrentUsage;
          state_.vramBudgetBytes = vram.Budget;
        }
        secStart = now;
        secPresentsStart = p;
        secOverflow = 0;
        secBoxes = 0;
        secLatency.clear();
        secPre.clear();
        secInfer.clear();
        secPost.clear();
      }
    }

    source->ReleaseFrame();
    shutdown();
    detector.SaveCaches();
    SetGpuPriority(GpuPriority::kNormal);
    std::lock_guard lock(mutex_);
    state_.phase = SessionPhase::kIdle;
  } catch (const winrt::hresult_error& e) {
    shutdown();
    SetGpuPriority(GpuPriority::kNormal);
    Fail("WinRT 错误：" + Utf8(std::wstring(e.message())));
  } catch (const std::exception& e) {
    shutdown();
    SetGpuPriority(GpuPriority::kNormal);
    Fail(e.what());
  }
}

}  // namespace nr
