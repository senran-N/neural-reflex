#include "pipeline/session.h"

#include "capture/dd_capture.h"
#include "capture/wgc_capture.h"
#include "common/d3d_util.h"
#include "common/latest_slot.h"
#include "common/qpc.h"
#include "common/win_util.h"
#include "inference/d3d_cuda_interop.h"
#include "overlay/overlay.h"

#include <unknwn.h>
#include <winrt/base.h>

#include <algorithm>
#include <exception>
#include <iterator>
#include <memory>

namespace nr {
namespace {

constexpr size_t kMaxSeconds = 600;

struct OverlayFrame {
  std::vector<RECT> boxes;
  std::vector<int> classes;
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
    Overlay overlay(adapter, screenRect);
    HANDLE waitable = overlay.FrameWaitable();
    uint64_t seen = 0;
    OverlayFrame current;
    std::vector<ColorRect> rects;
    // 与界面强调色一致的首色，其余按类别循环
    static constexpr float kPalette[][3] = {{0.24f, 0.86f, 0.52f}, {1.0f, 0.36f, 0.48f}, {0.29f, 0.60f, 1.0f},
                                            {1.0f, 0.80f, 0.20f}, {1.0f, 0.55f, 0.20f}, {0.40f, 0.90f, 0.95f}};
    while (!stop) {
      const DWORD r = MsgWaitForMultipleObjectsEx(1, &waitable, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
      if (r == WAIT_OBJECT_0 + 1) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) DispatchMessageW(&msg);
        continue;
      }
      if (r != WAIT_OBJECT_0) continue;
      slot.TakeNewer(seen, current);
      rects.clear();
      if (QpcToMs(QpcNow() - current.acquireQpc) <= staleMs) {
        for (size_t i = 0; i < current.boxes.size(); ++i) {
          const RECT& b = current.boxes[i];
          const float* c = kPalette[static_cast<size_t>(current.classes[i]) % std::size(kPalette)];
          AppendOutline(rects, static_cast<float>(b.left), static_cast<float>(b.top),
                        static_cast<float>(b.right - b.left), static_cast<float>(b.bottom - b.top), 2.0f, c[0],
                        c[1], c[2], 1.0f);
        }
      }
      overlay.Present(rects);
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

void Session::Run(SessionConfig config) {
  SetCurrentThreadName(L"nr-session");
  std::atomic<bool> overlayStop{false}, overlayFailed{false};
  std::atomic<uint64_t> presents{0};
  std::string overlayError;
  std::thread overlayThread;
  try {
    try {
      winrt::init_apartment(winrt::apartment_type::multi_threaded);
    } catch (const winrt::hresult_error&) {
      // 线程已在其他套间中初始化
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
    Detector detector(device.Get(), CudaDeviceForLuid(adapter.luid), config.detector);

    std::unique_ptr<CaptureSource> source;
    if (config.mode == CaptureMode::kDuplication) {
      source = CreateDuplicationCapture(device.Get(), adapter.output.Get());
    } else if (window) {
      source = CreateWgcWindowCapture(device.Get(), window);
    } else {
      source = CreateWgcMonitorCapture(device.Get(), monitor);
    }

    LatestSlot<OverlayFrame> slot;
    if (config.overlay) {
      overlayThread = std::thread(OverlayLoop, adapter.adapter.Get(), mi.rcMonitor, std::cref(slot),
                                  std::cref(overlayStop), config.staleMs, std::ref(presents),
                                  std::ref(overlayFailed), std::ref(overlayError));
    }

    const int64_t start = QpcNow();
    {
      std::lock_guard lock(mutex_);
      state_.phase = SessionPhase::kRunning;
      state_.prepareMs = QpcToMs(start - prepareStart);
      state_.adapter = Utf8(adapter.name);
      state_.inputSize = detector.InputSize();
      state_.numClasses = detector.NumClasses();
    }
    float conf = conf_, iou = iou_;

    const int64_t freq = QpcFrequency();
    int64_t secStart = start;
    uint64_t secPresentsStart = 0;
    uint32_t secOverflow = 0;
    std::vector<double> secLatency, secPre, secInfer, secPost;
    double secBoxes = 0;
    OverlayFrame frameOut;

    while (!stop_) {
      if (overlayFailed) throw std::runtime_error("覆盖层失败：" + overlayError);
      if (window && !IsWindow(window)) throw std::runtime_error("目标窗口已关闭");

      CapturedFrame frame;
      const AcquireStatus status = source->AcquireLatest(100, frame);
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
          if (conf != conf_ || iou != iou_) {
            conf = conf_;
            iou = iou_;
            detector.SetThresholds(conf, iou);
          }
          const DetectionOutput& r = detector.Run(context.Get(), frame.texture, region,
                                                  static_cast<float>(origin.x), static_cast<float>(origin.y));
          const double ms = QpcToMs(QpcNow() - frame.acquireQpc);

          frameOut.boxes.clear();
          frameOut.classes.clear();
          std::map<int, int> classes;
          for (int i = 0; i < r.count; ++i) {
            const Detection& d = r.detections[i];
            frameOut.boxes.push_back({static_cast<LONG>(d.x1), static_cast<LONG>(d.y1), static_cast<LONG>(d.x2),
                                      static_cast<LONG>(d.y2)});
            frameOut.classes.push_back(d.cls);
            ++classes[d.cls];
          }
          frameOut.acquireQpc = frame.acquireQpc;
          if (config.overlay) slot.Publish(frameOut);

          const StageTimes& t = detector.LastTimes();
          secLatency.push_back(ms);
          secPre.push_back(t.preprocessMs);
          secInfer.push_back(t.inferenceMs);
          secPost.push_back(t.postprocessMs);
          secBoxes += r.count;
          if (r.overflow) ++secOverflow;

          std::lock_guard lock(mutex_);
          latency_.Add(ms);
          pre_.Add(t.preprocessMs);
          infer_.Add(t.inferenceMs);
          post_.Add(t.postprocessMs);
          ++state_.detections;
          if (r.overflow) ++state_.overflowFrames;
          state_.lastClasses = std::move(classes);
        }
      }

      const int64_t now = QpcNow();
      if (now - secStart >= freq) {
        const double secs = static_cast<double>(now - secStart) / freq;
        SessionSecond s;
        s.t = QpcToMs(now - start) / 1000.0;
        s.detectionsPerSec = secLatency.size() / secs;
        const uint64_t p = presents.load();
        s.overlayPerSec = (p - secPresentsStart) / secs;
        s.latency = ComputePercentiles(secLatency);
        s.preMs = Median(secPre);
        s.inferMs = Median(secInfer);
        s.postMs = Median(secPost);
        s.boxes = secLatency.empty() ? 0 : secBoxes / secLatency.size();
        s.overflowFrames = secOverflow;

        DXGI_QUERY_VIDEO_MEMORY_INFO vram{};
        const bool haveVram = adapter3 && SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vram));

        std::lock_guard lock(mutex_);
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

    overlayStop = true;
    if (overlayThread.joinable()) overlayThread.join();
    detector.SaveCaches();
    std::lock_guard lock(mutex_);
    state_.phase = SessionPhase::kIdle;
  } catch (const winrt::hresult_error& e) {
    overlayStop = true;
    if (overlayThread.joinable()) overlayThread.join();
    Fail("WinRT 错误：" + Utf8(std::wstring(e.message())));
  } catch (const std::exception& e) {
    overlayStop = true;
    if (overlayThread.joinable()) overlayThread.join();
    Fail(e.what());
  }
}

}  // namespace nr
