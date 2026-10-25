#include "capture/wgc_capture.h"

#include "common/qpc.h"
#include "common/win_util.h"

#include <unknwn.h>
#include <dwmapi.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Metadata.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Security.Authorization.AppCapabilityAccess.h>

#include <algorithm>
#include <atomic>
#include <chrono>

namespace nr {
namespace {

namespace wgc = winrt::Windows::Graphics::Capture;
namespace wdx = winrt::Windows::Graphics::DirectX;

constexpr auto kPixelFormat = wdx::DirectXPixelFormat::B8G8R8A8UIntNormalized;
constexpr int32_t kPoolBuffers = 2;

wdx::Direct3D11::IDirect3DDevice WrapDevice(ID3D11Device* device) {
  winrt::com_ptr<IDXGIDevice> dxgi;
  winrt::check_hresult(device->QueryInterface(IID_PPV_ARGS(dxgi.put())));
  winrt::com_ptr<::IInspectable> inspectable;
  winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(), inspectable.put()));
  return inspectable.as<wdx::Direct3D11::IDirect3DDevice>();
}

class WgcCapture final : public CaptureSource {
 public:
  WgcCapture(ID3D11Device* device, wgc::GraphicsCaptureItem item, HWND window, POINT monitorOrigin)
      : device_(device), item_(std::move(item)), window_(window), monitorOrigin_(monitorOrigin) {
    device_->GetImmediateContext(context_.put());
    winrtDevice_ = WrapDevice(device_);
    frameEvent_.attach(CreateEventW(nullptr, FALSE, FALSE, nullptr));
    closedToken_ = item_.Closed([this](auto&&, auto&&) {
      closed_ = true;
      SetEvent(frameEvent_.get());
    });
    Start();
  }

  ~WgcCapture() override {
    Stop();
    item_.Closed(closedToken_);
  }

  const char* Name() const override { return window_ ? "WGC 窗口" : "WGC 显示器"; }

  AcquireStatus AcquireLatest(uint32_t timeoutMs, CapturedFrame& out) override {
    if (closed_) return AcquireStatus::kLost;
    if (WaitForSingleObject(frameEvent_.get(), timeoutMs) != WAIT_OBJECT_0) return AcquireStatus::kTimeout;
    if (closed_) return AcquireStatus::kLost;

    // 取最新一帧，期间到达的旧帧直接关闭
    wgc::Direct3D11CaptureFrame latest{nullptr};
    uint32_t count = 0;
    while (auto frame = pool_.TryGetNextFrame()) {
      if (latest) latest.Close();
      latest = frame;
      ++count;
    }
    if (!latest) return AcquireStatus::kTimeout;
    const int64_t acquireQpc = QpcNow();

    const auto size = latest.ContentSize();
    if (size.Width != poolSize_.Width || size.Height != poolSize_.Height) {
      // 窗口尺寸变化：按新尺寸重建帧池，本帧仍按有效区域使用
      poolSize_ = size;
      pool_.Recreate(winrtDevice_, kPixelFormat, kPoolBuffers, size);
    }

    auto access = latest.Surface().as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<ID3D11Texture2D> surface;
    winrt::check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D), surface.put_void()));

    D3D11_TEXTURE2D_DESC sd{};
    surface->GetDesc(&sd);
    const uint32_t w = std::min<uint32_t>(size.Width, sd.Width);
    const uint32_t h = std::min<uint32_t>(size.Height, sd.Height);
    EnsureOwnedTexture(w, h, sd.Format);

    // 设计方案 4.2 允许的那一次 GPU 内复制
    D3D11_BOX box{0, 0, 0, w, h, 1};
    context_->CopySubresourceRegion(owned_.get(), 0, 0, 0, 0, surface.get(), 0, &box);
    const int64_t sourceQpc = Hns100ToQpc(latest.SystemRelativeTime().count());
    latest.Close();

    out.texture = owned_.get();
    out.width = w;
    out.height = h;
    out.screenOrigin = CurrentOrigin();
    out.sourceQpc = sourceQpc;
    out.acquireQpc = acquireQpc;
    out.accumulated = count;
    return AcquireStatus::kNewFrame;
  }

  bool Recreate() override {
    // 采集对象关闭（例如目标窗口销毁）后无法恢复，由上层重新选择目标
    return false;
  }

 private:
  void Start() {
    poolSize_ = item_.Size();
    pool_ = wgc::Direct3D11CaptureFramePool::CreateFreeThreaded(winrtDevice_, kPixelFormat,
                                                                kPoolBuffers, poolSize_);
    frameToken_ = pool_.FrameArrived([this](auto&&, auto&&) { SetEvent(frameEvent_.get()); });
    session_ = pool_.CreateCaptureSession(item_);

    using winrt::Windows::Foundation::Metadata::ApiInformation;
    constexpr wchar_t kSession[] = L"Windows.Graphics.Capture.GraphicsCaptureSession";
    if (ApiInformation::IsPropertyPresent(kSession, L"IsCursorCaptureEnabled")) {
      session_.IsCursorCaptureEnabled(false);
    }
    if (ApiInformation::IsPropertyPresent(kSession, L"IsBorderRequired")) {
      try {
        wgc::GraphicsCaptureAccess::RequestAccessAsync(wgc::GraphicsCaptureAccessKind::Borderless).get();
        session_.IsBorderRequired(false);
      } catch (const winrt::hresult_error&) {
        // 不允许去掉黄色边框时保留边框，不影响采集
      }
    }
    if (ApiInformation::IsPropertyPresent(kSession, L"MinUpdateInterval")) {
      // 不让系统节流本程序的采集速率
      session_.MinUpdateInterval(std::chrono::milliseconds(1));
    }
    session_.StartCapture();
  }

  void Stop() {
    if (session_) session_.Close();
    if (pool_) {
      pool_.FrameArrived(frameToken_);
      pool_.Close();
    }
  }

  void EnsureOwnedTexture(uint32_t w, uint32_t h, DXGI_FORMAT format) {
    if (owned_ && ownedW_ == w && ownedH_ == h) return;
    D3D11_TEXTURE2D_DESC td{};
    td.Width = w;
    td.Height = h;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = format;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    owned_ = nullptr;
    Check(device_->CreateTexture2D(&td, nullptr, owned_.put()), "CreateTexture2D(采集副本)");
    ownedW_ = w;
    ownedH_ = h;
  }

  POINT CurrentOrigin() const {
    if (!window_) return monitorOrigin_;
    // WGC 窗口采集的画面以 DWM 扩展边框为界（不含不可见的缩放边框）
    RECT bounds{};
    if (SUCCEEDED(DwmGetWindowAttribute(window_, DWMWA_EXTENDED_FRAME_BOUNDS, &bounds, sizeof(bounds)))) {
      return {bounds.left, bounds.top};
    }
    GetWindowRect(window_, &bounds);
    return {bounds.left, bounds.top};
  }

  ID3D11Device* device_;
  winrt::com_ptr<ID3D11DeviceContext> context_;
  wdx::Direct3D11::IDirect3DDevice winrtDevice_{nullptr};
  wgc::GraphicsCaptureItem item_;
  HWND window_;
  POINT monitorOrigin_;
  wgc::Direct3D11CaptureFramePool pool_{nullptr};
  wgc::GraphicsCaptureSession session_{nullptr};
  winrt::Windows::Graphics::SizeInt32 poolSize_{};
  winrt::event_token frameToken_{};
  winrt::event_token closedToken_{};
  winrt::handle frameEvent_;
  std::atomic<bool> closed_{false};
  winrt::com_ptr<ID3D11Texture2D> owned_;
  uint32_t ownedW_ = 0, ownedH_ = 0;
};

auto CaptureInterop() {
  return winrt::get_activation_factory<wgc::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
}

}  // namespace

std::unique_ptr<CaptureSource> CreateWgcWindowCapture(ID3D11Device* device, HWND window) {
  wgc::GraphicsCaptureItem item{nullptr};
  winrt::check_hresult(CaptureInterop()->CreateForWindow(
      window, winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item)));
  return std::make_unique<WgcCapture>(device, std::move(item), window, POINT{});
}

std::unique_ptr<CaptureSource> CreateWgcMonitorCapture(ID3D11Device* device, HMONITOR monitor) {
  wgc::GraphicsCaptureItem item{nullptr};
  winrt::check_hresult(CaptureInterop()->CreateForMonitor(
      monitor, winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item)));
  MONITORINFO mi{sizeof(mi)};
  GetMonitorInfoW(monitor, &mi);
  return std::make_unique<WgcCapture>(device, std::move(item), nullptr,
                                      POINT{mi.rcMonitor.left, mi.rcMonitor.top});
}

}  // namespace nr
