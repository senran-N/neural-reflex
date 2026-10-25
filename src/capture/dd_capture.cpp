#include "capture/dd_capture.h"

#include "common/qpc.h"
#include "common/win_util.h"

#include <dxgi1_6.h>
#include <wrl/client.h>

namespace nr {
namespace {

using Microsoft::WRL::ComPtr;

class DuplicationCapture final : public CaptureSource {
 public:
  DuplicationCapture(ID3D11Device* device, IDXGIOutput* output) : device_(device), output_(output) {
    device_->GetImmediateContext(&context_);
    DXGI_OUTPUT_DESC desc{};
    output_->GetDesc(&desc);
    origin_ = {desc.DesktopCoordinates.left, desc.DesktopCoordinates.top};
    Check(Duplicate(), "DuplicateOutput");
  }

  ~DuplicationCapture() override {
    if (holdingFrame_) duplication_->ReleaseFrame();
  }

  const char* Name() const override { return "Desktop Duplication"; }

  AcquireStatus AcquireLatest(uint32_t timeoutMs, CapturedFrame& out) override {
    if (!duplication_) return AcquireStatus::kLost;
    const int64_t deadline = QpcNow() + static_cast<int64_t>(timeoutMs) * QpcFrequency() / 1000;

    for (;;) {
      const int64_t remainingTicks = deadline - QpcNow();
      const UINT waitMs = remainingTicks > 0 ? static_cast<UINT>(QpcToMs(remainingTicks)) : 0;

      DXGI_OUTDUPL_FRAME_INFO info{};
      ComPtr<IDXGIResource> resource;
      HRESULT hr = duplication_->AcquireNextFrame(waitMs, &info, &resource);
      if (hr == DXGI_ERROR_WAIT_TIMEOUT) return AcquireStatus::kTimeout;
      if (hr == DXGI_ERROR_ACCESS_LOST || hr == DXGI_ERROR_INVALID_CALL) {
        duplication_.Reset();
        return AcquireStatus::kLost;
      }
      Check(hr, "AcquireNextFrame");
      holdingFrame_ = true;
      const int64_t acquireQpc = QpcNow();

      // LastPresentTime 为 0 表示只有鼠标指针更新，没有新画面
      if (info.LastPresentTime.QuadPart == 0) {
        Release();
        if (QpcNow() >= deadline) return AcquireStatus::kTimeout;
        continue;
      }

      ComPtr<ID3D11Texture2D> surface;
      Check(resource.As(&surface), "IDXGIResource -> ID3D11Texture2D");
      D3D11_TEXTURE2D_DESC sd{};
      surface->GetDesc(&sd);
      EnsureOwnedTexture(sd);
      // 复制后立即释放，缩短持有桌面帧的时间（设计方案 4.2）
      context_->CopyResource(owned_.Get(), surface.Get());
      Release();

      out.texture = owned_.Get();
      out.width = sd.Width;
      out.height = sd.Height;
      out.screenOrigin = origin_;
      out.sourceQpc = info.LastPresentTime.QuadPart;
      out.acquireQpc = acquireQpc;
      out.accumulated = info.AccumulatedFrames;
      return AcquireStatus::kNewFrame;
    }
  }

  bool Recreate() override {
    if (holdingFrame_) Release();
    duplication_.Reset();
    DXGI_OUTPUT_DESC desc{};
    output_->GetDesc(&desc);
    origin_ = {desc.DesktopCoordinates.left, desc.DesktopCoordinates.top};
    return SUCCEEDED(Duplicate());
  }

 private:
  HRESULT Duplicate() {
    ComPtr<IDXGIOutput5> output5;
    if (SUCCEEDED(output_->QueryInterface(IID_PPV_ARGS(&output5)))) {
      const DXGI_FORMAT formats[] = {DXGI_FORMAT_B8G8R8A8_UNORM};
      return output5->DuplicateOutput1(device_, 0, ARRAYSIZE(formats), formats, &duplication_);
    }
    ComPtr<IDXGIOutput1> output1;
    HRESULT hr = output_->QueryInterface(IID_PPV_ARGS(&output1));
    if (FAILED(hr)) return hr;
    return output1->DuplicateOutput(device_, &duplication_);
  }

  void Release() {
    duplication_->ReleaseFrame();
    holdingFrame_ = false;
  }

  void EnsureOwnedTexture(const D3D11_TEXTURE2D_DESC& sd) {
    if (owned_) {
      D3D11_TEXTURE2D_DESC od{};
      owned_->GetDesc(&od);
      if (od.Width == sd.Width && od.Height == sd.Height && od.Format == sd.Format) return;
    }
    D3D11_TEXTURE2D_DESC td{};
    td.Width = sd.Width;
    td.Height = sd.Height;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = sd.Format;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    owned_.Reset();
    Check(device_->CreateTexture2D(&td, nullptr, &owned_), "CreateTexture2D(采集副本)");
  }

  ID3D11Device* device_;
  ComPtr<ID3D11DeviceContext> context_;
  ComPtr<IDXGIOutput> output_;
  ComPtr<IDXGIOutputDuplication> duplication_;
  ComPtr<ID3D11Texture2D> owned_;
  POINT origin_{};
  bool holdingFrame_ = false;
};

}  // namespace

std::unique_ptr<CaptureSource> CreateDuplicationCapture(ID3D11Device* device, IDXGIOutput* output) {
  return std::make_unique<DuplicationCapture>(device, output);
}

}  // namespace nr
