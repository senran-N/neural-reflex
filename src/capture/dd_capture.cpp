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
    DXGI_OUTPUT_DESC desc{};
    output_->GetDesc(&desc);
    origin_ = {desc.DesktopCoordinates.left, desc.DesktopCoordinates.top};
    Check(Duplicate(), "DuplicateOutput");
  }

  ~DuplicationCapture() override { ReleaseFrame(); }

  const char* Name() const override { return "Desktop Duplication"; }

  AcquireStatus AcquireLatest(uint32_t timeoutMs, CapturedFrame& out) override {
    ReleaseFrame();
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
        ReleaseFrame();
        if (QpcNow() >= deadline) return AcquireStatus::kTimeout;
        continue;
      }

      surface_.Reset();
      Check(resource.As(&surface_), "IDXGIResource -> ID3D11Texture2D");
      D3D11_TEXTURE2D_DESC sd{};
      surface_->GetDesc(&sd);

      // 直接交出桌面帧，使用方复制所需区域后调用 ReleaseFrame()（设计方案 4.2）
      out.texture = surface_.Get();
      out.width = sd.Width;
      out.height = sd.Height;
      out.screenOrigin = origin_;
      out.sourceQpc = info.LastPresentTime.QuadPart;
      out.acquireQpc = acquireQpc;
      out.accumulated = info.AccumulatedFrames;
      return AcquireStatus::kNewFrame;
    }
  }

  void ReleaseFrame() override {
    surface_.Reset();
    if (holdingFrame_ && duplication_) duplication_->ReleaseFrame();
    holdingFrame_ = false;
  }

  bool Recreate() override {
    ReleaseFrame();
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

  ID3D11Device* device_;
  ComPtr<IDXGIOutput> output_;
  ComPtr<IDXGIOutputDuplication> duplication_;
  ComPtr<ID3D11Texture2D> surface_;
  POINT origin_{};
  bool holdingFrame_ = false;
};

}  // namespace

std::unique_ptr<CaptureSource> CreateDuplicationCapture(ID3D11Device* device, IDXGIOutput* output) {
  return std::make_unique<DuplicationCapture>(device, output);
}

}  // namespace nr
