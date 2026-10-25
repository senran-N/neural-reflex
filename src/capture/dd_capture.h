#pragma once

#include "capture/capture_source.h"

#include <dxgi.h>

#include <memory>

namespace nr {

// DXGI Desktop Duplication。output 必须属于创建 device 的适配器。
// 画面是 DWM 合成结果，覆盖层会出现在其中（设计方案 4.4）。
std::unique_ptr<CaptureSource> CreateDuplicationCapture(ID3D11Device* device, IDXGIOutput* output);

}  // namespace nr
