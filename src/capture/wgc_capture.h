#pragma once

#include "capture/capture_source.h"

#include <memory>

namespace nr {

// Windows.Graphics.Capture。窗口模式只采集目标窗口，覆盖层不会进入画面（设计方案 4.1）。
// 需在已初始化 WinRT 多线程套间的线程上创建。
std::unique_ptr<CaptureSource> CreateWgcWindowCapture(ID3D11Device* device, HWND window);
std::unique_ptr<CaptureSource> CreateWgcMonitorCapture(ID3D11Device* device, HMONITOR monitor);

}  // namespace nr
