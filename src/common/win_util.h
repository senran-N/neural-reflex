#pragma once

#include <windows.h>

#include <string>

namespace nr {

[[noreturn]] void Fail(const char* what, HRESULT hr);

inline void Check(HRESULT hr, const char* what) {
  if (FAILED(hr)) Fail(what, hr);
}

// 以物理像素工作，DuplicateOutput1 也要求每显示器 DPI 感知。
void EnableDpiAwareness();

// 退出 EcoQoS 电源节流（设计方案 6.5）。
bool DisablePowerThrottling();

void SetCurrentThreadName(const wchar_t* name);

std::string ToUtf8(const std::wstring& w);
std::wstring FromUtf8(const std::string& s);

// 进程级 GPU 调度优先级（WDDM，D3DKMTSetProcessSchedulingPriorityClass），作用于本进程的所有 GPU 上下文，
// 包括 CUDA（设计方案 6.4）。返回 NTSTATUS，0 为成功。返回成功不代表按请求生效，以 GetGpuPriority() 读回为准。
enum class GpuPriority { kIdle = 0, kBelowNormal = 1, kNormal = 2, kAboveNormal = 3, kHigh = 4, kRealtime = 5 };
long SetGpuPriority(GpuPriority priority);
// 读回实际生效的级别（GpuPriority 数值），失败返回 -1。
int GetGpuPriority();
const char* GpuPriorityName(GpuPriority priority);
bool ParseGpuPriority(const std::string& name, GpuPriority& out);

}  // namespace nr
