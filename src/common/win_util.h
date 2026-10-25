#pragma once

#include <windows.h>

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

}  // namespace nr
