#include "common/win_util.h"

#include <cstdio>
#include <stdexcept>

namespace nr {

void Fail(const char* what, HRESULT hr) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s 失败：HRESULT 0x%08X", what, static_cast<unsigned>(hr));
  throw std::runtime_error(buf);
}

void EnableDpiAwareness() {
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
}

bool DisablePowerThrottling() {
  PROCESS_POWER_THROTTLING_STATE state{};
  state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  state.ControlMask =
      PROCESS_POWER_THROTTLING_EXECUTION_SPEED | PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
  state.StateMask = 0;
  return SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state,
                               sizeof(state)) != FALSE;
}

void SetCurrentThreadName(const wchar_t* name) {
  SetThreadDescription(GetCurrentThread(), name);
}

long SetGpuPriority(GpuPriority priority) {
  using Fn = LONG(APIENTRY*)(HANDLE, INT);
  static const Fn fn = reinterpret_cast<Fn>(
      GetProcAddress(GetModuleHandleW(L"gdi32.dll") ? GetModuleHandleW(L"gdi32.dll") : LoadLibraryW(L"gdi32.dll"),
                     "D3DKMTSetProcessSchedulingPriorityClass"));
  if (!fn) return -1;
  return fn(GetCurrentProcess(), static_cast<INT>(priority));
}

int GetGpuPriority() {
  using Fn = LONG(APIENTRY*)(HANDLE, INT*);
  static const Fn fn = reinterpret_cast<Fn>(
      GetProcAddress(GetModuleHandleW(L"gdi32.dll") ? GetModuleHandleW(L"gdi32.dll") : LoadLibraryW(L"gdi32.dll"),
                     "D3DKMTGetProcessSchedulingPriorityClass"));
  INT value = -1;
  if (!fn || fn(GetCurrentProcess(), &value) != 0) return -1;
  return value;
}

namespace {
constexpr const char* kGpuPriorityNames[] = {"idle", "below", "normal", "above", "high", "realtime"};
}

const char* GpuPriorityName(GpuPriority priority) { return kGpuPriorityNames[static_cast<int>(priority)]; }

bool ParseGpuPriority(const std::string& name, GpuPriority& out) {
  for (int i = 0; i < 6; ++i) {
    if (name == kGpuPriorityNames[i]) {
      out = static_cast<GpuPriority>(i);
      return true;
    }
  }
  return false;
}

}  // namespace nr
