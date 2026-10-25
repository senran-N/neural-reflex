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

}  // namespace nr
