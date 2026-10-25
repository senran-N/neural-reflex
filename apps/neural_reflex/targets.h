#pragma once

#include <windows.h>

#include <string>
#include <vector>

namespace nr::app {

struct WindowTarget {
  HWND hwnd = nullptr;
  std::wstring title;
  std::wstring exePath;
  DWORD pid = 0;
  int clientWidth = 0, clientHeight = 0;
  bool minimized = false;
  std::string icon;  // PNG data URL，取不到时为空
};

struct MonitorTarget {
  HMONITOR handle = nullptr;
  int index = 0;
  RECT rect{};
  bool primary = false;
  int refreshHz = 0;
};

// 与任务切换列表相同的规则：可见、无所有者、非工具窗口、未被隐藏（cloaked）、有标题。
std::vector<WindowTarget> ListWindows(DWORD excludePid);
std::vector<MonitorTarget> ListMonitors();

std::string ToUtf8(const std::wstring& w);
std::wstring FromUtf8(const std::string& s);

}  // namespace nr::app
