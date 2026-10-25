#include "targets.h"

#include <dwmapi.h>
#include <shellapi.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <map>

namespace nr::app {
namespace {

using Microsoft::WRL::ComPtr;

constexpr UINT kIconSize = 32;

std::string Base64(const std::vector<uint8_t>& data) {
  static constexpr char kTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((data.size() + 2) / 3 * 4);
  for (size_t i = 0; i < data.size(); i += 3) {
    const uint32_t b0 = data[i], b1 = i + 1 < data.size() ? data[i + 1] : 0, b2 = i + 2 < data.size() ? data[i + 2] : 0;
    const uint32_t v = (b0 << 16) | (b1 << 8) | b2;
    out += kTable[(v >> 18) & 63];
    out += kTable[(v >> 12) & 63];
    out += i + 1 < data.size() ? kTable[(v >> 6) & 63] : '=';
    out += i + 2 < data.size() ? kTable[v & 63] : '=';
  }
  return out;
}

std::string IconToPng(HICON icon) {
  ComPtr<IWICImagingFactory> factory;
  if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)))) return {};
  ComPtr<IWICBitmap> bitmap;
  if (FAILED(factory->CreateBitmapFromHICON(icon, &bitmap))) return {};
  ComPtr<IWICBitmapScaler> scaler;
  if (FAILED(factory->CreateBitmapScaler(&scaler)) ||
      FAILED(scaler->Initialize(bitmap.Get(), kIconSize, kIconSize, WICBitmapInterpolationModeHighQualityCubic))) {
    return {};
  }
  ComPtr<IStream> stream;
  if (FAILED(CreateStreamOnHGlobal(nullptr, TRUE, &stream))) return {};
  ComPtr<IWICBitmapEncoder> encoder;
  if (FAILED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder)) ||
      FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache))) {
    return {};
  }
  ComPtr<IWICBitmapFrameEncode> frame;
  if (FAILED(encoder->CreateNewFrame(&frame, nullptr)) || FAILED(frame->Initialize(nullptr)) ||
      FAILED(frame->WriteSource(scaler.Get(), nullptr)) || FAILED(frame->Commit()) || FAILED(encoder->Commit())) {
    return {};
  }
  HGLOBAL global = nullptr;
  if (FAILED(GetHGlobalFromStream(stream.Get(), &global))) return {};
  STATSTG stat{};
  stream->Stat(&stat, STATFLAG_NONAME);
  const auto* p = static_cast<const uint8_t*>(GlobalLock(global));
  std::vector<uint8_t> bytes(p, p + stat.cbSize.QuadPart);
  GlobalUnlock(global);
  return "data:image/png;base64," + Base64(bytes);
}

// 同一程序的多个窗口共用一份图标
std::string WindowIcon(HWND hwnd, const std::wstring& exePath) {
  static std::map<std::wstring, std::string> cache;
  if (auto it = cache.find(exePath); it != cache.end()) return it->second;

  HICON icon = nullptr;
  DWORD_PTR result = 0;
  if (SendMessageTimeoutW(hwnd, WM_GETICON, ICON_BIG, 0, SMTO_ABORTIFHUNG, 50, &result)) {
    icon = reinterpret_cast<HICON>(result);
  }
  if (!icon) icon = reinterpret_cast<HICON>(GetClassLongPtrW(hwnd, GCLP_HICON));
  HICON owned = nullptr;
  if (!icon && !exePath.empty()) {
    SHFILEINFOW info{};
    if (SHGetFileInfoW(exePath.c_str(), 0, &info, sizeof(info), SHGFI_ICON | SHGFI_LARGEICON)) owned = icon = info.hIcon;
  }
  std::string png = icon ? IconToPng(icon) : std::string();
  if (owned) DestroyIcon(owned);
  cache[exePath] = png;
  return png;
}

std::wstring ExePath(DWORD pid) {
  HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
  if (!process) return {};
  wchar_t buffer[MAX_PATH * 2];
  DWORD size = static_cast<DWORD>(std::size(buffer));
  std::wstring path;
  if (QueryFullProcessImageNameW(process, 0, buffer, &size)) path.assign(buffer, size);
  CloseHandle(process);
  return path;
}

}  // namespace

std::string ToUtf8(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(static_cast<size_t>(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

std::wstring FromUtf8(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}

std::vector<WindowTarget> ListWindows(DWORD excludePid) {
  struct Context {
    DWORD excludePid;
    std::vector<WindowTarget> items;
  } ctx{excludePid, {}};

  EnumWindows(
      [](HWND hwnd, LPARAM lp) -> BOOL {
        auto& c = *reinterpret_cast<Context*>(lp);
        if (!IsWindowVisible(hwnd) || GetWindow(hwnd, GW_OWNER)) return TRUE;
        if (GetWindowLongPtrW(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW) return TRUE;
        DWORD cloaked = 0;
        if (SUCCEEDED(DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked))) && cloaked) return TRUE;
        wchar_t title[512];
        const int len = GetWindowTextW(hwnd, title, static_cast<int>(std::size(title)));
        if (len <= 0) return TRUE;
        DWORD pid = 0;
        GetWindowThreadProcessId(hwnd, &pid);
        if (pid == c.excludePid) return TRUE;

        WindowTarget t;
        t.hwnd = hwnd;
        t.title.assign(title, static_cast<size_t>(len));
        t.pid = pid;
        t.exePath = ExePath(pid);
        t.minimized = IsIconic(hwnd) != FALSE;
        RECT client{};
        GetClientRect(hwnd, &client);
        t.clientWidth = client.right;
        t.clientHeight = client.bottom;
        t.icon = WindowIcon(hwnd, t.exePath);
        c.items.push_back(std::move(t));
        return TRUE;
      },
      reinterpret_cast<LPARAM>(&ctx));
  return ctx.items;
}

std::vector<MonitorTarget> ListMonitors() {
  std::vector<MonitorTarget> items;
  EnumDisplayMonitors(
      nullptr, nullptr,
      [](HMONITOR monitor, HDC, LPRECT, LPARAM lp) -> BOOL {
        auto& list = *reinterpret_cast<std::vector<MonitorTarget>*>(lp);
        MONITORINFOEXW info{};
        info.cbSize = sizeof(info);
        GetMonitorInfoW(monitor, &info);
        MonitorTarget m;
        m.handle = monitor;
        m.index = static_cast<int>(list.size());
        m.rect = info.rcMonitor;
        m.primary = (info.dwFlags & MONITORINFOF_PRIMARY) != 0;
        DEVMODEW mode{};
        mode.dmSize = sizeof(mode);
        if (EnumDisplaySettingsW(info.szDevice, ENUM_CURRENT_SETTINGS, &mode)) {
          m.refreshHz = static_cast<int>(mode.dmDisplayFrequency);
        }
        list.push_back(m);
        return TRUE;
      },
      reinterpret_cast<LPARAM>(&items));
  return items;
}

}  // namespace nr::app
