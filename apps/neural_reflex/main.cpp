// Neural Reflex 桌面程序：Win32 窗口承载 WebView2 界面，检测会话在独立线程运行。
// 界面在 WebView2 的独立进程中渲染，并关闭其 GPU 加速，避免与检测流水线争用 GPU（设计方案 6.4）。

#include "common/win_util.h"
#include "controller.h"
#include "resource.h"

#include <WebView2.h>
#include <WebView2EnvironmentOptions.h>
#include <dwmapi.h>
#include <shellapi.h>
#include <shlobj.h>
#include <wrl.h>

#include <filesystem>
#include <memory>
#include <string>

using Microsoft::WRL::Callback;
using Microsoft::WRL::ComPtr;

namespace {

constexpr wchar_t kClassName[] = L"NeuralReflexMain";
constexpr wchar_t kHost[] = L"app.neural-reflex";
constexpr UINT_PTR kStatsTimer = 1;
constexpr COLORREF kGround = RGB(0x17, 0x17, 0x19);

struct App {
  HWND hwnd = nullptr;
  std::wstring startView;  // --view=<名称>：启动时打开的视图，便于截图检查
  std::filesystem::path appDir, dataDir;
  ComPtr<ICoreWebView2Controller> controller;
  ComPtr<ICoreWebView2> webview;
  std::unique_ptr<nr::app::Controller> bridge;
};

App g_app;

std::filesystem::path ExeDir() {
  wchar_t path[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(nullptr, path, static_cast<DWORD>(std::size(path)));
  return std::filesystem::path(std::wstring(path, n)).parent_path();
}

std::filesystem::path DataDir() {
  PWSTR local = nullptr;
  std::filesystem::path dir;
  if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &local))) dir = std::filesystem::path(local) / L"NeuralReflex";
  CoTaskMemFree(local);
  std::filesystem::create_directories(dir);
  return dir;
}

void ResizeWebView() {
  if (!g_app.controller) return;
  RECT bounds{};
  GetClientRect(g_app.hwnd, &bounds);
  g_app.controller->put_Bounds(bounds);
}

void SendToUi(const std::string& json) {
  if (!g_app.webview) return;
  const std::wstring w = nr::FromUtf8(json);
  g_app.webview->PostWebMessageAsJson(w.c_str());
}

void Fatal(const wchar_t* what, HRESULT hr) {
  wchar_t msg[512];
  swprintf_s(msg, L"%ls 失败（0x%08X）。\n\n请确认已安装 Microsoft Edge WebView2 运行时。", what, static_cast<unsigned>(hr));
  MessageBoxW(g_app.hwnd, msg, L"Neural Reflex", MB_ICONERROR);
  PostQuitMessage(1);
}

HRESULT OnControllerCreated(HRESULT result, ICoreWebView2Controller* controller) {
  if (FAILED(result) || !controller) {
    Fatal(L"创建 WebView2 控件", result);
    return S_OK;
  }
  g_app.controller = controller;
  controller->get_CoreWebView2(&g_app.webview);

  ComPtr<ICoreWebView2Controller2> controller2;
  if (SUCCEEDED(controller->QueryInterface(IID_PPV_ARGS(&controller2)))) {
    controller2->put_DefaultBackgroundColor({255, 0x17, 0x17, 0x19});
  }

  ComPtr<ICoreWebView2Settings> settings;
  g_app.webview->get_Settings(&settings);
  wchar_t devtools[8] = {};
  const bool allowDevtools = GetEnvironmentVariableW(L"NR_DEVTOOLS", devtools, 8) > 0;
  settings->put_AreDevToolsEnabled(allowDevtools);
  settings->put_AreDefaultContextMenusEnabled(allowDevtools);
  settings->put_IsStatusBarEnabled(FALSE);
  settings->put_IsZoomControlEnabled(FALSE);
  ComPtr<ICoreWebView2Settings3> settings3;
  if (SUCCEEDED(settings.As(&settings3))) settings3->put_AreBrowserAcceleratorKeysEnabled(allowDevtools);

  ComPtr<ICoreWebView2_3> webview3;
  if (FAILED(g_app.webview.As(&webview3))) {
    Fatal(L"WebView2 运行时版本过旧", E_NOINTERFACE);
    return S_OK;
  }
  const std::wstring uiDir = (g_app.appDir / L"ui").wstring();
  webview3->SetVirtualHostNameToFolderMapping(kHost, uiDir.c_str(), COREWEBVIEW2_HOST_RESOURCE_ACCESS_KIND_DENY_CORS);

  g_app.bridge = std::make_unique<nr::app::Controller>(SendToUi, g_app.appDir, g_app.dataDir);

  EventRegistrationToken token{};
  g_app.webview->add_WebMessageReceived(
      Callback<ICoreWebView2WebMessageReceivedEventHandler>(
          [](ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
            // 只接受本程序界面发来的消息
            LPWSTR source = nullptr;
            args->get_Source(&source);
            const bool trusted = source && std::wstring(source).rfind(L"https://app.neural-reflex/", 0) == 0;
            CoTaskMemFree(source);
            if (!trusted) return S_OK;
            LPWSTR message = nullptr;
            if (SUCCEEDED(args->TryGetWebMessageAsString(&message)) && message) {
              g_app.bridge->OnMessage(nr::ToUtf8(message), g_app.hwnd);
              CoTaskMemFree(message);
            }
            return S_OK;
          })
          .Get(),
      &token);

  // 界面只加载本地页面，拒绝跳转到其他地址
  g_app.webview->add_NavigationStarting(
      Callback<ICoreWebView2NavigationStartingEventHandler>(
          [](ICoreWebView2*, ICoreWebView2NavigationStartingEventArgs* args) -> HRESULT {
            LPWSTR uri = nullptr;
            args->get_Uri(&uri);
            if (!uri || std::wstring(uri).rfind(L"https://app.neural-reflex/", 0) != 0) args->put_Cancel(TRUE);
            CoTaskMemFree(uri);
            return S_OK;
          })
          .Get(),
      &token);
  g_app.webview->add_NewWindowRequested(
      Callback<ICoreWebView2NewWindowRequestedEventHandler>(
          [](ICoreWebView2*, ICoreWebView2NewWindowRequestedEventArgs* args) -> HRESULT {
            args->put_Handled(TRUE);
            return S_OK;
          })
          .Get(),
      &token);

  ResizeWebView();
  const std::wstring url = L"https://app.neural-reflex/index.html" + (g_app.startView.empty() ? L"" : L"#/" + g_app.startView);
  g_app.webview->Navigate(url.c_str());
  SetTimer(g_app.hwnd, kStatsTimer, 250, nullptr);
  return S_OK;
}

void CreateWebView() {
  auto options = Microsoft::WRL::Make<CoreWebView2EnvironmentOptions>();
  // 界面以软件方式渲染：每秒只更新几次，换取不占用检测所用的 GPU
  options->put_AdditionalBrowserArguments(L"--disable-gpu --disable-gpu-compositing");
  const std::wstring userData = (g_app.dataDir / L"WebView2").wstring();
  const HRESULT hr = CreateCoreWebView2EnvironmentWithOptions(
      nullptr, userData.c_str(), options.Get(),
      Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
          [](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {
            if (FAILED(result) || !env) {
              Fatal(L"创建 WebView2 环境", result);
              return S_OK;
            }
            env->CreateCoreWebView2Controller(
                g_app.hwnd, Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(OnControllerCreated).Get());
            return S_OK;
          })
          .Get());
  if (FAILED(hr)) Fatal(L"启动 WebView2", hr);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  switch (msg) {
    case WM_SIZE:
      ResizeWebView();
      if (g_app.controller) g_app.controller->put_IsVisible(wp != SIZE_MINIMIZED);
      return 0;
    case WM_TIMER:
      if (wp == kStatsTimer && g_app.bridge) g_app.bridge->OnTimer();
      return 0;
    case WM_GETMINMAXINFO: {
      auto* info = reinterpret_cast<MINMAXINFO*>(lp);
      const UINT dpi = GetDpiForWindow(hwnd);
      info->ptMinTrackSize = {MulDiv(980, dpi, 96), MulDiv(640, dpi, 96)};
      return 0;
    }
    case WM_DPICHANGED: {
      const RECT* r = reinterpret_cast<const RECT*>(lp);
      SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
      return 0;
    }
    case WM_DESTROY:
      KillTimer(hwnd, kStatsTimer);
      g_app.bridge.reset();  // 停止会话并保存设置
      if (g_app.controller) g_app.controller->Close();
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR cmdLine, int show) {
  if (const wchar_t* v = wcsstr(cmdLine, L"--view=")) {
    for (v += 7; iswalpha(*v); ++v) g_app.startView += *v;
  }
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  nr::DisablePowerThrottling();
  if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED))) return 1;

  g_app.appDir = ExeDir();
  // --data-dir=<目录>：使用单独的数据目录（设置、引擎缓存、插件数据与日志），便于测试时不影响日常使用的配置
  int argc = 0;
  if (wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc)) {
    for (int i = 1; i < argc; ++i) {
      const std::wstring a = argv[i];
      if (a.rfind(L"--data-dir=", 0) == 0) g_app.dataDir = std::filesystem::absolute(a.substr(11));
    }
    LocalFree(argv);
  }
  if (g_app.dataDir.empty()) {
    g_app.dataDir = DataDir();
  } else {
    std::filesystem::create_directories(g_app.dataDir);
  }

  WNDCLASSEXW wc{sizeof(wc)};
  wc.lpfnWndProc = WndProc;
  wc.hInstance = instance;
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.hbrBackground = CreateSolidBrush(kGround);
  wc.hIcon = LoadIconW(instance, MAKEINTRESOURCEW(IDI_APP));
  wc.hIconSm = LoadIconW(instance, MAKEINTRESOURCEW(IDI_APP));
  wc.lpszClassName = kClassName;
  RegisterClassExW(&wc);

  const UINT dpi = GetDpiForSystem();
  g_app.hwnd = CreateWindowExW(0, kClassName, L"Neural Reflex", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                               MulDiv(1360, dpi, 96), MulDiv(860, dpi, 96), nullptr, nullptr, instance, nullptr);
  if (!g_app.hwnd) return 1;
  const BOOL dark = TRUE;
  DwmSetWindowAttribute(g_app.hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &dark, sizeof(dark));
  const COLORREF caption = kGround;
  DwmSetWindowAttribute(g_app.hwnd, DWMWA_CAPTION_COLOR, &caption, sizeof(caption));
  ShowWindow(g_app.hwnd, show);
  UpdateWindow(g_app.hwnd);

  CreateWebView();

  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }
  CoUninitialize();
  return static_cast<int>(msg.wParam);
}
