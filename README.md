# neural-reflex

Windows 实时视觉检测软件。设计见 [docs/performance-design.md](docs/performance-design.md)，扩展机制（模型适配、结果共享）见 [docs/plugin-design.md](docs/plugin-design.md)。

当前进度：完整链路已打通（采集 → D3D11/CUDA 交接 → GPU 预处理 → TensorRT for RTX → GPU NMS → 覆盖层），并提供桌面程序 `NeuralReflex.exe`。

## 桌面程序

`NeuralReflex.exe`：选择采集目标（任意窗口或整个显示器）、加载 ONNX 模型、调整阈值与采集方式、开始和停止检测，并实时查看延迟、帧预算、显存与每类框数。界面为 WebView2（Windows 11 自带运行时），设置与引擎缓存保存在 `%LOCALAPPDATA%\NeuralReflex`。

打包发布目录：

```
cmake --install build --config Release --prefix dist/NeuralReflex
```

首次使用一个模型时，TensorRT for RTX 需要编译引擎（数秒到数十秒），之后从缓存加载。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/common` | QPC 时钟、适配器选择、帧标签编解码、矩形批量绘制、最新值交接 |
| `src/capture` | Windows.Graphics.Capture（窗口 / 显示器）与 DXGI Desktop Duplication |
| `src/overlay` | DirectComposition 透明覆盖层，可等待交换链，最大帧延迟 1 |
| `apps/neural_reflex` | 桌面程序：WebView2 界面宿主、目标枚举、设置持久化、插件托管 |
| `src/pipeline` | 检测会话：工作线程、覆盖层线程、每秒统计；结果发布到共享内存 |
| `src/inference` | D3D11/CUDA 交接、GPU 预处理、TensorRT for RTX 引擎、模型清单与输出解码器 |
| `ui` | 界面（HTML/CSS/JS）；在浏览器中直接打开时使用示例数据 |
| `apps/nr_detect` | 命令行检测：实时模式，或 `--image-list` 离线模式用于与 Python 参考对比 |
| `apps/nr_probe` | 链路探针：采集测试源 → 解码帧标签 → 覆盖层画框，并统计延迟 |
| `tools/nr_testsource` | 合成测试源：移动目标 + 左上角帧标签（帧号、生成时刻、目标框） |
| `tools/model_export` | 模型导出为 ONNX 并写出模型清单；Python 参考流水线与精度对比 |
| `tools/results` | 结果共享的读取示例（只依赖 Python 标准库） |
| `plugins`、`sdk/python` | 示例插件与 Python 插件 SDK |
| `tools/bench`、`tools/nr_gpuload` | 并行负载下的检测测量与可控 GPU 负载 |

## 构建

需要 Visual Studio 2022（或 Build Tools）、Windows SDK 10.0.22621 以上、CMake 3.25 以上，以及环境变量 `CUDA_PATH`（CUDA 13.x）、`TENSORRT_RTX_ROOT`、`WEBVIEW2_ROOT`（Microsoft.Web.WebView2 NuGet 包解压目录）。不需要推理模块时可用 `vs2022-nocuda` 预设。

```
cmake --preset vs2022
cmake --build --preset release
```

产物位于 `build/bin/Release`。

## 运行链路验证

```
build\bin\Release\nr_testsource.exe --fps 144
build\bin\Release\nr_probe.exe --mode wgc --duration 30
```

`--mode` 可选 `wgc`（窗口）、`wgc-monitor`、`dd`。探针输出每秒统计和汇总分位数；`--csv` 可导出逐帧记录，测试源的 `--log` 可导出生成记录，两者按 `frame_id` 关联。

探针画出的框来自测试源标签里的真值，用于验证采集和坐标还原，不是检测结果。

## 参考模型

```
cd models/reference
<导出环境>/python ../../tools/model_export/export_reference.py --out .
tensorrt_rtx.exe --onnx=yolo26n-640-nms-fp32.onnx --saveEngine=yolo26n-640-nms-fp32.engine
```

导出脚本同时写出模型清单（同名 `.json`）：解码器、类别名、关键点等。`--weights yolo26n-pose.pt` 导出姿态模型，`--end2end` 导出无 NMS 的端到端头。目前支持的解码器为 `yolo-anchors`、`yolo-end2end`、`yolo-pose-anchors`（扩展机制方案 4.4）。

模型、引擎和缓存不入库（体积大，且引擎与设备、驱动绑定）。Ultralytics 权重为 AGPL-3.0。

## 结果共享

桌面程序“设置 → 结果共享”或 `nr_detect --publish` 开启后，每帧结果写入共享内存 `Local\NeuralReflex.Results.v1`，任意语言的程序都可读取，布局见扩展机制方案 5.2。示例：

```
python tools/results/read_results.py --print
```

## 插件

插件是一个含 `plugin.json` 的文件夹，由桌面程序启动和托管，通过结果共享读取每帧结果，在“插件”页中启用、配置、查看状态与日志。程序目录与源码树的 `plugins`（示例：`plugins/target-alert`）和 `%LOCALAPPDATA%\NeuralReflex\plugins` 中的插件都会被发现。清单格式、环境变量与输出协议见扩展机制方案 5.5；Python 插件可使用 `sdk/python/neural_reflex.py`。

## 开发环境（本机）

| 组件 | 位置 |
| --- | --- |
| CUDA 13.4.3（免安装组件） | `D:\SDK\cuda-13.4.3` |
| TensorRT for RTX 1.7.1.116（CUDA 13.4） | `D:\SDK\TensorRT-RTX-1.7.1.116` |
| Nsight Systems | `D:\SDK\nsight-systems` |
| 模型导出 Python 环境 | `D:\SDK\venvs\nr-export` |

## 许可证

本项目以 [GNU Affero General Public License v3.0](LICENSE)（AGPL-3.0）发布。参考模型 Ultralytics YOLO26n 的权重同为 AGPL-3.0；其他第三方组件的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

AGPL-3.0 要求：分发本程序（含修改版）时须同时提供对应源代码；若修改后通过网络向他人提供服务，也须向这些用户提供源代码。
