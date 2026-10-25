# neural-reflex

Windows 实时视觉检测软件。设计见 [docs/performance-design.md](docs/performance-design.md)。

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
| `apps/neural_reflex` | 桌面程序：WebView2 界面宿主、目标枚举、设置持久化 |
| `src/pipeline` | 检测会话：工作线程、覆盖层线程、每秒统计 |
| `src/inference` | D3D11/CUDA 交接、GPU 预处理与后处理、TensorRT for RTX 引擎 |
| `ui` | 界面（HTML/CSS/JS）；在浏览器中直接打开时使用示例数据 |
| `apps/nr_detect` | 命令行检测：实时模式，或 `--image-list` 离线模式用于与 Python 参考对比 |
| `apps/nr_probe` | 链路探针：采集测试源 → 解码帧标签 → 覆盖层画框，并统计延迟 |
| `tools/nr_testsource` | 合成测试源：移动目标 + 左上角帧标签（帧号、生成时刻、目标框） |
| `tools/model_export` | 参考模型（YOLO26n）导出为 ONNX 并记录输入输出契约 |

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

模型、引擎和缓存不入库（体积大，且引擎与设备、驱动绑定）。Ultralytics 权重为 AGPL-3.0。

## 开发环境（本机）

| 组件 | 位置 |
| --- | --- |
| CUDA 13.4.3（免安装组件） | `D:\SDK\cuda-13.4.3` |
| TensorRT for RTX 1.7.1.116（CUDA 13.4） | `D:\SDK\TensorRT-RTX-1.7.1.116` |
| Nsight Systems | `D:\SDK\nsight-systems` |
| 模型导出 Python 环境 | `D:\SDK\venvs\nr-export` |
