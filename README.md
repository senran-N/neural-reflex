# neural-reflex

Windows 实时视觉检测软件。设计见 [docs/performance-design.md](docs/performance-design.md)。

当前进度：最小端到端原型的非推理部分（采集、覆盖层、合成测试源、链路探针）。推理模块待接入。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/common` | QPC 时钟、适配器选择、帧标签编解码、矩形批量绘制、最新值交接 |
| `src/capture` | Windows.Graphics.Capture（窗口 / 显示器）与 DXGI Desktop Duplication |
| `src/overlay` | DirectComposition 透明覆盖层，可等待交换链，最大帧延迟 1 |
| `apps/nr_probe` | 链路探针：采集测试源 → 解码帧标签 → 覆盖层画框，并统计延迟 |
| `tools/nr_testsource` | 合成测试源：移动目标 + 左上角帧标签（帧号、生成时刻、目标框） |

## 构建

需要 Visual Studio 2022（或 Build Tools）、Windows SDK 10.0.22621 以上、CMake 3.25 以上。

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

## 开发环境（本机）

| 组件 | 位置 |
| --- | --- |
| CUDA 13.4.3（免安装组件） | `D:\SDK\cuda-13.4.3` |
| TensorRT for RTX 1.7.1.116（CUDA 13.4） | `D:\SDK\TensorRT-RTX-1.7.1.116` |
| Nsight Systems | `D:\SDK\nsight-systems` |
