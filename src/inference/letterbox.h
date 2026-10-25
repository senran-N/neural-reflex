#pragma once

#include <algorithm>
#include <cmath>

namespace nr {

// 等比例缩放并居中填充的参数，与 tools/model_export/reference_pipeline.py 的 Letterbox 一致。
// 主机端计算，按值传给 CUDA 内核。
struct LetterboxParams {
  int srcW = 0, srcH = 0;  // 检测区域尺寸（共享纹理左上角起）
  int size = 0;            // 模型输入边长
  int newW = 0, newH = 0;  // 缩放后尺寸
  int left = 0, top = 0;   // 填充偏移
  float scale = 1.0f;
};

inline LetterboxParams ComputeLetterbox(int srcW, int srcH, int size) {
  LetterboxParams p;
  p.srcW = srcW;
  p.srcH = srcH;
  p.size = size;
  // 用 double 和就近取偶（std::nearbyint 默认舍入模式）对齐 Python 的 round()
  const double scale = std::min(static_cast<double>(size) / srcH, static_cast<double>(size) / srcW);
  p.scale = static_cast<float>(scale);
  p.newW = static_cast<int>(std::nearbyint(srcW * scale));
  p.newH = static_cast<int>(std::nearbyint(srcH * scale));
  const double dw = (size - p.newW) / 2.0, dh = (size - p.newH) / 2.0;
  p.top = static_cast<int>(std::nearbyint(dh - 0.1));
  p.left = static_cast<int>(std::nearbyint(dw - 0.1));
  return p;
}

}  // namespace nr
