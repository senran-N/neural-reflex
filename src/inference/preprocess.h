#pragma once

#include "inference/letterbox.h"

#include <cuda_runtime.h>

namespace nr {

// 一次遍历完成：读取检测区域、等比例双线性缩放、填充、BGR→RGB、归一化、
// 写成 NCHW FP32（设计方案 4.3）。插值按 OpenCV INTER_LINEAR 的半像素对齐，
// 并按 8 位取整，与 Python 参考流水线一致。
void LaunchLetterbox(cudaTextureObject_t source, const LetterboxParams& params, float* output,
                     cudaStream_t stream);

}  // namespace nr
