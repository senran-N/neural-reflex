#pragma once

#include <cuda_runtime.h>

namespace nr {

// 实现在 cuda_check.cpp。nvcc 前端按系统代码页读取预处理后的源码，.cu 文件
// 及其包含的头文件中不能出现非 ASCII 字符串字面量（注释在预处理时已去掉，不受影响）。
void CudaCheck(cudaError_t err, const char* what);

}  // namespace nr
