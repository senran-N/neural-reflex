#pragma once

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class ICudaEngine;
class IExecutionContext;
class IRuntime;
class IRuntimeCache;
class IRuntimeConfig;
}  // namespace nvinfer1

namespace nr {

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  void* device = nullptr;  // 启动时分配，运行期间地址不变（设计方案 3.3）
  size_t bytes = 0;
  bool input = false;
};

struct EngineOptions {
  std::filesystem::path onnx;
  std::filesystem::path cacheDir;  // 存放序列化引擎与运行时缓存
  bool cudaGraph = true;
};

// TensorRT for RTX：启动时完成构建（或读取缓存引擎）、即时编译预热和缓冲分配，
// 实时路径只调用 Enqueue。仅支持 FP32 输入输出的静态形状模型。
class TrtEngine {
 public:
  explicit TrtEngine(const EngineOptions& options);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  const TensorInfo& Input() const { return tensors_[inputIndex_]; }
  const TensorInfo& Output() const { return tensors_[outputIndex_]; }

  void Enqueue(cudaStream_t stream);

  // 把即时编译产生的内核写回运行时缓存文件，下次启动免编译
  void SaveRuntimeCache();

 private:
  std::filesystem::path runtimeCachePath_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IRuntimeConfig> runtimeConfig_;
  std::unique_ptr<nvinfer1::IRuntimeCache> runtimeCache_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<TensorInfo> tensors_;
  size_t inputIndex_ = 0, outputIndex_ = 0;
};

}  // namespace nr
