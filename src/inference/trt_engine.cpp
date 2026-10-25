#include "inference/trt_engine.h"

#include "inference/cuda_check.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace nr {
namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT-RTX] %s\n", msg);
  }
};

Logger& GetLogger() {
  static Logger logger;
  return logger;
}

std::vector<char> ReadFile(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

void WriteFile(const std::filesystem::path& path, const void* data, size_t size) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
  if (!f) throw std::runtime_error("写入失败：" + path.string());
}

std::vector<char> BuildEngine(const std::filesystem::path& onnx) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(GetLogger()));
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, GetLogger()));
  if (!parser->parseFromFile(onnx.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("ONNX 解析失败：" + onnx.string());
  }
  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) throw std::runtime_error("引擎构建失败");
  const auto* p = static_cast<const char*>(plan->data());
  return std::vector<char>(p, p + plan->size());
}

}  // namespace

TrtEngine::TrtEngine(const EngineOptions& options) {
  std::filesystem::create_directories(options.cacheDir);
  const auto stem = options.onnx.stem().string();
  const auto enginePath = options.cacheDir / (stem + ".engine");
  runtimeCachePath_ = options.cacheDir / (stem + ".runtime-cache");

  // 引擎缓存比 ONNX 旧时重建（设计方案 3.1：失配时重建）
  std::vector<char> plan;
  if (std::filesystem::exists(enginePath) &&
      std::filesystem::last_write_time(enginePath) >= std::filesystem::last_write_time(options.onnx)) {
    plan = ReadFile(enginePath);
  }
  if (plan.empty()) {
    plan = BuildEngine(options.onnx);
    WriteFile(enginePath, plan.data(), plan.size());
  }

  runtime_.reset(nvinfer1::createInferRuntime(GetLogger()));
  engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine_) {
    // 运行时或驱动版本变化导致缓存失效，重建一次
    plan = BuildEngine(options.onnx);
    WriteFile(enginePath, plan.data(), plan.size());
    engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    if (!engine_) throw std::runtime_error("引擎反序列化失败");
  }

  runtimeConfig_.reset(engine_->createRuntimeConfig());
  runtimeCache_.reset(runtimeConfig_->createRuntimeCache());
  const auto cacheBlob = ReadFile(runtimeCachePath_);
  if (!cacheBlob.empty() && !runtimeCache_->deserialize(cacheBlob.data(), cacheBlob.size())) {
    std::fprintf(stderr, "运行时缓存无效，将重新编译\n");
  }
  runtimeConfig_->setRuntimeCache(*runtimeCache_);
  runtimeConfig_->setCudaGraphStrategy(options.cudaGraph ? nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE
                                                         : nvinfer1::CudaGraphStrategy::kDISABLED);
  context_.reset(engine_->createExecutionContext(runtimeConfig_.get()));
  if (!context_) throw std::runtime_error("创建执行上下文失败");

  bool haveInput = false, haveOutput = false;
  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    TensorInfo t;
    t.name = engine_->getIOTensorName(i);
    if (engine_->getTensorDataType(t.name.c_str()) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("仅支持 FP32 输入输出：" + t.name);
    }
    const nvinfer1::Dims dims = engine_->getTensorShape(t.name.c_str());
    size_t elements = 1;
    for (int32_t d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] < 0) throw std::runtime_error("仅支持静态形状：" + t.name);
      t.shape.push_back(dims.d[d]);
      elements *= static_cast<size_t>(dims.d[d]);
    }
    t.bytes = elements * sizeof(float);
    t.input = engine_->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    CudaCheck(cudaMalloc(&t.device, t.bytes), "cudaMalloc(张量)");
    if (!context_->setTensorAddress(t.name.c_str(), t.device)) {
      throw std::runtime_error("setTensorAddress 失败：" + t.name);
    }
    if (t.input && !haveInput) {
      inputIndex_ = tensors_.size();
      haveInput = true;
    } else if (!t.input && !haveOutput) {
      outputIndex_ = tensors_.size();
      haveOutput = true;
    }
    tensors_.push_back(std::move(t));
  }
  if (!haveInput || !haveOutput) throw std::runtime_error("模型缺少输入或输出");
}

TrtEngine::~TrtEngine() {
  context_.reset();
  for (auto& t : tensors_) cudaFree(t.device);
}

void TrtEngine::Enqueue(cudaStream_t stream) {
  if (!context_->enqueueV3(stream)) throw std::runtime_error("enqueueV3 失败");
}

void TrtEngine::SaveRuntimeCache() {
  std::unique_ptr<nvinfer1::IHostMemory> blob(runtimeCache_->serialize());
  if (blob && blob->size() > 0) WriteFile(runtimeCachePath_, blob->data(), blob->size());
}

}  // namespace nr
