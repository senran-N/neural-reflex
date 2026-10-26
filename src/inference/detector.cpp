#include "inference/detector.h"

#include "inference/cuda_check.h"
#include "inference/preprocess.h"

#include <stdexcept>

namespace nr {

Detector::Detector(ID3D11Device* device, int cudaDevice, const DetectorOptions& options)
    : device_(device), options_(options) {
  CudaCheck(cudaSetDevice(cudaDevice), "cudaSetDevice");
  CudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");

  // 清单与预处理声明在构建引擎之前检查，避免为不支持的模型花时间编译
  manifest_ = LoadModelManifest(options.engine.onnx);
  CheckSupportedPreprocess(manifest_);

  engine_ = std::make_unique<TrtEngine>(options.engine);
  const TensorInfo& in = engine_->Input();
  if (in.shape.size() != 4 || in.shape[0] != 1 || in.shape[1] != 3 || in.shape[2] != in.shape[3]) {
    throw std::runtime_error("模型输入须为 [1, 3, S, S]");
  }
  inputSize_ = static_cast<int>(in.shape[2]);
  decoder_ = CreateDecoder(manifest_, engine_->Output().shape);
  decode_.confThreshold = options.confThreshold;
  decode_.iouThreshold = options.iouThreshold;

  CudaCheck(cudaMalloc(&deviceOutput_, sizeof(DetectionOutput)), "cudaMalloc(结果)");
  for (Slot& s : slots_) {
    CudaCheck(cudaMallocHost(&s.host, sizeof(DetectionOutput)), "cudaMallocHost(结果)");
    for (auto& e : s.events) CudaCheck(cudaEventCreate(&e), "cudaEventCreate");
  }
  fence_ = std::make_unique<SharedFence>(device);
}

Detector::~Detector() {
  if (stream_) cudaStreamSynchronize(stream_);
  for (Slot& s : slots_) {
    s.texture.reset();
    if (s.host) cudaFreeHost(s.host);
    for (auto& e : s.events) if (e) cudaEventDestroy(e);
  }
  fence_.reset();
  decoder_.reset();
  engine_.reset();
  if (deviceOutput_) cudaFree(deviceOutput_);
  if (stream_) cudaStreamDestroy(stream_);
}

int Detector::Submit(ID3D11DeviceContext* context, ID3D11Texture2D* source, const D3D11_BOX& region,
                     float offsetX, float offsetY, int slotIndex) {
  Slot& slot = slots_[slotIndex];
  const uint32_t w = region.right - region.left, h = region.bottom - region.top;
  if (!slot.texture || slot.texture->Width() != w || slot.texture->Height() != h) {
    // 检测区域尺寸变化时重建（例如窗口缩放），之后长期复用
    CudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    slot.texture.reset();
    slot.texture = std::make_unique<SharedTexture>(device_, w, h);
    slot.released = 0;
  }

  // D3D：等 CUDA 读完该槽位上一帧后再覆盖，复制检测区域（全程唯一的一次画面复制），然后 signal
  if (slot.released) fence_->D3DWait(context, slot.released);
  context->CopySubresourceRegion(slot.texture->D3D(), 0, 0, 0, 0, source, 0, &region);
  const uint64_t ready = fence_->Next();
  fence_->D3DSignal(context, ready);

  DecodeParams decode = decode_;
  decode.letterbox = ComputeLetterbox(static_cast<int>(w), static_cast<int>(h), inputSize_);
  decode.offsetX = offsetX;
  decode.offsetY = offsetY;

  fence_->CudaWait(stream_, ready);
  CudaCheck(cudaEventRecord(slot.events[0], stream_), "cudaEventRecord");
  LaunchLetterbox(slot.texture->Cuda(), decode.letterbox, static_cast<float*>(engine_->Input().device), stream_);
  CudaCheck(cudaGetLastError(), "LetterboxKernel");
  slot.released = fence_->Next();
  fence_->CudaSignal(stream_, slot.released);
  CudaCheck(cudaEventRecord(slot.events[1], stream_), "cudaEventRecord");
  engine_->Enqueue(stream_);
  CudaCheck(cudaEventRecord(slot.events[2], stream_), "cudaEventRecord");
  decoder_->Enqueue(static_cast<const float*>(engine_->Output().device), decode, deviceOutput_, stream_);
  CudaCheck(cudaGetLastError(), "解码器");
  CudaCheck(cudaMemcpyAsync(slot.host, deviceOutput_, kDetectionOutputHeadBytes, cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync(结果)");
  if (const int k = decoder_->NumKeypoints(); k > 0) {
    CudaCheck(cudaMemcpyAsync(slot.host->keypoints, deviceOutput_->keypoints,
                              sizeof(Keypoint) * kMaxDetections * static_cast<size_t>(k), cudaMemcpyDeviceToHost,
                              stream_),
              "cudaMemcpyAsync(关键点)");
  }
  CudaCheck(cudaEventRecord(slot.events[3], stream_), "cudaEventRecord");
  return slotIndex;
}

const DetectionOutput& Detector::Wait(int slotIndex, StageTimes* times) {
  Slot& slot = slots_[slotIndex];
  CudaCheck(cudaEventSynchronize(slot.events[3]), "cudaEventSynchronize");
  if (times) {
    cudaEventElapsedTime(&times->preprocessMs, slot.events[0], slot.events[1]);
    cudaEventElapsedTime(&times->inferenceMs, slot.events[1], slot.events[2]);
    cudaEventElapsedTime(&times->postprocessMs, slot.events[2], slot.events[3]);
  }
  return *slot.host;
}

const DetectionOutput& Detector::Run(ID3D11DeviceContext* context, ID3D11Texture2D* source, const D3D11_BOX& region,
                                     float offsetX, float offsetY) {
  Submit(context, source, region, offsetX, offsetY, 0);
  return Wait(0);
}

void Detector::ReadInput(float* host) const {
  CudaCheck(cudaMemcpy(host, engine_->Input().device, engine_->Input().bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(输入)");
}

}  // namespace nr
