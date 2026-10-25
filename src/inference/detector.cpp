#include "inference/detector.h"

#include "inference/cuda_check.h"
#include "inference/preprocess.h"

#include <stdexcept>

namespace nr {

Detector::Detector(ID3D11Device* device, int cudaDevice, const DetectorOptions& options)
    : device_(device), options_(options) {
  CudaCheck(cudaSetDevice(cudaDevice), "cudaSetDevice");
  CudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
  for (auto& e : events_) CudaCheck(cudaEventCreate(&e), "cudaEventCreate");

  engine_ = std::make_unique<TrtEngine>(options.engine);
  const TensorInfo& in = engine_->Input();
  const TensorInfo& out = engine_->Output();
  if (in.shape.size() != 4 || in.shape[0] != 1 || in.shape[1] != 3 || in.shape[2] != in.shape[3]) {
    throw std::runtime_error("模型输入须为 [1, 3, S, S]");
  }
  if (out.shape.size() != 3 || out.shape[0] != 1 || out.shape[1] <= 4) {
    throw std::runtime_error("模型输出须为 [1, 4 + nc, N]（一对多检测头）");
  }
  inputSize_ = static_cast<int>(in.shape[2]);
  post_.numClasses = static_cast<int>(out.shape[1] - 4);
  post_.numAnchors = static_cast<int>(out.shape[2]);
  post_.confThreshold = options.confThreshold;
  post_.iouThreshold = options.iouThreshold;

  buffers_ = AllocatePostprocessBuffers();
  CudaCheck(cudaMallocHost(&hostOutput_, sizeof(DetectionOutput)), "cudaMallocHost(结果)");
  fence_ = std::make_unique<SharedFence>(device);
}

Detector::~Detector() {
  if (stream_) cudaStreamSynchronize(stream_);
  texture_.reset();
  fence_.reset();
  engine_.reset();
  FreePostprocessBuffers(buffers_);
  if (hostOutput_) cudaFreeHost(hostOutput_);
  for (auto& e : events_) if (e) cudaEventDestroy(e);
  if (stream_) cudaStreamDestroy(stream_);
}

const DetectionOutput& Detector::Run(ID3D11DeviceContext* context, ID3D11Texture2D* source,
                                     const D3D11_BOX& region, float offsetX, float offsetY) {
  const uint32_t w = region.right - region.left, h = region.bottom - region.top;
  if (!texture_ || texture_->Width() != w || texture_->Height() != h) {
    // 检测区域尺寸变化时重建（例如窗口缩放），之后长期复用
    CudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    texture_.reset();
    texture_ = std::make_unique<SharedTexture>(device_, w, h);
    textureReleased_ = 0;
  }

  // D3D：等 CUDA 读完上一帧后再覆盖共享纹理，复制检测区域，然后 signal
  if (textureReleased_) fence_->D3DWait(context, textureReleased_);
  context->CopySubresourceRegion(texture_->D3D(), 0, 0, 0, 0, source, 0, &region);
  const uint64_t ready = fence_->Next();
  fence_->D3DSignal(context, ready);

  letterbox_ = ComputeLetterbox(static_cast<int>(w), static_cast<int>(h), inputSize_);
  post_.offsetX = offsetX;
  post_.offsetY = offsetY;

  fence_->CudaWait(stream_, ready);
  CudaCheck(cudaEventRecord(events_[0], stream_), "cudaEventRecord");
  LaunchLetterbox(texture_->Cuda(), letterbox_, static_cast<float*>(engine_->Input().device), stream_);
  CudaCheck(cudaGetLastError(), "LetterboxKernel");
  textureReleased_ = fence_->Next();
  fence_->CudaSignal(stream_, textureReleased_);
  CudaCheck(cudaEventRecord(events_[1], stream_), "cudaEventRecord");
  engine_->Enqueue(stream_);
  CudaCheck(cudaEventRecord(events_[2], stream_), "cudaEventRecord");
  LaunchPostprocess(static_cast<const float*>(engine_->Output().device), post_, letterbox_, buffers_, stream_);
  CudaCheck(cudaGetLastError(), "Postprocess");
  CudaCheck(cudaMemcpyAsync(hostOutput_, buffers_.output, sizeof(DetectionOutput), cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync(结果)");
  CudaCheck(cudaEventRecord(events_[3], stream_), "cudaEventRecord");
  CudaCheck(cudaEventSynchronize(events_[3]), "cudaEventSynchronize");

  cudaEventElapsedTime(&times_.preprocessMs, events_[0], events_[1]);
  cudaEventElapsedTime(&times_.inferenceMs, events_[1], events_[2]);
  cudaEventElapsedTime(&times_.postprocessMs, events_[2], events_[3]);
  return *hostOutput_;
}

void Detector::ReadInput(float* host) const {
  CudaCheck(cudaMemcpy(host, engine_->Input().device, engine_->Input().bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(输入)");
}

}  // namespace nr
