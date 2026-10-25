#include "inference/d3d_cuda_interop.h"

#include "common/win_util.h"
#include "inference/cuda_check.h"

#include <cstring>
#include <stdexcept>

namespace nr {

int CudaDeviceForLuid(const LUID& luid) {
  int count = 0;
  CudaCheck(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  for (int i = 0; i < count; ++i) {
    cudaDeviceProp prop{};
    CudaCheck(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties");
    if (std::memcmp(prop.luid, &luid, sizeof(luid)) == 0) return i;
  }
  throw std::runtime_error("没有与 D3D 适配器 LUID 匹配的 CUDA 设备");
}

SharedTexture::SharedTexture(ID3D11Device* device, uint32_t width, uint32_t height)
    : width_(width), height_(height) {
  D3D11_TEXTURE2D_DESC td{};
  td.Width = width;
  td.Height = height;
  td.MipLevels = 1;
  td.ArraySize = 1;
  td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  td.SampleDesc.Count = 1;
  td.Usage = D3D11_USAGE_DEFAULT;
  td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  td.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
  Check(device->CreateTexture2D(&td, nullptr, &texture_), "CreateTexture2D(共享纹理)");

  Microsoft::WRL::ComPtr<IDXGIResource1> resource;
  Check(texture_.As(&resource), "ID3D11Texture2D -> IDXGIResource1");
  Check(resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                     nullptr, &sharedHandle_),
        "CreateSharedHandle(纹理)");

  cudaExternalMemoryHandleDesc memDesc{};
  memDesc.type = cudaExternalMemoryHandleTypeD3D11Resource;
  memDesc.handle.win32.handle = sharedHandle_;
  memDesc.size = static_cast<unsigned long long>(width) * height * 4;
  memDesc.flags = cudaExternalMemoryDedicated;
  CudaCheck(cudaImportExternalMemory(&externalMemory_, &memDesc), "cudaImportExternalMemory");

  cudaExternalMemoryMipmappedArrayDesc arrayDesc{};
  arrayDesc.formatDesc = cudaCreateChannelDesc<uchar4>();
  arrayDesc.extent = make_cudaExtent(width, height, 0);
  arrayDesc.numLevels = 1;
  CudaCheck(cudaExternalMemoryGetMappedMipmappedArray(&mipmappedArray_, externalMemory_, &arrayDesc),
            "cudaExternalMemoryGetMappedMipmappedArray");
  cudaArray_t level0 = nullptr;
  CudaCheck(cudaGetMipmappedArrayLevel(&level0, mipmappedArray_, 0), "cudaGetMipmappedArrayLevel");

  cudaResourceDesc res{};
  res.resType = cudaResourceTypeArray;
  res.res.array.array = level0;
  cudaTextureDesc tex{};
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModePoint;
  tex.readMode = cudaReadModeElementType;
  tex.normalizedCoords = 0;
  CudaCheck(cudaCreateTextureObject(&textureObject_, &res, &tex, nullptr), "cudaCreateTextureObject");
}

SharedTexture::~SharedTexture() {
  if (textureObject_) cudaDestroyTextureObject(textureObject_);
  if (mipmappedArray_) cudaFreeMipmappedArray(mipmappedArray_);
  if (externalMemory_) cudaDestroyExternalMemory(externalMemory_);
  if (sharedHandle_) CloseHandle(sharedHandle_);
}

SharedFence::SharedFence(ID3D11Device* device) {
  Microsoft::WRL::ComPtr<ID3D11Device5> device5;
  Check(device->QueryInterface(IID_PPV_ARGS(&device5)), "ID3D11Device -> ID3D11Device5");
  Check(device5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_)), "CreateFence");
  Check(fence_->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &sharedHandle_), "CreateSharedHandle(fence)");

  cudaExternalSemaphoreHandleDesc desc{};
  desc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
  desc.handle.win32.handle = sharedHandle_;
  CudaCheck(cudaImportExternalSemaphore(&semaphore_, &desc), "cudaImportExternalSemaphore");
}

SharedFence::~SharedFence() {
  if (semaphore_) cudaDestroyExternalSemaphore(semaphore_);
  if (sharedHandle_) CloseHandle(sharedHandle_);
}

void SharedFence::D3DSignal(ID3D11DeviceContext* context, uint64_t value) {
  Microsoft::WRL::ComPtr<ID3D11DeviceContext4> context4;
  Check(context->QueryInterface(IID_PPV_ARGS(&context4)), "ID3D11DeviceContext4");
  Check(context4->Signal(fence_.Get(), value), "ID3D11DeviceContext4::Signal");
  // 不 Flush 时命令可能停留在驱动队列中，CUDA 会一直等待
  context->Flush();
}

void SharedFence::D3DWait(ID3D11DeviceContext* context, uint64_t value) {
  Microsoft::WRL::ComPtr<ID3D11DeviceContext4> context4;
  Check(context->QueryInterface(IID_PPV_ARGS(&context4)), "ID3D11DeviceContext4");
  Check(context4->Wait(fence_.Get(), value), "ID3D11DeviceContext4::Wait");
}

void SharedFence::CudaWait(cudaStream_t stream, uint64_t value) {
  cudaExternalSemaphoreWaitParams params{};
  params.params.fence.value = value;
  CudaCheck(cudaWaitExternalSemaphoresAsync(&semaphore_, &params, 1, stream), "cudaWaitExternalSemaphoresAsync");
}

void SharedFence::CudaSignal(cudaStream_t stream, uint64_t value) {
  cudaExternalSemaphoreSignalParams params{};
  params.params.fence.value = value;
  CudaCheck(cudaSignalExternalSemaphoresAsync(&semaphore_, &params, 1, stream),
            "cudaSignalExternalSemaphoresAsync");
}

}  // namespace nr
