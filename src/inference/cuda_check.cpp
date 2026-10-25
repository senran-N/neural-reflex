#include "inference/cuda_check.h"

#include <stdexcept>
#include <string>

namespace nr {

void CudaCheck(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " 失败：" + cudaGetErrorName(err) + " " +
                             cudaGetErrorString(err));
  }
}

}  // namespace nr
