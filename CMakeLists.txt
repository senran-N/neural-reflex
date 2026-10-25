cmake_minimum_required(VERSION 3.25)
project(neural_reflex LANGUAGES CXX)

if(NOT MSVC)
  message(FATAL_ERROR "当前仅支持 MSVC（Windows）构建")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4> $<$<COMPILE_LANGUAGE:CXX>:/permissive->
                    $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
                    $<$<COMPILE_LANGUAGE:CXX>:/MP>)
add_compile_definitions(UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)

# 推理部分：CUDA 预处理与后处理 + TensorRT for RTX
# VS 生成器需以 -T cuda=<CUDA 根目录> 指定工具链（见 CMakePresets.json）
option(NR_ENABLE_CUDA "构建 CUDA 与推理模块" ON)

add_library(nr_common STATIC
  src/common/win_util.cpp
  src/common/d3d_util.cpp
  src/common/frame_tag.cpp
  src/common/quad_renderer.cpp
  src/common/wic_image.cpp
)
target_include_directories(nr_common PUBLIC src)
target_link_libraries(nr_common PUBLIC d3d11 dxgi d3dcompiler windowscodecs)

add_library(nr_capture STATIC
  src/capture/wgc_capture.cpp
  src/capture/dd_capture.cpp
)
target_link_libraries(nr_capture PUBLIC nr_common windowsapp dwmapi)

add_library(nr_overlay STATIC
  src/overlay/overlay.cpp
)
target_link_libraries(nr_overlay PUBLIC nr_common dcomp)

add_executable(nr_probe apps/nr_probe/main.cpp)
target_link_libraries(nr_probe PRIVATE nr_capture nr_overlay)

add_executable(nr_testsource tools/nr_testsource/main.cpp)
target_link_libraries(nr_testsource PRIVATE nr_common)

if(NR_ENABLE_CUDA)
  enable_language(CUDA)
  # 自有内核覆盖 Turing 到 Blackwell，并附带 PTX 以便向前兼容
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 89-real 120)
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  find_package(CUDAToolkit REQUIRED)
  find_package(TensorRT-RTX REQUIRED CONFIG PATHS "$ENV{TENSORRT_RTX_ROOT}/cmake")

  add_library(nr_inference STATIC
    src/inference/cuda_check.cpp
    src/inference/d3d_cuda_interop.cpp
    src/inference/trt_engine.cpp
    src/inference/preprocess.cu
    src/inference/postprocess.cu
    src/inference/detector.cpp
  )
  target_compile_options(nr_inference PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  target_link_libraries(nr_inference PUBLIC nr_common CUDA::cudart_static TRTRTX::tensorrt
                                            TRTRTX::nvonnxparser)

  add_executable(nr_detect apps/nr_detect/main.cpp)
  target_link_libraries(nr_detect PRIVATE nr_inference nr_capture nr_overlay)
  # 运行时需要 TensorRT for RTX 的 DLL
  add_custom_command(TARGET nr_detect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TRTRTX::tensorrt>
            $<TARGET_FILE:TRTRTX::nvonnxparser> $<TARGET_FILE_DIR:nr_detect>)
endif()
