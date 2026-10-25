cmake_minimum_required(VERSION 3.25)
project(neural_reflex LANGUAGES CXX)

if(NOT MSVC)
  message(FATAL_ERROR "当前仅支持 MSVC（Windows）构建")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_compile_options(/W4 /permissive- /utf-8 /EHsc /MP)
add_compile_definitions(UNICODE _UNICODE WIN32_LEAN_AND_MEAN NOMINMAX)

# 推理部分（CUDA / TensorRT for RTX）待环境就绪后接入
option(NR_ENABLE_CUDA "构建 CUDA 与推理模块" OFF)

add_library(nr_common STATIC
  src/common/win_util.cpp
  src/common/d3d_util.cpp
  src/common/frame_tag.cpp
  src/common/quad_renderer.cpp
)
target_include_directories(nr_common PUBLIC src)
target_link_libraries(nr_common PUBLIC d3d11 dxgi d3dcompiler)

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
