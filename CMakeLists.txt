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

add_executable(nr_gpuload tools/nr_gpuload/main.cpp)
target_link_libraries(nr_gpuload PRIVATE nr_common)

# 结果共享与外部进程插件（扩展机制方案第 5、7 节），不依赖 CUDA
add_library(nr_results STATIC src/results/result_publisher.cpp)
target_link_libraries(nr_results PUBLIC nr_common)

add_library(nr_plugins STATIC src/plugins/plugin_host.cpp)
target_include_directories(nr_plugins PUBLIC third_party)
target_link_libraries(nr_plugins PUBLIC nr_results psapi)

# 插件开发与测试工具（7.7）：合成或回放结果发布；命令行托管单个插件
add_executable(nr_publish tools/nr_publish/main.cpp)
target_include_directories(nr_publish PRIVATE third_party)
target_link_libraries(nr_publish PRIVATE nr_results winmm)

add_executable(nr_plugin_run tools/nr_plugin_run/main.cpp)
target_link_libraries(nr_plugin_run PRIVATE nr_plugins)

# 示例插件 udp-forward（C++ SDK，7.8）：只依赖 SDK 头文件与 nlohmann/json，不链接宿主的任何库。
# 输出到插件目录的 bin，开发时宿主可直接从源码树启动
add_executable(udp-forward plugins/udp-forward/src/main.cpp)
target_include_directories(udp-forward PRIVATE sdk/cpp third_party)
target_link_libraries(udp-forward PRIVATE ws2_32)
set_target_properties(udp-forward PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_SOURCE_DIR}/plugins/udp-forward/bin>)

# 协议测试用的 C++ 插件（tests/plugins），与 Python 测试插件行为相同
add_executable(fx-consumer-cpp tests/plugins/fixtures/fx-consumer-cpp/src/main.cpp)
target_include_directories(fx-consumer-cpp PRIVATE sdk/cpp third_party)
set_target_properties(fx-consumer-cpp PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_SOURCE_DIR}/tests/plugins/fixtures/fx-consumer-cpp/bin>)

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
    src/inference/model_manifest.cpp
    src/inference/decoder.cpp
    src/inference/detector.cpp
  )
  target_include_directories(nr_inference PRIVATE third_party)
  target_compile_options(nr_inference PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  target_link_libraries(nr_inference PUBLIC nr_common CUDA::cudart_static TRTRTX::tensorrt
                                            TRTRTX::nvonnxparser)

  add_library(nr_pipeline STATIC src/pipeline/session.cpp)
  target_include_directories(nr_pipeline PRIVATE third_party)
  target_link_libraries(nr_pipeline PUBLIC nr_inference nr_capture nr_overlay nr_results avrt)

  add_executable(nr_detect apps/nr_detect/main.cpp)
  target_link_libraries(nr_detect PRIVATE nr_pipeline)
  # 运行时需要 TensorRT for RTX 的 DLL
  add_custom_command(TARGET nr_detect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TRTRTX::tensorrt>
            $<TARGET_FILE:TRTRTX::nvonnxparser> $<TARGET_FILE_DIR:nr_detect>)

  # 桌面程序：WebView2 界面（加载器静态链接，运行时由系统提供）
  set(NR_WEBVIEW2_ROOT "$ENV{WEBVIEW2_ROOT}" CACHE PATH "WebView2 SDK（NuGet 包解压目录）")
  if(NOT EXISTS "${NR_WEBVIEW2_ROOT}/build/native/include/WebView2.h")
    message(FATAL_ERROR "未找到 WebView2 SDK，请设置 WEBVIEW2_ROOT 或 NR_WEBVIEW2_ROOT")
  endif()
  add_executable(NeuralReflex WIN32
    apps/neural_reflex/main.cpp
    apps/neural_reflex/controller.cpp
    apps/neural_reflex/targets.cpp
    apps/neural_reflex/app.rc
    apps/neural_reflex/app.manifest
  )
  target_include_directories(NeuralReflex PRIVATE "${NR_WEBVIEW2_ROOT}/build/native/include" third_party)
  target_link_libraries(NeuralReflex PRIVATE nr_pipeline nr_plugins
                        "${NR_WEBVIEW2_ROOT}/build/native/x64/WebView2LoaderStatic.lib" dwmapi shell32 version)
  add_custom_command(TARGET NeuralReflex POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:TRTRTX::tensorrt>
            $<TARGET_FILE:TRTRTX::nvonnxparser> $<TARGET_FILE_DIR:NeuralReflex>
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/ui $<TARGET_FILE_DIR:NeuralReflex>/ui)

  # 发布目录：cmake --install build --config Release --prefix dist/NeuralReflex
  install(TARGETS NeuralReflex RUNTIME DESTINATION .)
  install(FILES $<TARGET_FILE:TRTRTX::tensorrt> $<TARGET_FILE:TRTRTX::nvonnxparser> DESTINATION .)
  install(DIRECTORY ui/ DESTINATION ui)
  # 外部进程插件：示例插件与 Python SDK（扩展机制方案 5.5）
  install(DIRECTORY plugins/ DESTINATION plugins PATTERN "__pycache__" EXCLUDE PATTERN "src" EXCLUDE
          PATTERN "*.pdb" EXCLUDE)
  install(DIRECTORY sdk/python/ DESTINATION sdk/python PATTERN "__pycache__" EXCLUDE)
  install(DIRECTORY sdk/cpp/ DESTINATION sdk/cpp)
  install(DIRECTORY third_party/nlohmann DESTINATION sdk/cpp)
  install(FILES models/reference/yolo26n-640-nms-fp16.onnx models/reference/yolo26n-640-nms-fp32.json
          DESTINATION models OPTIONAL)
  install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION . OPTIONAL)
  install(DIRECTORY third_party/licenses/ DESTINATION third_party/licenses)
endif()
