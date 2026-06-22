cmake_minimum_required(VERSION 3.16)
project(unitree_go2_onnx_deploy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# ONNX Runtime — adjust path for x64 vs aarch64 (see deploy/thirdparty/README.md).
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.23.2" CACHE PATH "ONNX Runtime root")
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. See deploy/thirdparty/README.md")
endif()

find_library(UNITREE_SDK2 unitree_sdk2 REQUIRED)

include_directories(
    /usr/include/eigen3
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

set(GO2_DEPLOY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/deploy_params.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/onnx_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/go2_observation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/go2_lowlevel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/go2_controller.cpp
)

add_library(go2_deploy_lib ${GO2_DEPLOY_SOURCES})
target_link_libraries(go2_deploy_lib
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
    yaml-cpp
)

add_executable(go2_onnx_ctrl main.cpp)
target_link_libraries(go2_onnx_ctrl
    go2_deploy_lib
    ${UNITREE_SDK2}
    ddsc
    ddscxx
    rt
    pthread
    Boost::program_options
    yaml-cpp
)

set_target_properties(go2_onnx_ctrl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

install(TARGETS go2_onnx_ctrl RUNTIME DESTINATION bin)
