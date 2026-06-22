cmake_minimum_required(VERSION 3.16)
project(unitree_go2_onnx_deploy CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# ONNX Runtime lives in deploy/thirdparty/ (two levels above this CMakeLists.txt).
set(DEPLOY_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ONNX_DEFAULT "${DEPLOY_THIRDPARTY_DIR}/onnxruntime-linux-aarch64-gpu-1.16.0")
else()
    set(_ONNX_DEFAULT "${DEPLOY_THIRDPARTY_DIR}/onnxruntime-linux-x64-1.23.2")
endif()

set(ONNXRUNTIME_ROOT "${_ONNX_DEFAULT}" CACHE PATH "ONNX Runtime root")

# Relative -DONNXRUNTIME_ROOT=... is resolved from deploy/robots/go2/, not from build/.
if(NOT IS_ABSOLUTE "${ONNXRUNTIME_ROOT}")
    get_filename_component(
        ONNXRUNTIME_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/${ONNXRUNTIME_ROOT}"
        ABSOLUTE
    )
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}\n"
        "Download and extract under: ${DEPLOY_THIRDPARTY_DIR}/\n"
        "See deploy/thirdparty/README.md\n"
        "Example (Orin NX): cmake ..   # auto-detects aarch64 path\n"
        "Or: cmake .. -DONNXRUNTIME_ROOT=${DEPLOY_THIRDPARTY_DIR}/onnxruntime-linux-aarch64-gpu-1.16.0"
    )
endif()

file(GLOB _ONNX_LIBS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
list(LENGTH _ONNX_LIBS _ONNX_LIBS_COUNT)
if(_ONNX_LIBS_COUNT EQUAL 0)
    message(FATAL_ERROR "No libonnxruntime.so found in ${ONNXRUNTIME_ROOT}/lib")
endif()
list(GET _ONNX_LIBS 0 ONNXRUNTIME_LIBRARY)
message(STATUS "Using ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# unitree_sdk2 IDL headers include CycloneDDS C++ (ddscxx) and iceoryx.
find_path(DDSCXX_INCLUDE_DIR dds/topic/TopicTraits.hpp
    PATHS
        /usr/local/include/ddscxx
        /usr/include/ddscxx
    NO_DEFAULT_PATH
)
if(NOT DDSCXX_INCLUDE_DIR)
    message(FATAL_ERROR
        "CycloneDDS C++ headers not found (dds/topic/TopicTraits.hpp).\n"
        "Install unitree_sdk2 with CycloneDDS, or add /usr/local/include/ddscxx.\n"
        "See: https://github.com/unitreerobotics/unitree_sdk2"
    )
endif()

find_path(ICEROX_INCLUDE_DIR iceoryx/v2.0.2/iceoryx_posh/internal/posh_error_reporting.hpp
    PATHS
        /usr/local/include
        /usr/include
    NO_DEFAULT_PATH
)
set(UNITREE_EXTRA_INCLUDES "${DDSCXX_INCLUDE_DIR}")
if(ICEROX_INCLUDE_DIR)
    list(APPEND UNITREE_EXTRA_INCLUDES "${ICEROX_INCLUDE_DIR}")
    message(STATUS "Found iceoryx includes: ${ICEROX_INCLUDE_DIR}")
endif()
message(STATUS "Found ddscxx includes: ${DDSCXX_INCLUDE_DIR}")

find_library(UNITREE_SDK2 unitree_sdk2 REQUIRED)
find_library(DDSC_LIB ddsc REQUIRED)
find_library(DDSCXX_LIB ddscxx REQUIRED)

include_directories(
    /usr/include/eigen3
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${UNITREE_EXTRA_INCLUDES}
)

link_directories(${ONNXRUNTIME_ROOT}/lib)

set(GO2_DEPLOY_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/deploy_params.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/onnx_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/go2_observation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/go2_lowlevel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/go2_controller.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/go2_sport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/scenario_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/fault_scenario.cpp
)

add_library(go2_deploy_lib ${GO2_DEPLOY_SOURCES})
target_include_directories(go2_deploy_lib PUBLIC ${UNITREE_EXTRA_INCLUDES})
target_link_libraries(go2_deploy_lib
    ${ONNXRUNTIME_LIBRARY}
    ${UNITREE_SDK2}
    ${DDSC_LIB}
    ${DDSCXX_LIB}
    yaml-cpp
)

add_executable(go2_onnx_ctrl main.cpp)
target_link_libraries(go2_onnx_ctrl
    go2_deploy_lib
    ${UNITREE_SDK2}
    ${DDSC_LIB}
    ${DDSCXX_LIB}
    rt
    pthread
    Boost::program_options
    yaml-cpp
)

set_target_properties(go2_onnx_ctrl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

install(TARGETS go2_onnx_ctrl RUNTIME DESTINATION bin)
