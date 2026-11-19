cmake_minimum_required(VERSION 3.18)
project(qidk_stable_diffusion CXX)

set(CMAKE_CXX_STANDARD 17)

# Set path to Qualcomm AI Engine Direct SDK
set(QNN_SDK_ROOT $ENV{QAIRT_ROOT})

if(NOT DEFINED QNN_SDK_ROOT OR "${QNN_SDK_ROOT}" STREQUAL "")
    if(DEFINED ENV{QNN_SDK_ROOT})
        set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
    else()
        message(FATAL_ERROR "QAIRT_ROOT or QNN_SDK_ROOT is not set.")
    endif()
endif()

include_directories(
    ${QNN_SDK_ROOT}/include/QNN
    ${QNN_SDK_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
    ${QNN_SDK_ROOT}/lib/aarch64-android
)

add_executable(sd_qidk_runner sd_runner.cpp)
add_executable(sd_qidk_runner_encoder sd_runner_encoder.cpp)
add_executable(sd_qidk_runner_inpaint sd_runner_inpaint.cpp)
add_executable(sd_qidk_runner_dual_zerocopy sd_dual.cpp)

# C++20 and ARMv8.2-A FP16/DotProd flags required by sd_dual.cpp
target_compile_features(sd_qidk_runner_dual_zerocopy PRIVATE cxx_std_20)
target_compile_options(sd_qidk_runner_dual_zerocopy PRIVATE -O3 -march=armv8.2-a+fp16+dotprod)

# Link against Qualcomm HTP and QNN System libraries
target_link_libraries(sd_qidk_runner
    QnnHtp
    QnnSystem
    dl
    log
    android
)

target_link_libraries(sd_qidk_runner_encoder
    QnnHtp
    QnnSystem
    dl
    log
    android
)

target_link_libraries(sd_qidk_runner_inpaint
    QnnHtp
    QnnSystem
    dl
    log
    android
)

target_link_libraries(sd_qidk_runner_dual_zerocopy
    dl
    log
    android
)