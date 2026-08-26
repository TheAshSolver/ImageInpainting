cmake_minimum_required(VERSION 3.18)
project(qidk_stable_diffusion CXX)

set(CMAKE_CXX_STANDARD 17)

# Set path to Qualcomm AI Engine Direct SDK
set(QNN_SDK_ROOT $ENV{QAIRT_ROOT})

include_directories(
    ${QNN_SDK_ROOT}/include/QNN
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

link_directories(
    ${QNN_SDK_ROOT}/lib/aarch64-android
)

add_executable(sd_qidk_runner sd_runner.cpp)

# Link against Qualcomm HTP and QNN System libraries
target_link_libraries(sd_qidk_runner
    QnnHtp
    QnnSystem
    dl
    log
    android
)