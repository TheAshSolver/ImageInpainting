cmake_minimum_required(VERSION 3.22.1)
project("snpe_inpaint_jni")

add_library(snpe_inpaint_jni SHARED snpe_jni.cpp)

find_library(log-lib log)

target_link_libraries(snpe_inpaint_jni ${log-lib})

# --- classical inpainting kernels (Criminisi, biharmonic): pure C++17, no OpenCV, no Android headers ---------------------
add_library(qidk_classical SHARED
        classical/criminisi.cpp
        classical/biharmonic.cpp
        classical/classical_jni.cpp)
target_compile_features(qidk_classical PRIVATE cxx_std_17)
target_compile_options(qidk_classical PRIVATE -O3 -Wall -Wextra)
target_link_options(qidk_classical PRIVATE -Wl,-z,max-page-size=16384)
