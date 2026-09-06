cmake_minimum_required(VERSION 3.22.1)
project("snpe_inpaint_jni")

add_library(snpe_inpaint_jni SHARED snpe_jni.cpp)

find_library(log-lib log)

target_link_libraries(snpe_inpaint_jni ${log-lib})
