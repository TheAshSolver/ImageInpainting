#!/usr/bin/env bash
set -e

NDK_PATH="/home/tarsh/Android/Sdk/ndk/26.2.11394342"
CLANG="${NDK_PATH}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_OUT="${SRC_DIR}/qidk_load_generator"

echo "Building QIDK Native Load Generator with Android NDK 26.2..."
echo "Source: ${SRC_DIR}/src/main.cpp"
echo "Output: ${BIN_OUT}"

"${CLANG}" -O3 -pthread -std=c++17 \
    -static-libstdc++ \
    "${SRC_DIR}/src/main.cpp" \
    -o "${BIN_OUT}"

echo "Build complete: $(ls -lh "${BIN_OUT}")"
file "${BIN_OUT}"
