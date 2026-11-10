// classical_jni.cpp - JNI entry points for qidk.fix.ClassicalNative (package fixed on purpose, see ClassicalNative.kt).
#include <jni.h>

#include <cstdint>
#include <exception>

#include "classical.h"

namespace {

struct Pinned {
    JNIEnv* env;
    jbyteArray arr;
    jbyte* ptr;
    Pinned(JNIEnv* e, jbyteArray a) : env(e), arr(a), ptr(e->GetByteArrayElements(a, nullptr)) {}
    ~Pinned() { if (ptr) env->ReleaseByteArrayElements(arr, ptr, 0); }  // mode 0: copy back (the kernels write in place)
    uint8_t* data() const { return reinterpret_cast<uint8_t*>(ptr); }
};

bool sizesOk(JNIEnv* env, jbyteArray rgba, jbyteArray mask, jint w, jint h) {
    if (!rgba || !mask || w <= 0 || h <= 0) return false;
    return static_cast<jlong>(env->GetArrayLength(rgba)) == static_cast<jlong>(w) * h * 4 &&
           static_cast<jlong>(env->GetArrayLength(mask)) == static_cast<jlong>(w) * h;
}

}  // namespace

extern "C" {

// rgba (w*h*4) and mask (w*h, non-zero = hole) are modified in place. Returns the number of unfilled hole pixels
// (mask holds them), or a negative error code (see classical.h).
JNIEXPORT jint JNICALL Java_qidk_fix_ClassicalNative_criminisi(
        JNIEnv* env, jclass, jbyteArray rgba, jbyteArray mask, jint w, jint h,
        jint patchRadius, jint searchRadius, jint threads, jint budgetMs) {
    if (!sizesOk(env, rgba, mask, w, h)) return -1;
    try {
        Pinned p(env, rgba);
        Pinned m(env, mask);
        if (!p.ptr || !m.ptr) return -3;
        return qidk::criminisi_inpaint(p.data(), m.data(), w, h, patchRadius, searchRadius, threads, budgetMs);
    } catch (const std::exception&) {
        return -3;
    }
}

// rgba is modified in place, mask is read only. Returns 0 on success or a negative error code (see classical.h).
JNIEXPORT jint JNICALL Java_qidk_fix_ClassicalNative_biharmonic(
        JNIEnv* env, jclass, jbyteArray rgba, jbyteArray mask, jint w, jint h, jlong maxCells) {
    if (!sizesOk(env, rgba, mask, w, h)) return -1;
    try {
        Pinned p(env, rgba);
        Pinned m(env, mask);
        if (!p.ptr || !m.ptr) return -3;
        return qidk::biharmonic_inpaint(p.data(), m.data(), w, h, static_cast<int64_t>(maxCells));
    } catch (const std::exception&) {
        return -3;
    }
}

// Compatibility stub for prebuilt libopencv_java4.so missing symbol on Android 15
JNIEXPORT void __sfp_handle_exceptions() {}

}  // extern "C"
