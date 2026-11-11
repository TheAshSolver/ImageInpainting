// sr_jni.cpp - JNI entry points for qidk.fix.SrNative (package fixed on purpose: the symbol names depend on it).
#include <jni.h>
#include <android/bitmap.h>

#include <cstdint>
#include <exception>
#include <vector>

#include "sr_core.h"

namespace {

// RAII lock of an ARGB_8888 bitmap (R,G,B,A byte order in memory).
struct LockedBitmap {
    JNIEnv* env;
    jobject bmp;
    void* pixels = nullptr;
    AndroidBitmapInfo info{};
    bool ok = false;
    LockedBitmap(JNIEnv* e, jobject b) : env(e), bmp(b) {
        if (!b) return;
        if (AndroidBitmap_getInfo(e, b, &info) != ANDROID_BITMAP_RESULT_SUCCESS) return;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return;
        if (AndroidBitmap_lockPixels(e, b, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) { pixels = nullptr; return; }
        ok = pixels != nullptr;
    }
    ~LockedBitmap() { if (ok) AndroidBitmap_unlockPixels(env, bmp); }
    uint8_t* data() const { return static_cast<uint8_t*>(pixels); }
};

struct FloatsRO {
    JNIEnv* env; jfloatArray arr; jfloat* p;
    FloatsRO(JNIEnv* e, jfloatArray a) : env(e), arr(a), p(a ? e->GetFloatArrayElements(a, nullptr) : nullptr) {}
    ~FloatsRO() { if (p) env->ReleaseFloatArrayElements(arr, p, JNI_ABORT); }
};
struct FloatsRW {
    JNIEnv* env; jfloatArray arr; jfloat* p;
    FloatsRW(JNIEnv* e, jfloatArray a) : env(e), arr(a), p(a ? e->GetFloatArrayElements(a, nullptr) : nullptr) {}
    ~FloatsRW() { if (p) env->ReleaseFloatArrayElements(arr, p, 0); }
};

bool lenAtLeast(JNIEnv* env, jfloatArray a, jlong n) { return a && env->GetArrayLength(a) >= n; }

}  // namespace

extern "C" {

// out = 2x cubic base of the bitmap for the window [x0, x0+win) x [y0, y0+win) of the 2x grid (win*win*3 floats).
JNIEXPORT jboolean JNICALL Java_qidk_fix_SrNative_sampleBase(JNIEnv* env, jclass, jobject src, jint x0, jint y0,
                                                              jint win, jfloatArray out) {
    if (win <= 0 || (win & 1) || (x0 & 1) || (y0 & 1)) return JNI_FALSE;
    if (!lenAtLeast(env, out, static_cast<jlong>(win) * win * 3)) return JNI_FALSE;
    try {
        LockedBitmap b(env, src);
        if (!b.ok) return JNI_FALSE;
        FloatsRW o(env, out);
        if (!o.p) return JNI_FALSE;
        qsr::sampleBase2x(b.data(), static_cast<int>(b.info.width), static_cast<int>(b.info.height),
                          static_cast<int>(b.info.stride), x0, y0, win, o.p);
        return JNI_TRUE;
    } catch (const std::exception&) {
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL Java_qidk_fix_SrNative_holeMask(JNIEnv* env, jclass, jint x0, jint y0, jint win, jint cell,
                                                            jint phase, jfloatArray out) {
    if (win <= 0 || cell <= 0 || (phase != 0 && phase != 1)) return JNI_FALSE;
    if (!lenAtLeast(env, out, static_cast<jlong>(win) * win)) return JNI_FALSE;
    FloatsRW o(env, out);
    if (!o.p) return JNI_FALSE;
    qsr::holeMask(x0, y0, win, cell, phase, o.p);
    return JNI_TRUE;
}

// o0 / o1 may be null (that phase failed or was rejected); both null writes the classical base. Returns 14 doubles (order: sumAbsD, sumD2, sumNyq2, sumHad2, sumDc2, sumAbsOU, sumOU, samples, pureHole, identityHole,
// clamped, nonFinite, phasesUsed, phasesRejected) or null on bad arguments.
JNIEXPORT jdoubleArray JNICALL Java_qidk_fix_SrNative_compose(
        JNIEnv* env, jclass, jfloatArray U, jfloatArray o0, jfloatArray o1, jint win, jint winX0, jint winY0, jint cell,
        jint margin, jfloat alpha, jfloat tau, jfloat driftLimit, jint cx0, jint cy0, jint cw, jint ch, jobject dst,
        jint dstX, jint dstY) {
    const jlong n = static_cast<jlong>(win) * win * 3;
    if (win <= 0 || (win & 1) || cell <= 0 || cw <= 0 || ch <= 0 || cx0 < 0 || cy0 < 0 || cx0 + cw > win || cy0 + ch > win) return nullptr;
    if (!lenAtLeast(env, U, n) || (o0 && !lenAtLeast(env, o0, n)) || (o1 && !lenAtLeast(env, o1, n))) return nullptr;
    try {
        LockedBitmap d(env, dst);
        if (!d.ok) return nullptr;
        FloatsRO u(env, U), a(env, o0), b(env, o1);
        if (!u.p || (o0 && !a.p) || (o1 && !b.p)) return nullptr;
        const qsr::ComposeStats st = qsr::composeCore(u.p, o0 ? a.p : nullptr, o1 ? b.p : nullptr, win, winX0, winY0, cell, margin, alpha,
                                                      tau, driftLimit, cx0, cy0, cw, ch, d.data(),
                                                      static_cast<int>(d.info.width), static_cast<int>(d.info.height),
                                                      static_cast<int>(d.info.stride), dstX, dstY);
        const jdouble v[14] = {st.sumAbsD, st.sumD2, st.sumNyq2, st.sumHad2, st.sumDc2, st.sumAbsOU, st.sumOU,
                               static_cast<jdouble>(st.samples), static_cast<jdouble>(st.pureHole),
                               static_cast<jdouble>(st.identityHole), static_cast<jdouble>(st.clamped),
                               static_cast<jdouble>(st.nonFinite), static_cast<jdouble>(st.phasesUsed),
                               static_cast<jdouble>(st.phasesRejected)};
        jdoubleArray r = env->NewDoubleArray(14);
        if (r) env->SetDoubleArrayRegion(r, 0, 14, v);
        return r;
    } catch (const std::exception&) {
        return nullptr;
    }
}

JNIEXPORT jboolean JNICALL Java_qidk_fix_SrNative_writeBase(JNIEnv* env, jclass, jfloatArray U, jint win, jint cx0,
                                                             jint cy0, jint cw, jint ch, jobject dst, jint dstX,
                                                             jint dstY) {
    if (win <= 0 || cw <= 0 || ch <= 0 || cx0 < 0 || cy0 < 0 || cx0 + cw > win || cy0 + ch > win) return JNI_FALSE;
    if (!lenAtLeast(env, U, static_cast<jlong>(win) * win * 3)) return JNI_FALSE;
    LockedBitmap d(env, dst);
    if (!d.ok) return JNI_FALSE;
    FloatsRO u(env, U);
    if (!u.p) return JNI_FALSE;
    qsr::writeBaseCore(u.p, win, cx0, cy0, cw, ch, d.data(), static_cast<int>(d.info.width),
                       static_cast<int>(d.info.height), static_cast<int>(d.info.stride), dstX, dstY);
    return JNI_TRUE;
}

JNIEXPORT jdouble JNICALL Java_qidk_fix_SrNative_lrPsnr(JNIEnv* env, jclass, jobject src, jobject dst, jint scale) {
    LockedBitmap s(env, src);
    LockedBitmap d(env, dst);
    if (!s.ok || !d.ok) return -1.0;
    return qsr::lrPsnr(s.data(), static_cast<int>(s.info.width), static_cast<int>(s.info.height),
                       static_cast<int>(s.info.stride), d.data(), static_cast<int>(d.info.width),
                       static_cast<int>(d.info.height), static_cast<int>(d.info.stride), scale);
}

}  // extern "C"
