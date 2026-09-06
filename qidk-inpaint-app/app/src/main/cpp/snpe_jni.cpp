#include <jni.h>
#include <android/log.h>

#define TAG "SNPE_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qualcomm_qidk_inpaint_engine_SnpeNativeBridge_nativeInitSNPE(
    JNIEnv* env,
    jobject /* this */,
    jstring dlcPath,
    jstring runtime
) {
    LOGI("Scaffolded SNPE Native JNI Initialized");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qualcomm_qidk_inpaint_engine_SnpeNativeBridge_nativeExecuteInference(
    JNIEnv* env,
    jobject /* this */,
    jbyteArray imageBytes,
    jbyteArray maskBytes,
    jbyteArray outputBytes
) {
    LOGI("Scaffolded SNPE Native JNI Inference Invoked");
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_qualcomm_qidk_inpaint_engine_SnpeNativeBridge_nativeRelease(
    JNIEnv* env,
    jobject /* this */
) {
    LOGI("Scaffolded SNPE Native JNI Released");
}
