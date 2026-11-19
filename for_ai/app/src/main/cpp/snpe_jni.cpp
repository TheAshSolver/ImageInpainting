#include <jni.h>
#include <android/log.h>

#include <dlfcn.h>
#include <cstdlib>
#include <ctime>

#define TAG "SNPE_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

typedef void* Handle_t;

typedef Handle_t (*DlContainer_Open_fn)(const char*);
typedef void (*DlContainer_Delete_fn)(Handle_t);
typedef Handle_t (*RuntimeList_Create_fn)();
typedef int (*RuntimeList_StringToRuntime_fn)(const char*);
typedef int (*RuntimeList_Add_fn)(Handle_t, int);
typedef void (*RuntimeList_Delete_fn)(Handle_t);
typedef Handle_t (*SNPEBuilder_Create_fn)(Handle_t);
typedef Handle_t (*SNPEBuilder_SetRuntimeProcessorOrder_fn)(Handle_t, Handle_t);
typedef Handle_t (*SNPEBuilder_SetPerformanceProfile_fn)(Handle_t, int);
typedef Handle_t (*SNPEBuilder_SetProfilingLevel_fn)(Handle_t, int);
typedef Handle_t (*SNPEBuilder_Build_fn)(Handle_t);
typedef void (*SNPEBuilder_Delete_fn)(Handle_t);
typedef const char* (*Util_GetLastError_fn)();

extern "C" JNIEXPORT jboolean JNICALL
Java_com_qualcomm_qidk_inpaint_engine_SnpeNativeBridge_nativeInitSNPE(
    JNIEnv* env,
    jobject /* this */,
    jstring dlcPathStr,
    jstring runtimeStr
) {
    LOGI("--> nativeInitSNPE called");
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp/lama/dsp/lib;/data/local/tmp/lama/dsp;/dsp", 1);
    setenv("LD_LIBRARY_PATH", "/data/local/tmp/lama/lib:/data/local/tmp/lama", 1);

    const char* preloads[] = {
        "/data/local/tmp/lama/lib/libcdsprpc.so",
        "/data/local/tmp/lama/lib/libQnnSystem.so",
        "/data/local/tmp/lama/lib/libQairtSystem.so",
        "/data/local/tmp/lama/lib/libQnnHtp.so",
        "/data/local/tmp/lama/lib/libQnnHtpV79Stub.so",
        "/data/local/tmp/lama/lib/libSnpeHtpV79Stub.so",
        "/data/local/tmp/lama/lib/libhta_hexagon_runtime_snpe.so",
        "/data/local/tmp/lama/lib/libSNPE.so"
    };
    for (const char* p : preloads) {
        void* h = dlopen(p, RTLD_NOW | RTLD_GLOBAL);
        if (!h) {
            LOGE("Preload %s failed: %s", p, dlerror());
        } else {
            LOGI("Preloaded %s -> %p", p, h);
        }
    }

    const char* libPath = "/data/local/tmp/lama/lib/libSNPE.so";
    void* snpeHandle = dlopen(libPath, RTLD_NOW | RTLD_GLOBAL);
    if (!snpeHandle) {
        LOGE("dlopen failed on %s: %s", libPath, dlerror());
        return JNI_FALSE;
    }
    LOGI("libSNPE.so handle: %p", snpeHandle);

    auto DlContainer_Open = (DlContainer_Open_fn)dlsym(snpeHandle, "Snpe_DlContainer_Open");
    auto DlContainer_Delete = (DlContainer_Delete_fn)dlsym(snpeHandle, "Snpe_DlContainer_Delete");
    auto RuntimeList_Create = (RuntimeList_Create_fn)dlsym(snpeHandle, "Snpe_RuntimeList_Create");
    auto RuntimeList_StringToRuntime = (RuntimeList_StringToRuntime_fn)dlsym(snpeHandle, "Snpe_RuntimeList_StringToRuntime");
    auto RuntimeList_Add = (RuntimeList_Add_fn)dlsym(snpeHandle, "Snpe_RuntimeList_Add");
    auto RuntimeList_Delete = (RuntimeList_Delete_fn)dlsym(snpeHandle, "Snpe_RuntimeList_Delete");
    auto SNPEBuilder_Create = (SNPEBuilder_Create_fn)dlsym(snpeHandle, "Snpe_SNPEBuilder_Create");
    auto SNPEBuilder_SetRuntimeProcessorOrder = (SNPEBuilder_SetRuntimeProcessorOrder_fn)dlsym(snpeHandle, "Snpe_SNPEBuilder_SetRuntimeProcessorOrder");
    auto SNPEBuilder_SetPerformanceProfile = (SNPEBuilder_SetPerformanceProfile_fn)dlsym(snpeHandle, "Snpe_SNPEBuilder_SetPerformanceProfile");
    auto SNPEBuilder_SetProfilingLevel = (SNPEBuilder_SetProfilingLevel_fn)dlsym(snpeHandle, "Snpe_SNPEBuilder_SetProfilingLevel");
    auto SNPEBuilder_Build = (SNPEBuilder_Build_fn)dlsym(snpeHandle, "Snpe_SNPEBuilder_Build");
    auto SNPEBuilder_Delete = (SNPEBuilder_Delete_fn)dlsym(snpeHandle, "Snpe_SNPEBuilder_Delete");
    auto Util_GetLastError = (Util_GetLastError_fn)dlsym(snpeHandle, "Snpe_Util_GetLastError");
    auto ErrorCode_GetLastErrorString = (Util_GetLastError_fn)dlsym(snpeHandle, "Snpe_ErrorCode_GetLastErrorString");
    auto ErrorCode_getLastInfoString = (Util_GetLastError_fn)dlsym(snpeHandle, "Snpe_ErrorCode_getLastInfoString");

    if (!DlContainer_Open || !SNPEBuilder_Create || !SNPEBuilder_Build) {
        LOGE("Failed to dlsym essential SNPE functions: %s", dlerror());
        return JNI_FALSE;
    }

    const char* dlcPath = env->GetStringUTFChars(dlcPathStr, nullptr);
    LOGI("Opening DLC: %s", dlcPath);
    Handle_t container = DlContainer_Open(dlcPath);
    if (!container) {
        LOGE("Failed to open DLC container: %s. Last SNPE error: %s", dlcPath, ErrorCode_GetLastErrorString ? ErrorCode_GetLastErrorString() : "unknown");
        env->ReleaseStringUTFChars(dlcPathStr, dlcPath);
        return JNI_FALSE;
    }
    LOGI("DLC Container opened successfully: %p", container);

    Handle_t builder = SNPEBuilder_Create(container);
    if (!builder) {
        LOGE("Failed to create SNPEBuilder. Last error: %s", ErrorCode_GetLastErrorString ? ErrorCode_GetLastErrorString() : "unknown");
        DlContainer_Delete(container);
        env->ReleaseStringUTFChars(dlcPathStr, dlcPath);
        return JNI_FALSE;
    }

    if (RuntimeList_Create && RuntimeList_StringToRuntime && RuntimeList_Add && SNPEBuilder_SetRuntimeProcessorOrder) {
        Handle_t rList = RuntimeList_Create();
        int dspRuntime = RuntimeList_StringToRuntime("dsp"); // Lowercase "dsp" -> enum 2
        LOGI("DSP Runtime enum value from 'dsp': %d", dspRuntime);
        if (dspRuntime < 0) dspRuntime = 2; // dsp_fixed8_tf fallback
        RuntimeList_Add(rList, dspRuntime);
        SNPEBuilder_SetRuntimeProcessorOrder(builder, rList);
        RuntimeList_Delete(rList);
    }

    typedef Handle_t (*PlatformConfig_Create_fn)();
    typedef void (*PlatformConfig_Delete_fn)(Handle_t);
    typedef Handle_t (*SNPEBuilder_SetPlatformConfig_fn)(Handle_t, Handle_t);

    auto PlatformConfig_Create = (PlatformConfig_Create_fn)dlsym(snpeHandle, "Snpe_PlatformConfig_Create");
    auto PlatformConfig_Delete = (PlatformConfig_Delete_fn)dlsym(snpeHandle, "Snpe_PlatformConfig_Delete");
    auto SNPEBuilder_SetPlatformConfig = (SNPEBuilder_SetPlatformConfig_fn)dlsym(snpeHandle, "Snpe_SNPEBuilder_SetPlatformConfig");

    Handle_t pConfig = PlatformConfig_Create ? PlatformConfig_Create() : nullptr;
    if (pConfig && SNPEBuilder_SetPlatformConfig) {
        LOGI("Created PlatformConfig: %p, applying to builder", pConfig);
        SNPEBuilder_SetPlatformConfig(builder, pConfig);
    }

    if (SNPEBuilder_SetPerformanceProfile) {
        SNPEBuilder_SetPerformanceProfile(builder, 6); // BURST
    }
    if (SNPEBuilder_SetProfilingLevel) {
        SNPEBuilder_SetProfilingLevel(builder, 0); // OFF
    }

    LOGI("Building SNPE network instance on Hexagon DSP...");
    Handle_t snpe = SNPEBuilder_Build(builder);
    if (!snpe) {
        const char* errStr = ErrorCode_GetLastErrorString ? ErrorCode_GetLastErrorString() : "unknown";
        const char* infoStr = ErrorCode_getLastInfoString ? ErrorCode_getLastInfoString() : "unknown";
        LOGE("SNPEBuilder_Build FAILED! Last error: %s | Info: %s", errStr, infoStr);
        if (pConfig && PlatformConfig_Delete) PlatformConfig_Delete(pConfig);
        SNPEBuilder_Delete(builder);
        DlContainer_Delete(container);
        env->ReleaseStringUTFChars(dlcPathStr, dlcPath);
        return JNI_FALSE;
    }

    LOGI("🎉🎉🎉 SUCCESS! In-Process SNPE Hexagon HTP Graph BUILT and RESIDENT: %p!", snpe);
    if (pConfig && PlatformConfig_Delete) PlatformConfig_Delete(pConfig);
    SNPEBuilder_Delete(builder);
    DlContainer_Delete(container);
    env->ReleaseStringUTFChars(dlcPathStr, dlcPath);
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
