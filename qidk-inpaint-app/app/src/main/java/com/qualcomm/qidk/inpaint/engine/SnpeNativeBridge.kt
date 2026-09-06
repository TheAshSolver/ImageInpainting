package com.qualcomm.qidk.inpaint.engine

/**
 * Option A Scaffolding: Native C++ JNI bridge loading libSNPE.so / libQnnHtp.so directly.
 */
object SnpeNativeBridge {
    init {
        try {
            System.loadLibrary("snpe_inpaint_jni")
        } catch (e: UnsatisfiedLinkError) {
            // Handled gracefully if native library is compiled separately
        }
    }

    external fun nativeInitSNPE(dlcPath: String, runtime: String): Boolean
    external fun nativeExecuteInference(imageBytes: ByteArray, maskBytes: ByteArray, outputBytes: ByteArray): Boolean
    external fun nativeRelease(): Unit
}
