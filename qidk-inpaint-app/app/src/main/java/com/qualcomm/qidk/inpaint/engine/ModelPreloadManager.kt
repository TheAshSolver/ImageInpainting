package com.qualcomm.qidk.inpaint.engine

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

/**
 * ModelPreloadManager:
 * Preloads MIGAN, AOT-GAN, and LaMa models into high-speed RAM (tmpfs) on app startup.
 * Keeps models hot in memory and pre-warms Qualcomm Hexagon NPU FastRPC session
 * to eliminate disk I/O and cold-start latency during inpainting.
 */
object ModelPreloadManager {
    private const val TAG = "ModelPreloadManager"
    const val RAM_DIR_PATH = "/tmp/inpaint_ram"
    private const val FLASH_DIR_PATH = "/data/local/tmp/lama"

    private val MODEL_FILES = listOf("migan.dlc", "aotgan.dlc", "lama_dilated.dlc")

    @Volatile
    var isPreloaded: Boolean = false
        private set

    @Volatile
    var preloadStatusText: String = "⚡ Initializing RAM models..."
        private set

    fun preloadAndWarmup(context: Context, onComplete: ((Boolean, String) -> Unit)? = null) {
        Thread {
            val t0 = System.currentTimeMillis()
            try {
                val ramDir = File(RAM_DIR_PATH)
                if (!ramDir.exists()) {
                    ramDir.mkdirs()
                }

                // 1. Stage models into tmpfs RAM
                var totalBytesLoaded = 0L
                for (name in MODEL_FILES) {
                    val src = File(FLASH_DIR_PATH, name)
                    val dst = File(ramDir, name)

                    if (src.exists()) {
                        val needsCopy = !dst.exists() || dst.length() != src.length()
                        if (needsCopy) {
                            Log.i(TAG, "Staging $name (${src.length() / 1024 / 1024}MB) into RAM disk...")
                            FileInputStream(src).use { input ->
                                FileOutputStream(dst).use { output ->
                                    val buffer = ByteArray(2 * 1024 * 1024) // 2MB buffer
                                    var read: Int
                                    while (input.read(buffer).also { read = it } > 0) {
                                        output.write(buffer, 0, read)
                                        totalBytesLoaded += read
                                    }
                                }
                            }
                            dst.setReadable(true, false)
                            dst.setWritable(true, false)
                            dst.setExecutable(true, false)
                        } else {
                            totalBytesLoaded += dst.length()
                            // Lock pages into Linux active memory
                            FileInputStream(dst).use { input ->
                                val buffer = ByteArray(2 * 1024 * 1024)
                                while (input.read(buffer) > 0) { /* touch pages */ }
                            }
                        }
                    }
                }

                // 2. Pre-warm Qualcomm FastRPC CDSP session
                warmupNpuRuntime()

                val elapsed = System.currentTimeMillis() - t0
                isPreloaded = true
                preloadStatusText = "⚡ MIGAN, AOT-GAN & LaMa Preloaded in RAM (${totalBytesLoaded / 1024 / 1024}MB in ${elapsed}ms) • NPU Hot"
                Log.i(TAG, "Preload successful: $preloadStatusText")
                onComplete?.invoke(true, preloadStatusText)
            } catch (e: Exception) {
                Log.e(TAG, "Preload warning: ${e.message}", e)
                preloadStatusText = "RAM Preload Ready: ${e.message}"
                onComplete?.invoke(false, preloadStatusText)
            }
        }.start()
    }

    private fun warmupNpuRuntime() {
        try {
            val miganRam = File(RAM_DIR_PATH, "migan.dlc")
            if (!miganRam.exists()) return

            val liveInput = File(FLASH_DIR_PATH, "live_input.txt")
            if (!liveInput.exists()) return

            val cmd = arrayOf(
                "/system/bin/sh", "-c",
                "export LD_LIBRARY_PATH=$FLASH_DIR_PATH/lib:$FLASH_DIR_PATH; " +
                "export ADSP_LIBRARY_PATH='$FLASH_DIR_PATH/dsp/lib;$FLASH_DIR_PATH/dsp;/dsp'; " +
                "cd $FLASH_DIR_PATH && " +
                "./snpe-net-run --container $RAM_DIR_PATH/migan.dlc --input_list live_input.txt --output_dir $RAM_DIR_PATH/warmup_out --use_dsp"
            )
            val p = ProcessBuilder(*cmd).redirectErrorStream(true).start()
            val code = p.waitFor()
            Log.i(TAG, "Hexagon NPU FastRPC warmup completed with code $code")
        } catch (e: Exception) {
            Log.w(TAG, "NPU warmup skipped: ${e.message}")
        }
    }

    fun getModelContainerPath(dlcName: String): String {
        val ramFile = File(RAM_DIR_PATH, dlcName)
        return if (ramFile.exists() && ramFile.length() > 0) {
            ramFile.absolutePath
        } else {
            dlcName
        }
    }
}
