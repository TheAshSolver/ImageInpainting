package com.qualcomm.qidk.inpaint.engine

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileInputStream

/**
 * ModelPreloadManager:
 * Preloads GAN models into high-speed memory and pre-warms the Qualcomm Hexagon NPU DSP session
 * synchronously during app startup, while streaming Stable Diffusion models in the background
 * asynchronously so heavy SD weights (~1.75 GB) do not block or slow down app launch time.
 */
object ModelPreloadManager {
    private const val TAG = "ModelPreloadManager"
    private const val FLASH_DIR_PATH = "/data/local/tmp/lama"
    private const val SD_DIR_PATH = "/data/local/tmp/sd_runtime"

    @Volatile var isGanPreloaded: Boolean = false
        private set

    @Volatile var isSdPreloaded: Boolean = false
        private set

    @Volatile var ganPreloadTimeMs: Long = 0L
        private set

    @Volatile var sdPreloadTimeMs: Long = 0L
        private set

    /**
     * Preloads GAN models into Linux page-cache and pre-warms the Hexagon HTP v79 DSP session.
     * Executed synchronously during app initialization.
     */
    fun preloadGanModelsSync(context: Context): Long {
        val t0 = System.currentTimeMillis()
        try {
            val ganFiles = listOf(
                "migan_htp_v79.dlc",
                "bench_cache_lama_dilated.dlc",
                "bench_cache_aotgan.dlc",
                "lama_dilated.dlc",
                "aotgan.dlc"
            )
            for (fName in ganFiles) {
                val f = File(FLASH_DIR_PATH, fName)
                if (f.exists() && f.canRead()) {
                    try {
                        FileInputStream(f).use { stream ->
                            val buf = ByteArray(1024 * 1024)
                            while (stream.read(buf) > 0) { /* cache pages */ }
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Page cache preload skipped for $fName: ${e.message}")
                    }
                }
            }

            warmupNpuGan()
            ensureResidentDaemonRunning()
            try {
                val initRes = SnpeNativeBridge.nativeInitSNPE("$FLASH_DIR_PATH/migan_htp_v79.dlc", "DSP")
                Log.i(TAG, "SnpeNativeBridge.nativeInitSNPE returned: $initRes")
            } catch (t: Throwable) {
                Log.w(TAG, "SnpeNativeBridge invocation: ${t.message}")
            }
            isGanPreloaded = true
        } catch (e: Exception) {
            Log.e(TAG, "Error in GAN preload: ${e.message}", e)
        }
        ganPreloadTimeMs = System.currentTimeMillis() - t0
        Log.i(TAG, "GAN models preloaded on NPU in ${ganPreloadTimeMs}ms")
        return ganPreloadTimeMs
    }

    /**
     * Preloads Stable Diffusion models asynchronously in a low-priority background daemon thread.
     * Streams UNet, VAE, Text Encoder, and CLIP Tokenizer into active memory.
     */
    fun preloadSdModelsAsync(onComplete: ((Long) -> Unit)? = null) {
        Thread {
            val t0 = System.currentTimeMillis()
            try {
                val sdFiles = listOf(
                    "models/unet_inpaint_16ch_ctx.bin",
                    "models/text_encoder.serialized.bin",
                    "models/vae_encoder_htp.serialized.bin",
                    "models/vae.serialized.bin",
                    "vocab.json",
                    "merges.txt"
                )
                for (relPath in sdFiles) {
                    val f = File(SD_DIR_PATH, relPath)
                    if (f.exists() && f.canRead()) {
                        try {
                            FileInputStream(f).use { stream ->
                                val buf = ByteArray(2 * 1024 * 1024)
                                while (stream.read(buf) > 0) { /* cache pages */ }
                            }
                        } catch (e: Exception) {
                            Log.w(TAG, "Failed reading SD file $relPath: ${e.message}")
                        }
                    }
                }
                isSdPreloaded = true
            } catch (e: Exception) {
                Log.e(TAG, "Error in SD background preload: ${e.message}", e)
            }
            sdPreloadTimeMs = System.currentTimeMillis() - t0
            Log.i(TAG, "SD models preloaded in background in ${sdPreloadTimeMs}ms")
            onComplete?.invoke(sdPreloadTimeMs)
        }.apply {
            isDaemon = true
            priority = Thread.MIN_PRIORITY
            start()
        }
    }

    private fun warmupNpuGan() {
        try {
            val flashDir = File(FLASH_DIR_PATH)
            val dlcName = if (File(flashDir, "migan_htp_v79.dlc").exists()) "migan_htp_v79.dlc" else "migan.dlc"
            val liveInput = if (File(flashDir, "live_input_v3.txt").exists()) {
                File(flashDir, "live_input_v3.txt")
            } else {
                File(flashDir, "live_input_v2.txt")
            }
            if (!liveInput.exists()) return

            val inputListName = liveInput.name
            val cmd = arrayOf(
                "/system/bin/sh", "-c",
                "export LD_LIBRARY_PATH=$FLASH_DIR_PATH/lib:$FLASH_DIR_PATH; " +
                "export ADSP_LIBRARY_PATH='$FLASH_DIR_PATH/dsp/lib;$FLASH_DIR_PATH/dsp;/dsp'; " +
                "cd $FLASH_DIR_PATH && " +
                "./snpe-net-run --container $dlcName --input_list $inputListName --output_dir app_live_output_v3 --use_dsp"
            )
            val p = ProcessBuilder(*cmd).redirectErrorStream(true).start()
            p.waitFor()
        } catch (e: Exception) {
            Log.w(TAG, "NPU warmup skipped: ${e.message}")
        }
    }

    fun ensureResidentDaemonRunning() {
        try {
            val daemonBin = File(FLASH_DIR_PATH, "snpe_resident_daemon")
            if (daemonBin.exists() && daemonBin.canExecute()) {
                val cmd = arrayOf(
                    "/system/bin/sh", "-c",
                    "export LD_LIBRARY_PATH=$FLASH_DIR_PATH/lib:$FLASH_DIR_PATH; " +
                    "export ADSP_LIBRARY_PATH='$FLASH_DIR_PATH/dsp/lib;$FLASH_DIR_PATH/dsp;/dsp'; " +
                    "pgrep -f snpe_resident_daemon >/dev/null || (nohup $FLASH_DIR_PATH/snpe_resident_daemon 55555 >/dev/null 2>&1 &)"
                )
                ProcessBuilder(*cmd).start()
                Log.i(TAG, "⚡ snpe_resident_daemon check/launch dispatched")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed launching snpe_resident_daemon: ${e.message}")
        }
    }
}
