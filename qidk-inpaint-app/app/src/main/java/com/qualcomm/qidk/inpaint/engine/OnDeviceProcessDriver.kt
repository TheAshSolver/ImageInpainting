package com.qualcomm.qidk.inpaint.engine

import android.graphics.Bitmap
import android.graphics.Color
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class InferenceTelemetry(
    val modelName: String,
    val executionMode: String,
    val latencyMs: Long,
    val engineLatencyMs: Long = latencyMs,
    val stagingLatencyMs: Long = 0L,
    val energyJoules: Float?,
    val powerWatts: Float?,
    val thermalDeltaC: Float,
    val startTempC: Float,
    val endTempC: Float,
    val rawCurrentUa: Long?,
    val rawVoltageUv: Long?,
    val resultBitmap: Bitmap
)

object OnDeviceProcessDriver {

    private const val DEVICE_LAMA_DIR = "/data/local/tmp/lama"
    private const val DEVICE_SD_DIR = "/data/local/tmp/sd_runtime"

    // -------------------------------------------------------------------------
    // Eager Pre-caching System (V3 Ultra-Low Latency Optimization)
    // -------------------------------------------------------------------------
    @Volatile private var precachedImageRef: Bitmap? = null
    @Volatile private var precachedMaskRef: Bitmap? = null
    @Volatile private var precachedImageGenId: Int = -1
    @Volatile private var precachedMaskGenId: Int = -1
    @Volatile private var isImagePrecached: Boolean = false
    @Volatile private var isMaskPrecached: Boolean = false
    @Volatile private var precachedMaskIsInverted: Boolean = false
    private val precacheExecutor = java.util.concurrent.Executors.newFixedThreadPool(2)
    @Volatile private var imagePrecacheFuture: java.util.concurrent.Future<*>? = null
    @Volatile private var maskPrecacheFuture: java.util.concurrent.Future<*>? = null

    fun eagerPrecacheImage(image: Bitmap) {
        precachedImageRef = image
        precachedImageGenId = image.generationId
        isImagePrecached = false
        imagePrecacheFuture = precacheExecutor.submit {
            try {
                val inputDir = File(DEVICE_LAMA_DIR, "input_v3")
                if (!inputDir.exists()) inputDir.mkdirs()
                val rawImgFile = File(inputDir, "live_img_v3.raw")
                val imgBytes = bitmapToFloat32Raw(image)
                FileOutputStream(rawImgFile).use { it.write(imgBytes) }
                rawImgFile.setReadable(true, false)
                rawImgFile.setWritable(true, false)
                isImagePrecached = true
                android.util.Log.d("OnDeviceProcessDriver", "⚡ V3 Eager image pre-cache complete (3.15 MB, genId=$precachedImageGenId)")
            } catch (e: Exception) {
                android.util.Log.w("OnDeviceProcessDriver", "Eager image pre-cache warning: ${e.message}")
            }
        }
    }

    fun eagerPrecacheMask(mask: Bitmap, inverted: Boolean = false) {
        precachedMaskRef = mask
        precachedMaskGenId = mask.generationId
        precachedMaskIsInverted = inverted
        isMaskPrecached = false
        maskPrecacheFuture = precacheExecutor.submit {
            try {
                val inputDir = File(DEVICE_LAMA_DIR, "input_v3")
                if (!inputDir.exists()) inputDir.mkdirs()
                val rawMaskFile = File(inputDir, "live_mask_v3.raw")
                val listFile = File(DEVICE_LAMA_DIR, "live_input_v3.txt")

                val maskBytes = maskToFloat32Raw(mask, inverted = inverted)
                FileOutputStream(rawMaskFile).use { it.write(maskBytes) }
                rawMaskFile.setReadable(true, false)
                rawMaskFile.setWritable(true, false)

                listFile.writeText("image:=input_v3/live_img_v3.raw mask:=input_v3/live_mask_v3.raw\n")
                listFile.setReadable(true, false)
                listFile.setWritable(true, false)

                isMaskPrecached = true
                android.util.Log.d("OnDeviceProcessDriver", "⚡ V3 Eager mask pre-cache complete (1.05 MB, inv=$inverted, genId=$precachedMaskGenId)")
            } catch (e: Exception) {
                android.util.Log.w("OnDeviceProcessDriver", "Eager mask pre-cache warning: ${e.message}")
            }
        }
    }

    /**
     * Reads current temperature in Celsius from thermal_zone0 directly.
     * Returns null if unreadable. Zero hardcoded fallback.
     */
    fun readCurrentTemperatureC(): Float? {
        return try {
            val f = File("/sys/class/thermal/thermal_zone0/temp")
            if (f.exists() && f.canRead()) {
                val t = f.readText().trim().toFloatOrNull()
                if (t != null) t / 1000.0f else null
            } else null
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Reads peak temperature across all SoC thermal zones live.
     * Returns 0.0f if no thermal zone can be read.
     */
    fun readPeakTemperatureC(): Float {
        var peak = 0.0f
        try {
            val tzDir = File("/sys/class/thermal")
            if (tzDir.exists() && tzDir.isDirectory) {
                val tzFiles = tzDir.listFiles { f -> f.name.startsWith("thermal_zone") } ?: emptyArray()
                for (tz in tzFiles) {
                    val tempFile = File(tz, "temp")
                    if (tempFile.exists() && tempFile.canRead()) {
                        val tVal = tempFile.readText().trim().toFloatOrNull()
                        if (tVal != null && tVal > -40000 && tVal < 150000) {
                            val c = tVal / 1000.0f
                            if (c > peak) peak = c
                        }
                    }
                }
            }
        } catch (_: Exception) {}
        return if (peak > 0f) peak else (readCurrentTemperatureC() ?: 0.0f)
    }

    /**
     * Reads raw instantaneous active current from live hardware sysfs nodes.
     * Returns exactly what the sensor reports (microamperes, uA).
     * Returns null if unavailable. NO fake fallback values.
     */
    fun readActiveCurrentUa(): Long? {
        // 1. Try PMIC feedback current
        try {
            val pmic = File("/sys/bus/iio/devices/iio:device0/in_current_pmih010x_ichg_fb_input")
            if (pmic.exists() && pmic.canRead()) {
                val s = pmic.readText().trim()
                val v = s.toLongOrNull()
                if (v != null) return v
            }
        } catch (_: Exception) {}

        // 2. Try Battery current_now node
        try {
            val bat = File("/sys/class/power_supply/battery/current_now")
            if (bat.exists() && bat.canRead()) {
                val s = bat.readText().trim()
                val v = s.toLongOrNull()
                if (v != null) return v
            }
        } catch (_: Exception) {}

        // 3. Try PMIC input current
        try {
            val iin = File("/sys/bus/iio/devices/iio:device0/in_current_pmih010x_iin_input")
            if (iin.exists() && iin.canRead()) {
                val s = iin.readText().trim()
                val v = s.toLongOrNull()
                if (v != null) return v
            }
        } catch (_: Exception) {}

        return null
    }

    /**
     * Reads raw active voltage from live hardware sysfs nodes.
     * Returns exactly what the sensor reports (microvolts, uV).
     * Returns null if unavailable. NO fake fallback values.
     */
    fun readActiveVoltageUv(): Long? {
        // 1. Battery voltage_now
        try {
            val bat = File("/sys/class/power_supply/battery/voltage_now")
            if (bat.exists() && bat.canRead()) {
                val v = bat.readText().trim().toLongOrNull()
                if (v != null) return v
            }
        } catch (_: Exception) {}

        // 2. PMIC system bus voltage
        try {
            val vph = File("/sys/bus/iio/devices/iio:device0/in_voltage_pmih010x_vph_pwr_input")
            if (vph.exists() && vph.canRead()) {
                val v = vph.readText().trim().toLongOrNull()
                if (v != null) return v * 2
            }
        } catch (_: Exception) {}

        return null
    }

    /**
     * Samples instantaneous power in Watts directly from live hardware sensors.
     * Returns null if current or voltage sensors are unreadable.
     */
    fun sampleInstantaneousPowerWatts(): Float? {
        val currUa = readActiveCurrentUa() ?: return null
        val voltUv = readActiveVoltageUv() ?: return null
        val p = (kotlin.math.abs(currUa) * 1e-6) * (voltUv * 1e-6)
        return p.toFloat()
    }

    class HardwareTelemetryTracker {
        private val powerSamples = mutableListOf<Float>()
        private val tempSamples = mutableListOf<Float>()
        @Volatile private var isRunning = false
        private var worker: Thread? = null

        fun start() {
            powerSamples.clear()
            tempSamples.clear()
            isRunning = true
            worker = Thread {
                while (isRunning) {
                    val p = sampleInstantaneousPowerWatts()
                    val t = readPeakTemperatureC()
                    synchronized(this) {
                        if (p != null) powerSamples.add(p)
                        if (t > 0f) tempSamples.add(t)
                    }
                    try {
                        Thread.sleep(50)
                    } catch (_: InterruptedException) {
                        break
                    }
                }
            }.apply {
                isDaemon = true
                start()
            }
        }

        fun stop(latencyMs: Long, startTempC: Float, endTempC: Float): Triple<Float?, Float?, Float> {
            isRunning = false
            worker?.interrupt()
            try { worker?.join(200) } catch (_: Exception) {}

            val (avgPower, peakTemp) = synchronized(this) {
                val avgP = if (powerSamples.isNotEmpty()) {
                    powerSamples.average().toFloat()
                } else {
                    sampleInstantaneousPowerWatts()
                }
                val maxT = if (tempSamples.isNotEmpty()) {
                    tempSamples.maxOrNull() ?: endTempC
                } else endTempC
                Pair(avgP, maxT)
            }

            val energyJoules = if (avgPower != null) avgPower * (latencyMs / 1000.0f) else null
            val deltaT = peakTemp - startTempC

            return Triple(
                avgPower?.let { String.format(java.util.Locale.US, "%.3f", it).toFloat() },
                energyJoules?.let { String.format(java.util.Locale.US, "%.3f", it).toFloat() },
                String.format(java.util.Locale.US, "%.1f", deltaT).toFloat()
            )
        }
    }

    fun executeInference(
        image: Bitmap,
        mask: Bitmap,
        modelKey: String,
        positivePrompt: String = "",
        negativePrompt: String = ""
    ): InferenceTelemetry {
        val startTime = System.currentTimeMillis()
        val targetModel = modelKey.uppercase()
        val isInefficientSd = targetModel.contains("INEFFICIENT")
        val isSd = targetModel.contains("SD") || targetModel.contains("DIFFUSION") || isInefficientSd

        // =====================================================================
        // 1. Stable Diffusion Pipeline (DPM-Solver++ 12/20-step on Hexagon NPU)
        // =====================================================================
        if (isSd) {
            val runnerBin = if (isInefficientSd) "./sd_qidk_runner_inefficient" else "./sd_qidk_runner_inpaint"
            val runnerName = if (isInefficientSd) "Stable Diffusion 1.5 (Inefficient RePaint)" else "Stable Diffusion 1.5 (Inpainting - DPM 12-step)"
            val execMode = if (isInefficientSd) "Qualcomm Hexagon NPU (HTP v79 Live Euler)" else "Qualcomm Hexagon NPU (HTP v79 Live DPM++)"

            // Direct user prompt (default empty string '' allows pure structural restoration without hallucinating external concepts)
            val finalPosPrompt = positivePrompt.trim()
            val sanitizedPos = finalPosPrompt.replace("\"", "\\\"").replace("'", "\\'")

            return try {
                val sdDir = File(DEVICE_SD_DIR)
                val imgRaw = File(sdDir, "image.raw")
                val maskRaw = File(sdDir, "mask.raw")
                val outPng = File(sdDir, "sd_output.png")

                if (outPng.exists()) {
                    outPng.delete()
                }

                val imgBytes = bitmapToFloat32Raw(image)
                val maskBytes = maskToFloat32Raw(mask, inverted = false)

                FileOutputStream(imgRaw).use { it.write(imgBytes) }
                FileOutputStream(maskRaw).use { it.write(maskBytes) }
                imgRaw.setReadable(true, false)
                imgRaw.setWritable(true, false)
                maskRaw.setReadable(true, false)
                maskRaw.setWritable(true, false)

                val cmd = arrayOf(
                    "/system/bin/sh", "-c",
                    "cd $DEVICE_SD_DIR && " +
                    "export LD_LIBRARY_PATH=$DEVICE_SD_DIR:\$LD_LIBRARY_PATH && " +
                    "export ADSP_LIBRARY_PATH='$DEVICE_SD_DIR;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && " +
                    "rm -f sd_output.png 2>/dev/null; " +
                    "$runnerBin \"$sanitizedPos\"; " +
                    "RET=\$?; chmod 666 sd_output.png image.raw mask.raw 2>/dev/null; exit \$RET"
                )

                val startTemp = readPeakTemperatureC()
                val tracker = HardwareTelemetryTracker()
                tracker.start()

                val processStart = System.currentTimeMillis()
                val process = ProcessBuilder(*cmd).redirectErrorStream(true).start()
                val logText = process.inputStream.bufferedReader().readText()
                val exitCode = process.waitFor()
                val engineLatency = System.currentTimeMillis() - processStart
                val totalLatency = System.currentTimeMillis() - startTime
                val stagingLatency = (totalLatency - engineLatency).coerceAtLeast(0L)
                android.util.Log.d("OnDeviceProcessDriver", "$runnerBin exit=$exitCode engineLatency=${engineLatency}ms: $logText")
                val endTemp = readPeakTemperatureC()
                val (realPower, realEnergy, realDelta) = tracker.stop(totalLatency, startTemp, endTemp)
                val rawCurrent = readActiveCurrentUa()
                val rawVoltage = readActiveVoltageUv()

                if (exitCode == 0 && outPng.exists() && outPng.lastModified() >= startTime) {
                    val rawBmp = android.graphics.BitmapFactory.decodeFile(outPng.absolutePath)
                    val resultBmp = if (rawBmp != null) compositeInpaintResult(image, rawBmp, mask) else image
                    InferenceTelemetry(
                        modelName = runnerName,
                        executionMode = execMode,
                        latencyMs = totalLatency,
                        engineLatencyMs = engineLatency,
                        stagingLatencyMs = stagingLatency,
                        energyJoules = realEnergy,
                        powerWatts = realPower,
                        thermalDeltaC = realDelta,
                        startTempC = startTemp,
                        endTempC = endTemp,
                        rawCurrentUa = rawCurrent,
                        rawVoltageUv = rawVoltage,
                        resultBitmap = resultBmp
                    )
                } else {
                    throw RuntimeException("$runnerName failed on NPU (exit $exitCode):\n$logText")
                }
            } catch (e: Exception) {
                android.util.Log.e("OnDeviceProcessDriver", "SD execution error", e)
                throw e
            }
        }

        // =====================================================================
        // 2. SNPE Pipeline (MIGAN, LaMa, AOT-GAN) -> ALL EXECUTED ON NPU (DSP)
        // =====================================================================
        val isMigan = targetModel.contains("MIGAN")
        val isAot = targetModel.contains("AOT")
        val isLama = targetModel.contains("LAMA")

        // Use pre-compiled HTP init cache if present on device to eliminate cold-start
        val (dlcName, initCacheFlag) = when {
            isMigan -> Pair(
                if (File(DEVICE_LAMA_DIR, "migan_htp_v79.dlc").exists()) "migan_htp_v79.dlc" else "migan.dlc",
                ""
            )
            isAot -> if (File(DEVICE_LAMA_DIR, "bench_cache_aotgan.dlc").exists()) {
                Pair("bench_cache_aotgan.dlc", "--enable_init_cache")
            } else {
                Pair("aotgan.dlc", "")
            }
            isLama -> if (File(DEVICE_LAMA_DIR, "bench_cache_lama_dilated.dlc").exists()) {
                Pair("bench_cache_lama_dilated.dlc", "--enable_init_cache")
            } else {
                Pair("lama_dilated.dlc", "")
            }
            else -> Pair(
                if (File(DEVICE_LAMA_DIR, "migan_htp_v79.dlc").exists()) "migan_htp_v79.dlc" else "migan.dlc",
                ""
            )
        }

        // ALL models run on Hexagon NPU via FastRPC
        val runtimeFlag = "--use_dsp"
        val displayName = when {
            isMigan -> "MIGAN (Portraits & Speed)"
            isAot -> "AOT-GAN (Dense Texture)"
            isLama -> "LaMa Dilated (Large Voids)"
            else -> targetModel
        }
        val execMode = "Qualcomm Hexagon NPU (HTP v79 DSP Live)"

        val stagingStart = System.currentTimeMillis()
        val inputDir = File(DEVICE_LAMA_DIR, "input_v3")
        if (!inputDir.exists()) inputDir.mkdirs()
        inputDir.setReadable(true, false)
        inputDir.setWritable(true, false)
        inputDir.setExecutable(true, false)

        val rawImgFile = File(inputDir, "live_img_v3.raw")
        val rawMaskFile = File(inputDir, "live_mask_v3.raw")
        val listFile = File(DEVICE_LAMA_DIR, "live_input_v3.txt")
        val outDir = File(DEVICE_LAMA_DIR, "app_live_output_v3")
        if (!outDir.exists()) outDir.mkdirs()
        outDir.setReadable(true, false)
        outDir.setWritable(true, false)
        outDir.setExecutable(true, false)

        // 1. Image Staging: Check if already pre-cached in background
        val imageMatches = isImagePrecached && rawImgFile.exists() && (precachedImageRef === image || (precachedImageGenId != -1 && precachedImageGenId == image.generationId))
        if (!imageMatches) {
            if (precachedImageRef === image && imagePrecacheFuture != null) {
                try {
                    imagePrecacheFuture?.get(500, java.util.concurrent.TimeUnit.MILLISECONDS)
                } catch (_: Exception) {}
            }
            if (!isImagePrecached || !rawImgFile.exists() || (precachedImageRef !== image && precachedImageGenId != image.generationId)) {
                val imgBytes = bitmapToFloat32Raw(image)
                FileOutputStream(rawImgFile).use { it.write(imgBytes) }
                rawImgFile.setReadable(true, false)
                rawImgFile.setWritable(true, false)
                precachedImageRef = image
                precachedImageGenId = image.generationId
                isImagePrecached = true
            }
        }

        // 2. Mask Staging: Check if already pre-cached with correct polarity
        val maskMatches = isMaskPrecached && rawMaskFile.exists() && precachedMaskIsInverted == isMigan && (precachedMaskRef === mask || (precachedMaskGenId != -1 && precachedMaskGenId == mask.generationId))
        if (!maskMatches) {
            if (precachedMaskRef === mask && precachedMaskIsInverted == isMigan && maskPrecacheFuture != null) {
                try {
                    maskPrecacheFuture?.get(500, java.util.concurrent.TimeUnit.MILLISECONDS)
                } catch (_: Exception) {}
            }
            if (!isMaskPrecached || !rawMaskFile.exists() || (precachedMaskRef !== mask && precachedMaskGenId != mask.generationId) || precachedMaskIsInverted != isMigan) {
                val maskBytes = maskToFloat32Raw(mask, inverted = isMigan)
                FileOutputStream(rawMaskFile).use { it.write(maskBytes) }
                rawMaskFile.setReadable(true, false)
                rawMaskFile.setWritable(true, false)
                listFile.writeText("image:=input_v3/live_img_v3.raw mask:=input_v3/live_mask_v3.raw\n")
                listFile.setReadable(true, false)
                listFile.setWritable(true, false)
                precachedMaskRef = mask
                precachedMaskGenId = mask.generationId
                precachedMaskIsInverted = isMigan
                isMaskPrecached = true
            }
        }

        val stagingLatency = System.currentTimeMillis() - stagingStart

        val outputCandidates = if (isMigan) {
            arrayOf("output_0.raw", "painted_image.raw")
        } else {
            arrayOf("painted_image.raw", "output_0.raw")
        }
        val targetOutputFile = File(outDir, "Result_0/${outputCandidates[0]}")
        val resDir = File(outDir, "Result_0")
        if (!resDir.exists()) resDir.mkdirs()

        // ---------------------------------------------------------------------
        // FastRPC Resident Daemon Engine Execution (Sub-50ms Steady-State NPU)
        // ---------------------------------------------------------------------
        val daemonKey = when {
            isMigan -> "migan"
            isLama -> "lama"
            isAot -> "aotgan"
            else -> "migan"
        }

        return try {
            val startTemp = readPeakTemperatureC()
            val tracker = HardwareTelemetryTracker()
            tracker.start()

            var engineLatency = 0L
            var residentSuccess = false
            var execMode = "Qualcomm Hexagon NPU (HTP v79 DSP Live)"
            var resultRawFile: File? = null
            var fallbackExitCode = 0
            var fallbackLogText = ""

            val (daemonOk, daemonLatency) = executeViaResidentDaemon(daemonKey, rawImgFile, rawMaskFile, targetOutputFile)
            if (daemonOk && targetOutputFile.exists() && targetOutputFile.length() > 0) {
                residentSuccess = true
                engineLatency = daemonLatency
                resultRawFile = targetOutputFile
                execMode = "Qualcomm Hexagon NPU (HTP v79 Resident Engine ⚡)"
                android.util.Log.i("OnDeviceProcessDriver", "⚡ In-Process FastRPC Resident Daemon executed $daemonKey in ${daemonLatency}ms")
            } else {
                // Auto-start daemon in background for subsequent requests if not alive
                ModelPreloadManager.ensureResidentDaemonRunning()

                // Subprocess Fallback with burst mode
                val engineStart = System.currentTimeMillis()
                val cmd = arrayOf(
                    "/system/bin/sh", "-c",
                    "export LD_LIBRARY_PATH=$DEVICE_LAMA_DIR/lib:$DEVICE_LAMA_DIR; " +
                    "export ADSP_LIBRARY_PATH='$DEVICE_LAMA_DIR/dsp/lib;$DEVICE_LAMA_DIR/dsp;/dsp'; " +
                    "export PATH=\$PATH:$DEVICE_LAMA_DIR; " +
                    "cd $DEVICE_LAMA_DIR && " +
                    "mkdir -p $DEVICE_LAMA_DIR/app_live_output_v3 && " +
                    "rm -f $DEVICE_LAMA_DIR/app_live_output_v3/Result_0/*.raw 2>/dev/null; " +
                    "./snpe-net-run --container $dlcName --input_list live_input_v3.txt --output_dir $DEVICE_LAMA_DIR/app_live_output_v3 $runtimeFlag $initCacheFlag --perf_profile burst --profiling_level off --priority_hint critical; " +
                    "RET=\$?; chmod -R 777 $DEVICE_LAMA_DIR/app_live_output_v3 $DEVICE_LAMA_DIR/input_v3 2>/dev/null; exit \$RET"
                )
                val process = ProcessBuilder(*cmd).redirectErrorStream(true).start()
                fallbackLogText = process.inputStream.bufferedReader().readText()
                fallbackExitCode = process.waitFor()
                engineLatency = System.currentTimeMillis() - engineStart
                android.util.Log.d("OnDeviceProcessDriver", "snpe-net-run ($dlcName) exit=$fallbackExitCode engineLatency=${engineLatency}ms stagingLatency=${stagingLatency}ms: $fallbackLogText")

                for (candName in outputCandidates) {
                    val directFile = File(outDir, "Result_0/$candName")
                    if (directFile.exists() && directFile.length() > 0 && directFile.lastModified() >= startTime - 1000) {
                        resultRawFile = directFile
                        break
                    }
                }

                if (resultRawFile == null) {
                    outDir.walkTopDown().forEach { file ->
                        if (file.name in outputCandidates && file.length() > 0 && file.lastModified() >= startTime - 1000) {
                            if (resultRawFile == null || file.name == outputCandidates[0]) {
                                resultRawFile = file
                            }
                        }
                    }
                }
            }

            val totalLatency = System.currentTimeMillis() - startTime
            val endTemp = readPeakTemperatureC()
            val (realPower, realEnergy, realDelta) = tracker.stop(totalLatency, startTemp, endTemp)
            val rawCurrent = readActiveCurrentUa()
            val rawVoltage = readActiveVoltageUv()

            if (resultRawFile != null && resultRawFile!!.exists()) {
                val rawOutBytes = resultRawFile!!.readBytes()
                val rawBmp = rawFloat32ToBitmap(rawOutBytes, 512, 512)
                val resultBmp = compositeInpaintResult(image, rawBmp, mask)
                InferenceTelemetry(
                    modelName = displayName,
                    executionMode = execMode,
                    latencyMs = totalLatency,
                    engineLatencyMs = engineLatency,
                    stagingLatencyMs = stagingLatency,
                    energyJoules = realEnergy,
                    powerWatts = realPower,
                    thermalDeltaC = realDelta,
                    startTempC = startTemp,
                    endTempC = endTemp,
                    rawCurrentUa = rawCurrent,
                    rawVoltageUv = rawVoltage,
                    resultBitmap = resultBmp
                )
            } else {
                throw RuntimeException("SNPE ($dlcName) inference failed on NPU (exit code $fallbackExitCode):\n$fallbackLogText")
            }
        } catch (e: Exception) {
            android.util.Log.e("OnDeviceProcessDriver", "SNPE execution error", e)
            throw e
        }
    }

    private fun isHolePixel(c: Int): Boolean {
        val a = Color.alpha(c)
        if (a <= 50) return false
        val r = Color.red(c)
        val g = Color.green(c)
        val b = Color.blue(c)
        val luma = (r * 299 + g * 587 + b * 114) / 1000
        return luma > 128 || r > 128
    }

    fun compositeInpaintResult(original: Bitmap, inpaintOutput: Bitmap, mask: Bitmap): Bitmap {
        val w = 512
        val h = 512
        val origScaled = Bitmap.createScaledBitmap(original, w, h, true)
        val outScaled = Bitmap.createScaledBitmap(inpaintOutput, w, h, true)
        val maskScaled = Bitmap.createScaledBitmap(mask, w, h, false)

        val origPixels = IntArray(w * h)
        val outPixels = IntArray(w * h)
        val maskPixels = IntArray(w * h)
        val finalPixels = IntArray(w * h)

        origScaled.getPixels(origPixels, 0, w, 0, 0, w, h)
        outScaled.getPixels(outPixels, 0, w, 0, 0, w, h)
        maskScaled.getPixels(maskPixels, 0, w, 0, 0, w, h)

        val maskWeights = FloatArray(w * h)
        for (i in 0 until (w * h)) {
            maskWeights[i] = if (isHolePixel(maskPixels[i])) 1.0f else 0.0f
        }

        // Fast separable 3x3 box blur on mask weights for a clean, seamless transition at hole boundary
        val blurredMask = FloatArray(w * h)
        val tempMask = FloatArray(w * h)

        for (y in 0 until h) {
            val yOffset = y * w
            for (x in 0 until w) {
                val x0 = if (x > 0) x - 1 else 0
                val x1 = x
                val x2 = if (x < w - 1) x + 1 else w - 1
                tempMask[yOffset + x] = (maskWeights[yOffset + x0] + maskWeights[yOffset + x1] + maskWeights[yOffset + x2]) / 3.0f
            }
        }
        for (x in 0 until w) {
            for (y in 0 until h) {
                val y0 = if (y > 0) y - 1 else 0
                val y1 = y
                val y2 = if (y < h - 1) y + 1 else h - 1
                blurredMask[y * w + x] = (tempMask[y0 * w + x] + tempMask[y1 * w + x] + tempMask[y2 * w + x]) / 3.0f
            }
        }

        for (i in 0 until (w * h)) {
            val maskNorm = blurredMask[i]
            val invNorm = 1.0f - maskNorm

            val origC = origPixels[i]
            val outC = outPixels[i]

            val r = ((Color.red(origC) * invNorm) + (Color.red(outC) * maskNorm)).toInt().coerceIn(0, 255)
            val g = ((Color.green(origC) * invNorm) + (Color.green(outC) * maskNorm)).toInt().coerceIn(0, 255)
            val b = ((Color.blue(origC) * invNorm) + (Color.blue(outC) * maskNorm)).toInt().coerceIn(0, 255)

            finalPixels[i] = Color.rgb(r, g, b)
        }

        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        result.setPixels(finalPixels, 0, w, 0, 0, w, h)
        return result
    }

    private val LUT_NORM_255 = FloatArray(256) { it / 255.0f }

    private fun bitmapToFloat32Raw(bitmap: Bitmap): ByteArray {
        val w = 512
        val h = 512
        val scaled = if (bitmap.width == w && bitmap.height == h) bitmap else Bitmap.createScaledBitmap(bitmap, w, h, true)
        val pixels = IntArray(w * h)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)

        val buffer = ByteBuffer.allocate(w * h * 3 * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (c in pixels) {
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            buffer.putFloat(LUT_NORM_255[r])
            buffer.putFloat(LUT_NORM_255[g])
            buffer.putFloat(LUT_NORM_255[b])
        }
        return buffer.array()
    }

    private fun maskToFloat32Raw(mask: Bitmap, inverted: Boolean): ByteArray {
        val w = 512
        val h = 512
        val scaled = if (mask.width == w && mask.height == h) mask else Bitmap.createScaledBitmap(mask, w, h, false)
        val pixels = IntArray(w * h)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)

        val buffer = ByteBuffer.allocate(w * h * 1 * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (c in pixels) {
            val isHole = isHolePixel(c)
            val v = if (inverted) {
                if (isHole) 0.0f else 1.0f
            } else {
                if (isHole) 1.0f else 0.0f
            }
            buffer.putFloat(v)
        }
        return buffer.array()
    }

    private fun rawFloat32ToBitmap(rawBytes: ByteArray, width: Int, height: Int): Bitmap {
        val totalPixels = width * height
        val expectedFloats = totalPixels * 3
        val buffer = ByteBuffer.wrap(rawBytes).order(ByteOrder.LITTLE_ENDIAN)
        val floatBuf = buffer.asFloatBuffer()
        val floatsAvailable = floatBuf.remaining()

        if (floatsAvailable < expectedFloats) {
            throw IllegalArgumentException("Raw buffer underflow: expected at least $expectedFloats floats, got $floatsAvailable")
        }

        // Fast bulk float transfer to eliminate per-float JNI overhead
        val floats = FloatArray(expectedFloats)
        floatBuf.get(floats)

        // Quick sample scan to detect dynamic range without scanning all 786,432 elements
        var minVal = Float.MAX_VALUE
        var maxVal = -Float.MAX_VALUE
        val step = 16
        for (i in 0 until expectedFloats step step) {
            val v = floats[i]
            if (v < minVal) minVal = v
            if (v > maxVal) maxVal = v
        }

        val isMinusOneToOne = (minVal < -0.1f)
        val isZeroToOne = (!isMinusOneToOne && maxVal <= 1.5f)

        // Single-pass direct bitwise ARGB packing
        val pixels = IntArray(totalPixels)
        var fIdx = 0
        for (i in 0 until totalPixels) {
            val rRaw = floats[fIdx++]
            val gRaw = floats[fIdx++]
            val bRaw = floats[fIdx++]

            val r = when {
                isMinusOneToOne -> ((rRaw + 1.0f) * 127.5f).toInt().coerceIn(0, 255)
                isZeroToOne -> (rRaw * 255.0f).toInt().coerceIn(0, 255)
                else -> rRaw.toInt().coerceIn(0, 255)
            }
            val g = when {
                isMinusOneToOne -> ((gRaw + 1.0f) * 127.5f).toInt().coerceIn(0, 255)
                isZeroToOne -> (gRaw * 255.0f).toInt().coerceIn(0, 255)
                else -> gRaw.toInt().coerceIn(0, 255)
            }
            val b = when {
                isMinusOneToOne -> ((bRaw + 1.0f) * 127.5f).toInt().coerceIn(0, 255)
                isZeroToOne -> (bRaw * 255.0f).toInt().coerceIn(0, 255)
                else -> bRaw.toInt().coerceIn(0, 255)
            }

            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }

    /**
     * Executes inference via the resident FastRPC C++ daemon running on 127.0.0.1:55555.
     * The daemon keeps MIGAN, LaMa, and AOT-GAN graphs permanently resident in FastRPC CDSP memory,
     * eliminating fork/execve subprocess overhead, dynamic linker overhead, and driver attach overhead.
     * Returns Pair(success, engineLatencyMs).
     */
    private fun executeViaResidentDaemon(
        modelKey: String,
        imgFile: File,
        maskFile: File,
        outFile: File
    ): Pair<Boolean, Long> {
        return try {
            val s = java.net.Socket()
            s.connect(java.net.InetSocketAddress("127.0.0.1", 55555), 250) // 250ms connect timeout
            s.soTimeout = 5000 // 5s read timeout
            val writer = java.io.PrintWriter(s.getOutputStream(), true)
            val reader = java.io.BufferedReader(java.io.InputStreamReader(s.getInputStream()))

            val t0 = System.currentTimeMillis()
            writer.println("INFER $modelKey ${imgFile.absolutePath} ${maskFile.absolutePath} ${outFile.absolutePath}")
            val resp = reader.readLine()
            val roundtripMs = System.currentTimeMillis() - t0
            s.close()

            if (resp != null && resp.startsWith("OK")) {
                val parts = resp.split(" ")
                val engineReportedMs = if (parts.size >= 2) parts[1].toDoubleOrNull()?.toLong() ?: roundtripMs else roundtripMs
                Pair(true, engineReportedMs)
            } else {
                android.util.Log.w("OnDeviceProcessDriver", "Resident daemon returned error or empty response: $resp")
                Pair(false, 0L)
            }
        } catch (e: Exception) {
            android.util.Log.d("OnDeviceProcessDriver", "Resident daemon unavailable (${e.message}), falling back to snpe-net-run")
            Pair(false, 0L)
        }
    }
}
