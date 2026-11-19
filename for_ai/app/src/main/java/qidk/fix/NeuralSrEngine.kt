package qidk.fix

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import com.qualcomm.qidk.inpaint.engine.OnDeviceProcessDriver
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "QIDK.SR"

/**
 * One 512 x 512 call of an inpainting network on the resident FastRPC daemon. The caller works in canonical form
 * (image [0,1], hole = 1); the [contract] decides what is actually written (mask polarity, [0,1] or [-1,1]) and how the
 * output is read back. Files live in the same directories the Inpaint screen already uses, which the daemon is known to reach.
 */
class DaemonTileModel(private val daemonKey: String, private val contract: ModelContract, tag: String) : TileModel {
    private val n = SrWindows.WIN
    private val imgBytes = n * n * 3 * 4
    private val maskBytes = n * n * 4
    private val inDir = File(LAMA_DIR, "input").apply { mkdirs(); setReadable(true, false); setWritable(true, false); setExecutable(true, false) }
    private val outDir = File(LAMA_DIR, "app_live_output_v4").apply { mkdirs(); setReadable(true, false); setWritable(true, false); setExecutable(true, false) }
    private val imgFile = File(inDir, "sr_img_$tag.raw")
    private val maskFile = File(inDir, "sr_mask_$tag.raw")
    private val outFile = File(outDir, "sr_out_$tag.raw")
    private val imgBuf = ByteBuffer.allocateDirect(imgBytes).order(ByteOrder.LITTLE_ENDIAN)
    private val maskBuf = ByteBuffer.allocateDirect(maskBytes).order(ByteOrder.LITTLE_ENDIAN)
    private val outBuf = ByteBuffer.allocateDirect(imgBytes).order(ByteOrder.LITTLE_ENDIAN)

    private fun writeFile(f: File, buf: ByteBuffer, len: Int) {
        buf.position(0); buf.limit(len)
        RandomAccessFile(f, "rw").use { raf -> raf.setLength(0); raf.channel.write(buf) }
        f.setReadable(true, false); f.setWritable(true, false)
    }

    override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock): Boolean {
        return try {
            clock.prep {
                imgBuf.clear()
                val fi = imgBuf.asFloatBuffer()
                if (contract.minusOneToOne) { for (v in image) fi.put(v * 2f - 1f) } else fi.put(image, 0, n * n * 3)
                maskBuf.clear()
                val fm = maskBuf.asFloatBuffer()
                val holeV = contract.holeValue
                val keepV = 1f - holeV
                for (i in 0 until n * n) fm.put(if (hole[i] > 0.5f) holeV else keepV)
                writeFile(imgFile, imgBuf, imgBytes)
                writeFile(maskFile, maskBuf, maskBytes)
                if (outFile.exists()) outFile.delete()
            }
            val t0 = System.nanoTime()
            val (ok, _) = OnDeviceProcessDriver.executeViaResidentDaemon(daemonKey, imgFile, maskFile, outFile)
            clock.addInferMs((System.nanoTime() - t0) / 1_000_000L)
            if (!ok || !outFile.exists() || outFile.length() < imgBytes) {
                Log.w(TAG, "daemon $daemonKey: no output (ok=$ok, bytes=${if (outFile.exists()) outFile.length() else -1})")
                return false
            }
            clock.post {
                outBuf.clear()
                RandomAccessFile(outFile, "r").use { raf -> raf.channel.read(outBuf) }
                outBuf.position(0)
                outBuf.asFloatBuffer().get(out, 0, n * n * 3)
                if (contract.minusOneToOne) for (i in 0 until n * n * 3) out[i] = (out[i] + 1f) * 0.5f
            }
            true
        } catch (e: Exception) {
            Log.w(TAG, "daemon $daemonKey call failed: ${e.message}")
            false
        }
    }

    companion object {
        const val LAMA_DIR = "/data/local/tmp/lama"
    }
}

/**
 * Neural super-resolution with the NPU/GPU/CPU inpainting networks, by Null-Space Detail Transfer (see [SrPlan] and the
 * repair guide). Every 2x pass:
 *   1. U = classical 2x base (Keys cubic) of the previous result, evaluated window by window (no full-size float planes);
 *   2. two complementary checkerboards of 32 px cells are inpainted by the network (50% context, holes the networks were
 *      trained for, not the 75% pixel lattice that produced the "screen door");
 *   3. X = U + alpha * soft(P(O - U)): only the part of the network output that the low-resolution grid cannot represent
 *      is kept, so geometry cannot move and the mesh cannot form;
 *   4. gates (identity, lattice, unstable output, LR consistency) decide whether the pass is accepted; a rejected pass is
 *      replaced by the classical base and the reason is reported. The result is never worse than bicubic.
 * Before the first window the engine probes which mask polarity and value range the installed network really uses.
 */
class NeuralSrEngine(
    override val id: String,
    override val label: String,
    override val family: ModelFamily,
    private val baseKey: String
) : SrEngine {

    /** 0..1: how much of the transferred detail is added. 0.6 is the default; the UI slider writes it. */
    @Volatile var detailStrength: Float = 0.6f

    override val unavailableReason: String?
        get() = if (!SrNative.available) "libqidk_sr.so is not loaded: rebuild the app with the NDK (CMake target qidk_sr)" else null

    private fun suffix(backend: Backend) = when (backend) { Backend.NPU -> ""; Backend.GPU -> "_gpu"; Backend.CPU -> "_cpu" }

    override fun runWithTelemetry(src: Bitmap, backend: Backend, scale: Int, progress: SrProgress?): SrExecutionResult {
        BackendMatrix.require(family, backend)
        val passes = SrRun.passesFor(scale)
        unavailableReason?.let { error(it) }

        val monitor = PowerMonitor.get().also { it.start() }
        val token = monitor.begin(SrRun.group(backend))
        val clock = PhaseClock()
        val notes = ArrayList<String>()
        val source = SrRun.prepareSource(src)
        val key = baseKey + suffix(backend)
        val kernels = NativeKernels()
        val alpha = detailStrength.coerceIn(0f, 1f)

        val probe = clock.prep { contractFor(key, clock) }
        if (probe.contract == null) {
            notes += "I/O probe failed for $key (${probe.lines.joinToString("; ")}): the classical base is used"
        } else {
            notes += "I/O contract of $key: ${probe.contract.describe()}"
        }
        Log.i(TAG, "[$id] key=$key scale=${scale}x alpha=$alpha contract=${probe.contract?.describe()} probe=${probe.lines}")

        var useNetwork = probe.contract != null && alpha > 0f
        var current = source
        try {
            for (pass in 1..passes) {
                val dst = SrRun.newOutput(current.width * 2, current.height * 2)
                try {
                    val model: TileModel? = probe.contract?.let { DaemonTileModel(key, it, id) }
                    val runner = NsdtPass(kernels, model ?: NoModel, clock, NsdtParams(alpha = alpha))
                    val srcImg = BitmapImage(current)
                    val dstImg = BitmapImage(dst)
                    var accepted = false
                    if (useNetwork) {
                        val outcome = runner.run(srcImg, dstImg, pass, passes, progress)
                        if (outcome.cancelled) throw SrCancelled()
                        accepted = outcome.gate.accepted
                        notes += "pass $pass (${current.width} -> ${dst.width}): " +
                            (if (accepted) "accepted" else "rejected (${outcome.gate.reason}), classical base used") + " | " + outcome.gate.metrics
                        Log.i(TAG, "[$id] pass $pass verdict=${outcome.gate.verdict} ${outcome.gate.metrics} failures=${outcome.modelFailures}")
                        if (!accepted) useNetwork = false          // the next pass would fail the same way: do not waste NPU time
                    } else if (pass == 1 && alpha <= 0f) {
                        notes += "detail strength 0: classical base only"
                    }
                    if (!accepted) {
                        runner.runClassical(srcImg, dstImg, progress, pass, passes)
                        if (progress?.cancelled == true) throw SrCancelled()
                    }
                } catch (t: Throwable) {
                    dst.recycle()
                    throw t
                }
                if (current !== source) current.recycle()
                current = dst
            }
        } catch (t: Throwable) {
            if (current !== source && current !== src) current.recycle()
            throw t
        }

        // Whole-picture low-resolution consistency of the final result (report only).
        val psnr = SrNative.lrPsnr(source, current, scale)
        if (psnr > 0) {
            notes += String.format(java.util.Locale.US, "LR consistency (box %dx down-sample vs input): %.1f dB", scale, psnr)
            Log.i(TAG, "[$id] lrPsnr=$psnr")
        }
        val telemetry = monitor.end(token, clock, SrRun.coreLabel(backend))
        Log.i(TAG, "[$id] done ${scale}x: prep=${telemetry.tPrepMs} infer=${telemetry.tInferMs} post=${telemetry.tPostMs} total=${telemetry.tTotalMs} ms")
        return SrExecutionResult(current, telemetry, notes)
    }

    private fun contractFor(key: String, clock: PhaseClock): ContractProbe.Result {
        probeCache[key]?.let { return it }
        val r = ContractProbe.run(ContractCandidates.forKey(baseKey), { DaemonTileModel(key, it, "probe_$id") }, clock)
        // a daemon that is simply down must be re-probed next time; a real verdict (pass or fail) is kept for the process
        if (r.contract != null || r.lines.any { !it.contains("model call failed") }) probeCache[key] = r
        return r
    }

    private object NoModel : TileModel {
        override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock) = false
    }

    companion object {
        private val probeCache = ConcurrentHashMap<String, ContractProbe.Result>()
        fun resetProbeCache() = probeCache.clear()
    }
}

/**
 * Stable Diffusion detail transfer ("SD upscale", SDEdit on tiles): the classical 2x base of every 512 px tile is
 * re-noised to a low level and denoised by the inpainting UNet with an EMPTY mask, then only the null-space part of the
 * result is kept, exactly as for the GAN engines. Needs the img2img-capable runner (marker file sd_img2img_v1, patch in
 * REPO/StableDiffusion/include/sd_img2img.hpp). Only the first 2x pass uses the diffusion model (about 9 tiles at 1024 px);
 * further passes use the classical base, because a 4096 px SD pass would take 121 tiles.
 */
class SdSuperResEngine(
    override val id: String = "sd_sr",
    override val label: String = "Stable Diffusion detail transfer (img2img, NPU)"
) : SrEngine {
    override val family: ModelFamily = ModelFamily.SR_SD
    @Volatile var detailStrength: Float = 0.8f

    override val unavailableReason: String?
        get() = when {
            !SrNative.available -> "libqidk_sr.so is not loaded: rebuild the app with the NDK"
            !File(SD_DIR, "sd_cli_v2").exists() && !File(SD_DIR, "sd_qidk_runner_inpaint").exists() && !File(SD_DIR, MARKER).exists() ->
                "the SD runner is not installed in $SD_DIR"
            else -> null
        }

    override fun runWithTelemetry(src: Bitmap, backend: Backend, scale: Int, progress: SrProgress?): SrExecutionResult {
        BackendMatrix.require(family, backend)
        val passes = SrRun.passesFor(scale)
        unavailableReason?.let { error(it) }
        val monitor = PowerMonitor.get().also { it.start() }
        val token = monitor.begin(ThermalZones.Group.NPU)
        val clock = PhaseClock()
        val notes = ArrayList<String>()
        val source = SrRun.prepareSource(src)
        val kernels = NativeKernels()
        var current = source
        var useSd = true
        try {
            for (pass in 1..passes) {
                val dst = SrRun.newOutput(current.width * 2, current.height * 2)
                try {
                    val srcImg = BitmapImage(current)
                    val dstImg = BitmapImage(dst)
                    val runner = NsdtPass(kernels, NoSdModel, clock)
                    var done = false
                    if (useSd && pass == 1) {
                        val outcome = sdPass(kernels, srcImg, dstImg, clock, progress)
                        if (outcome.cancelled) throw SrCancelled()
                        done = outcome.gate.accepted
                        notes += "pass 1 (SD img2img): " + (if (done) "accepted" else "rejected (${outcome.gate.reason}), classical base used") + " | " + outcome.gate.metrics
                    } else if (pass > 1) {
                        notes += "pass $pass: classical base (diffusion is applied to the first 2x pass only)"
                    }
                    useSd = false
                    if (!done) {
                        runner.runClassical(srcImg, dstImg, progress, pass, passes)
                        if (progress?.cancelled == true) throw SrCancelled()
                    }
                } catch (t: Throwable) {
                    dst.recycle()
                    throw t
                }
                if (current !== source) current.recycle()
                current = dst
            }
        } catch (t: Throwable) {
            if (current !== source && current !== src) current.recycle()
            throw t
        }
        return SrExecutionResult(current, monitor.end(token, clock, SrRun.coreLabel(Backend.NPU, resident = false)), notes)
    }

    private fun sdPass(k: SrKernels, src: SrImage, dst: SrImage, clock: PhaseClock, progress: SrProgress?): PassOutcome {
        val n = SrWindows.WIN
        val wins = SrWindows.plan(dst.width, dst.height)
        val u = FloatArray(n * n * 3)
        val o = FloatArray(n * n * 3)
        val agg = SrAggregate()
        val alpha = detailStrength.coerceIn(0f, 1f)
        var failures = 0
        for ((i, w) in wins.withIndex()) {
            if (progress?.cancelled == true) return PassOutcome(SrGateResult(SrVerdict.UNSTABLE, "cancelled", SrGates.metrics(agg)), agg, true, false, i, failures)
            progress?.report(1, 1, i, wins.size, "SD tile ${i + 1}/${wins.size} (about a few seconds per tile)")
            check(clock.prep { k.sampleBase(src, w.x0, w.y0, n, u) }) { "native sampler refused tile $i" }
            val ok = runTile(u, o, clock, seed = 42 + i)
            if (!ok) { failures++; if (failures >= 2) return PassOutcome(SrGateResult(SrVerdict.UNSTABLE, "the SD runner failed on two tiles", SrGates.metrics(agg)), agg, false, true, i + 1, failures) }
            val stats = clock.post {
                k.compose(u, if (ok) o else null, null, n, w.x0, w.y0, 0, 0, alpha, 0.08f, 0.30f,
                    w.coreX, w.coreY, w.coreW, w.coreH, dst, w.dstX, w.dstY) ?: error("native compose failed for tile $i")
            }
            agg.add(stats)
        }
        return PassOutcome(SrGates.evaluate(agg), agg, false, false, wins.size, failures)
    }

    /** One img2img tile through the patched runner: image.raw = base tile, mask.raw = all zeros, sd_output.png = detail result. */
    private fun runTile(u: FloatArray, out: FloatArray, clock: PhaseClock, seed: Int): Boolean {
        val n = SrWindows.WIN
        return try {
            val dir = File(SD_DIR)
            val imgRaw = File(dir, "image.raw"); val maskRaw = File(dir, "mask.raw"); val outPng = File(dir, "sd_output.png")
            clock.prep {
                if (outPng.exists()) outPng.delete()
                val bb = ByteBuffer.allocate(n * n * 3 * 4).order(ByteOrder.LITTLE_ENDIAN)
                bb.asFloatBuffer().put(u, 0, n * n * 3)
                FileOutputStream(imgRaw).use { it.write(bb.array()) }
                FileOutputStream(maskRaw).use { it.write(ByteArray(n * n * 4)) }          // all zeros = nothing is a hole
                imgRaw.setReadable(true, false); imgRaw.setWritable(true, false)
                maskRaw.setReadable(true, false); maskRaw.setWritable(true, false)
            }
            val script = SdRunnerCommand.script(
                "./sd_qidk_runner_inpaint", SD_DIR,
                listOf(
                    "export LD_LIBRARY_PATH=$SD_DIR:\$LD_LIBRARY_PATH",
                    "export ADSP_LIBRARY_PATH='$SD_DIR;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'"
                ),
                steps = 12, cfg = 2.0f, seed = seed, extra = listOf("--strength", "0.30", "--no-composite")
            )
            val pb = ProcessBuilder("/system/bin/sh", "-c", script).redirectErrorStream(true)
            pb.environment().putAll(SdRunnerCommand.environment(SR_PROMPT, SR_NEGATIVE))
            val code = clock.infer {
                val p = pb.start()
                val log = p.inputStream.bufferedReader().readText()
                val c = p.waitFor()
                Log.d(TAG, "sd img2img tile exit=$c: ${log.takeLast(300)}")
                c
            }
            if (code != 0 || !outPng.exists()) return false
            clock.post {
                val bmp = BitmapFactory.decodeFile(outPng.absolutePath) ?: return@post false
                val b = if (bmp.width == n && bmp.height == n) bmp else Bitmap.createScaledBitmap(bmp, n, n, true)
                val px = IntArray(n * n)
                b.getPixels(px, 0, n, 0, 0, n, n)
                for (i in 0 until n * n) {
                    val c = px[i]
                    out[i * 3] = ((c shr 16) and 0xFF) / 255f
                    out[i * 3 + 1] = ((c shr 8) and 0xFF) / 255f
                    out[i * 3 + 2] = (c and 0xFF) / 255f
                }
                true
            }
        } catch (e: Exception) {
            Log.w(TAG, "sd tile failed: ${e.message}")
            false
        }
    }

    private object NoSdModel : TileModel {
        override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock) = false
    }

    companion object {
        const val SD_DIR = "/data/local/tmp/sd_runtime"
        const val MARKER = "sd_img2img_v1"
        private const val SR_PROMPT = "high quality photograph, sharp focus, natural fine texture, highly detailed"
        private const val SR_NEGATIVE = "blurry, lowres, jpeg artifacts, oversharpened, painting, cartoon, noise"
    }
}
