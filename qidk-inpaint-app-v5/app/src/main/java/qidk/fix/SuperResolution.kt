package qidk.fix

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** Result of one super-resolution run. [notes] are the lines the UI prints under the result (gate verdicts, fallbacks). */
data class SrExecutionResult(
    val bitmap: Bitmap,
    val telemetry: RunTelemetry,
    val notes: List<String> = emptyList()
)

class SrCancelled : RuntimeException("Super-resolution cancelled")

/** Super-resolution engine. 2x passes are chained for 4x and 8x. */
interface SrEngine {
    val id: String
    val label: String
    val family: ModelFamily
    /** null = usable. Otherwise why it is disabled; the UI greys the entry out and shows this text. */
    val unavailableReason: String?

    /** Largest scale the engine supports (2, 4 or 8). */
    val maxScale: Int get() = 8

    fun run(src: Bitmap, backend: Backend, scale: Int): Bitmap = runWithTelemetry(src, backend, scale, null).bitmap

    /** Throws [SrCancelled] when [progress] reports cancellation. */
    fun runWithTelemetry(src: Bitmap, backend: Backend, scale: Int, progress: SrProgress? = null): SrExecutionResult
}

/** Shared helpers of the engines. */
object SrRun {
    fun passesFor(scale: Int): Int {
        require(scale == 2 || scale == 4 || scale == 8) { "Scale factor must be 2, 4 or 8 (got $scale)" }
        return when (scale) { 2 -> 1; 4 -> 2; else -> 3 }
    }

    fun group(backend: Backend): ThermalZones.Group = when (backend) {
        Backend.NPU -> ThermalZones.Group.NPU
        Backend.GPU -> ThermalZones.Group.GPU
        Backend.CPU -> ThermalZones.Group.CPU
    }

    fun coreLabel(backend: Backend, resident: Boolean = true): String = when (backend) {
        Backend.NPU -> if (resident) "Qualcomm Hexagon HTP v79 NPU (resident FastRPC)" else "Qualcomm Hexagon HTP v79 NPU"
        Backend.GPU -> "Qualcomm Adreno 830 GPU"
        Backend.CPU -> "Qualcomm Oryon CPU (ARM64)"
    }

    /** ARGB_8888, mutable, even sides: what the native kernels need. Returns [src] itself when it already qualifies. */
    fun prepareSource(src: Bitmap): Bitmap {
        var b = src
        if (b.config != Bitmap.Config.ARGB_8888) b = b.copy(Bitmap.Config.ARGB_8888, false)
        val w = b.width and 1.inv()
        val h = b.height and 1.inv()
        if (w != b.width || h != b.height) b = Bitmap.createBitmap(b, 0, 0, w, h)
        return b
    }

    /** Opaque, mutable ARGB_8888 destination. */
    fun newOutput(w: Int, h: Int): Bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { it.setHasAlpha(false) }
}

/** Placeholder for an engine that cannot run on this device; the UI greys the entry out and prints [reason]. */
class UnavailableSrEngine(
    override val id: String, override val label: String, override val family: ModelFamily, reason: String
) : SrEngine {
    override val unavailableReason: String? = reason
    override fun runWithTelemetry(src: Bitmap, backend: Backend, scale: Int, progress: SrProgress?): SrExecutionResult =
        error("$label is disabled: $unavailableReason")
}

/** Plain bicubic via OpenCV: the reference every other engine is compared against. */
class BicubicSrEngine : SrEngine {
    init {
        try {
            if (!org.opencv.android.OpenCVLoader.initDebug()) System.loadLibrary("opencv_java4")
        } catch (_: Throwable) {}
    }

    override val id: String = "bicubic"
    override val label: String = "OpenCV Bicubic Interpolation (CPU)"
    override val family: ModelFamily = ModelFamily.CLASSICAL
    override val unavailableReason: String? = null

    override fun runWithTelemetry(src: Bitmap, backend: Backend, scale: Int, progress: SrProgress?): SrExecutionResult {
        SrRun.passesFor(scale)
        val monitor = PowerMonitor.get().also { it.start() }
        val token = monitor.begin(ThermalZones.Group.CPU)
        val clock = PhaseClock()
        val srcMat = Mat()
        val dstMat = Mat()
        try {
            clock.prep { Utils.bitmapToMat(src, srcMat) }
            clock.infer { Imgproc.resize(srcMat, dstMat, Size((src.width * scale).toDouble(), (src.height * scale).toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC) }
            val out = clock.post {
                Bitmap.createBitmap(src.width * scale, src.height * scale, Bitmap.Config.ARGB_8888).also { Utils.matToBitmap(dstMat, it) }
            }
            return SrExecutionResult(out, monitor.end(token, clock, SrRun.coreLabel(Backend.CPU)))
        } finally {
            srcMat.release()
            dstMat.release()
        }
    }
}
