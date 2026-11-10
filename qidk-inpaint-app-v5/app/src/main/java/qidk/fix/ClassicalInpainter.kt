package qidk.fix

import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo

/** JNI bridge to libqidk_classical.so (sources in app/src/main/cpp/classical/). Package is fixed: the C symbol names depend on it. */
object ClassicalNative {
    init {
        try {
            if (!OpenCVLoader.initDebug()) {
                System.loadLibrary("opencv_java4")
            }
        } catch (_: Throwable) {}
        System.loadLibrary("qidk_classical")
    }

    /** rgba (w*h*4) and mask (w*h, non-zero = hole) are modified in place; mask keeps the pixels still unfilled.
     *  Returns that remaining count (0 = done) or a negative error: -1 bad input, -2 no source patch, -3 memory. */
    @JvmStatic external fun criminisi(rgba: ByteArray, mask: ByteArray, w: Int, h: Int,
                                      patchRadius: Int, searchRadius: Int, threads: Int, budgetMs: Int): Int

    /** rgba is modified in place. Returns 0 on success, -1 bad input, -2 hole too large, -3 memory / numerics. */
    @JvmStatic external fun biharmonic(rgba: ByteArray, mask: ByteArray, w: Int, h: Int, maxCells: Long): Int
}

enum class ClassicalAlgo(val label: String) {
    TELEA("Fast Marching (Telea)"),
    NAVIER_STOKES("Fluid Streamlines (Navier-Stokes)"),
    CRIMINISI("Criminisi Exemplar"),
    BIHARMONIC("Biharmonic Poisson");

    companion object {
        fun fromText(text: String): ClassicalAlgo? {
            val t = text.lowercase()
            return when {
                "telea" in t || "fast marching" in t -> TELEA
                "navier" in t || "fluid" in t || "streamline" in t || t == "ns" || "ns" in t -> NAVIER_STOKES
                "criminisi" in t || "exemplar" in t -> CRIMINISI
                "biharmonic" in t || "poisson" in t -> BIHARMONIC
                else -> null
            }
        }
    }
}

/**
 * Classical (non-neural) inpainting on the Oryon CPU.
 *  1. The user's mask is dilated by 5 px (ellipse kernel) first, for EVERY algorithm: the user's mask hugs the object,
 *     so anti-aliased object pixels just outside it would otherwise be diffused/copied into the hole.
 *  2. Telea / Navier-Stokes: OpenCV Photo.inpaint. Criminisi / Biharmonic: native kernels (cpp/classical/).
 *  3. If a native kernel cannot finish (time budget, hole too large) the rest is done by Telea / Navier-Stokes and
 *     [Output.note] says so - the UI must show it, nothing is silently swapped.
 */
object ClassicalInpainter {
    private const val TAG = "QIDK.Classic"
    const val MASK_DILATE_PX = 5
    const val TELEA_RADIUS = 3.0
    const val CRIMINISI_PATCH_RADIUS = 6
    const val CRIMINISI_SEARCH_RADIUS = 128
    const val CRIMINISI_BUDGET_MS = 20_000
    const val BIHARMONIC_MAX_CELLS = 30_000_000L

    class Output(val bitmap: Bitmap, val requested: ClassicalAlgo, val used: String, val note: String?, val ms: Long)

    /** Same rule the existing driver uses for "this mask pixel is a hole". */
    fun isHolePixel(c: Int): Boolean {
        if (Color.alpha(c) <= 50) return false
        val luma = (Color.red(c) * 299 + Color.green(c) * 587 + Color.blue(c) * 114) / 1000
        return luma > 128 || Color.red(c) > 128
    }

    fun holeBytes(mask: Bitmap): ByteArray {
        val w = mask.width; val h = mask.height
        val px = IntArray(w * h)
        mask.getPixels(px, 0, w, 0, 0, w, h)
        return ByteArray(w * h) { if (isHolePixel(px[it])) 1 else 0 }
    }

    /** [holeMask]: width*height bytes of the SAME size as [source], non-zero = hole (not yet dilated). */
    fun run(source: Bitmap, holeMask: ByteArray, algo: ClassicalAlgo): Output {
        val t0 = System.nanoTime()
        val argb = if (source.config == Bitmap.Config.ARGB_8888) source else source.copy(Bitmap.Config.ARGB_8888, false)
        val w = argb.width
        val h = argb.height
        require(holeMask.size == w * h) { "mask is ${holeMask.size} bytes, image is ${w}x$h" }
        val owned = ArrayList<Mat>()
        fun <T : Mat> T.own(): T { owned.add(this); return this }
        try {
            val rgba = Mat().own()
            Utils.bitmapToMat(argb, rgba)
            val m0 = Mat(h, w, CvType.CV_8UC1).own()
            m0.put(0, 0, holeMask)
            Imgproc.threshold(m0, m0, 0.0, 255.0, Imgproc.THRESH_BINARY)
            val k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size((2 * MASK_DILATE_PX + 1).toDouble(), (2 * MASK_DILATE_PX + 1).toDouble())).own()
            val hole = Mat().own()
            Imgproc.dilate(m0, hole, k)

            var used = algo.label
            var note: String? = null
            val result = Mat().own()

            fun opencvFill(src: Mat, mask: Mat, flag: Int): Mat {
                val rgb = Mat().own(); val dst = Mat().own(); val back = Mat().own()
                Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
                Photo.inpaint(rgb, mask, dst, TELEA_RADIUS, flag)
                Imgproc.cvtColor(dst, back, Imgproc.COLOR_RGB2RGBA)
                return back
            }

            when (algo) {
                ClassicalAlgo.TELEA -> opencvFill(rgba, hole, Photo.INPAINT_TELEA).copyTo(result)
                ClassicalAlgo.NAVIER_STOKES -> opencvFill(rgba, hole, Photo.INPAINT_NS).copyTo(result)
                ClassicalAlgo.CRIMINISI, ClassicalAlgo.BIHARMONIC -> {
                    val px = ByteArray(w * h * 4)
                    rgba.get(0, 0, px)
                    val mk = ByteArray(w * h)
                    hole.get(0, 0, mk)
                    val rc = if (algo == ClassicalAlgo.CRIMINISI)
                        ClassicalNative.criminisi(px, mk, w, h, CRIMINISI_PATCH_RADIUS, CRIMINISI_SEARCH_RADIUS,
                            Runtime.getRuntime().availableProcessors().coerceIn(1, 8), CRIMINISI_BUDGET_MS)
                    else
                        ClassicalNative.biharmonic(px, mk, w, h, BIHARMONIC_MAX_CELLS)
                    val cur = Mat(h, w, CvType.CV_8UC4).own()
                    cur.put(0, 0, px)
                    when {
                        rc == 0 -> cur.copyTo(result)
                        algo == ClassicalAlgo.CRIMINISI && rc > 0 -> {      // time budget hit: finish the rest with Telea
                            val rest = Mat(h, w, CvType.CV_8UC1).own()
                            rest.put(0, 0, mk)
                            Imgproc.threshold(rest, rest, 0.0, 255.0, Imgproc.THRESH_BINARY)
                            opencvFill(cur, rest, Photo.INPAINT_TELEA).copyTo(result)
                            used = "${algo.label} + Telea"; note = "time budget reached, $rc px finished with Telea"
                        }
                        else -> {                                           // native kernel refused / failed
                            val why = when (rc) {
                                -2 -> if (algo == ClassicalAlgo.BIHARMONIC) "hole too large for the exact biharmonic solver" else "no clean source patch"
                                -3 -> "out of memory"
                                else -> "invalid input"
                            }
                            opencvFill(rgba, hole, Photo.INPAINT_NS).copyTo(result)
                            used = "Navier-Stokes (fallback)"; note = "${algo.label}: $why (code $rc)"
                        }
                    }
                }
            }
            val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(result, out)
            val ms = (System.nanoTime() - t0) / 1_000_000
            Log.i(TAG, "requested=${algo.label} used=$used note=${note ?: "-"} holePx=${Core.countNonZero(hole)} ms=$ms")
            return Output(out, algo, used, note, ms)
        } finally {
            owned.forEach { it.release() }
        }
    }
}
