package qidk.fix

import android.graphics.Bitmap

/**
 * JNI bindings of libqidk_sr.so (app/src/main/cpp/sr). The package is fixed on purpose: the exported symbols are
 * Java_qidk_fix_SrNative_*. Every bitmap handed to these calls must be a mutable ARGB_8888 bitmap (RGBA byte order in
 * memory, alpha 255); the kernels refuse anything else and return false / null.
 */
object SrNative {
    @Volatile private var state = 0 // 0 = not tried, 1 = loaded, -1 = failed

    val available: Boolean
        get() {
            if (state == 0) synchronized(this) {
                if (state == 0) state = try { System.loadLibrary("qidk_sr"); 1 } catch (_: Throwable) { -1 }
            }
            return state == 1
        }

    /** out (win*win*3 floats, RGB [0,1]) = 2x Keys-cubic base of [src] for the window [x0,x0+win) x [y0,y0+win) of the 2x grid. */
    @JvmStatic external fun sampleBase(src: Bitmap, x0: Int, y0: Int, win: Int, out: FloatArray): Boolean

    /** out (win*win) = 1 for the hole pixels of checkerboard [phase]; cell <= 0 = uniform (phase 0 everywhere). */
    @JvmStatic external fun holeMask(x0: Int, y0: Int, win: Int, cell: Int, phase: Int, out: FloatArray): Boolean

    /** Null-space detail transfer of one window; returns the 14 statistics of [SrStat] or null on bad arguments. */
    @JvmStatic external fun compose(
        u: FloatArray, o0: FloatArray?, o1: FloatArray?, win: Int, winX0: Int, winY0: Int, cell: Int, margin: Int,
        alpha: Float, tau: Float, driftLimit: Float, cx0: Int, cy0: Int, cw: Int, ch: Int, dst: Bitmap, dstX: Int, dstY: Int
    ): DoubleArray?

    /** Writes the classical base of the core rectangle into [dst]. */
    @JvmStatic external fun writeBase(u: FloatArray, win: Int, cx0: Int, cy0: Int, cw: Int, ch: Int, dst: Bitmap, dstX: Int, dstY: Int): Boolean

    /** PSNR (dB) between [src] and the scale x scale box average of [dst]; 99 = identical, negative = bad arguments. */
    @JvmStatic external fun lrPsnr(src: Bitmap, dst: Bitmap, scale: Int): Double
}

class BitmapImage(val bitmap: Bitmap) : SrImage {
    override val width: Int get() = bitmap.width
    override val height: Int get() = bitmap.height
}

/** [SrKernels] over the native library. */
class NativeKernels : SrKernels {
    private fun bmp(i: SrImage): Bitmap = (i as BitmapImage).bitmap

    override fun sampleBase(src: SrImage, x0: Int, y0: Int, win: Int, out: FloatArray) = SrNative.sampleBase(bmp(src), x0, y0, win, out)

    override fun holeMask(x0: Int, y0: Int, win: Int, cell: Int, phase: Int, out: FloatArray) = SrNative.holeMask(x0, y0, win, cell, phase, out)

    override fun compose(
        u: FloatArray, o0: FloatArray?, o1: FloatArray?, win: Int, winX0: Int, winY0: Int, cell: Int, margin: Int,
        alpha: Float, tau: Float, driftLimit: Float, cx0: Int, cy0: Int, cw: Int, ch: Int, dst: SrImage, dstX: Int, dstY: Int
    ): DoubleArray? = SrNative.compose(u, o0, o1, win, winX0, winY0, cell, margin, alpha, tau, driftLimit, cx0, cy0, cw, ch, bmp(dst), dstX, dstY)

    override fun writeBase(u: FloatArray, win: Int, cx0: Int, cy0: Int, cw: Int, ch: Int, dst: SrImage, dstX: Int, dstY: Int) =
        SrNative.writeBase(u, win, cx0, cy0, cw, ch, bmp(dst), dstX, dstY)
}
