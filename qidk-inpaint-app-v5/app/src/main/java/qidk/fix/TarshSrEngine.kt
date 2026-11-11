package qidk.fix

import android.graphics.Bitmap

/**
 * Tarsh's hand-built 4-quadrant sub-pixel reconstructor (port of meow.py lines 170-220), numerically identical to the
 * previous version of this file but evaluated in horizontal bands: the old code allocated three full-size float planes
 * for the reconstruction plus a full-size IntArray, about 400 MB at the 2048 -> 4096 pass (out of memory on the default
 * Java heap). Now the peak is the source pixels plus a few hundred output rows.
 *
 *   1. every source pixel (x, y) yields four sub-pixels, each (4 * centre + 1 * diagonal + 2 * vertical + 2 * horizontal) / 9
 *      with the neighbours on the side of the sub-pixel (edge pixels replicate);
 *   2. the 2x plane is sharpened with (16 * c - 2 * cardinals - 0.25 * diagonals) / 7 and truncated to [0, 255].
 */
object TarshCore {
    const val BAND_ROWS = 128

    /** Reconstruction (before sharpening) of output row [r] for one channel; writes outW floats. */
    private fun reconRow(src: IntArray, w: Int, h: Int, shift: Int, r: Int, out: FloatArray, outOff: Int) {
        val y = r shr 1
        val dy = r and 1
        val yv = if (dy == 0) (if (y > 0) y - 1 else 0) else (if (y < h - 1) y + 1 else h - 1)
        val rowC = y * w
        val rowV = yv * w
        for (x in 0 until w) {
            val xn = if (x > 0) x - 1 else 0
            val xp = if (x < w - 1) x + 1 else w - 1
            val center = ((src[rowC + x] ushr shift) and 0xFF).toFloat()
            val vert = ((src[rowV + x] ushr shift) and 0xFF).toFloat()
            val left = ((src[rowC + xn] ushr shift) and 0xFF).toFloat()
            val right = ((src[rowC + xp] ushr shift) and 0xFF).toFloat()
            val dl = ((src[rowV + xn] ushr shift) and 0xFF).toFloat()
            val dr = ((src[rowV + xp] ushr shift) and 0xFF).toFloat()
            out[outOff + 2 * x] = (4f * center + dl + 2f * vert + 2f * left) / 9.0f
            out[outOff + 2 * x + 1] = (4f * center + dr + 2f * vert + 2f * right) / 9.0f
        }
    }

    /**
     * Renders output rows [y0, y1) of the 2x result into [out] (outW * (y1 - y0) ARGB ints).
     * [src] holds w * h ARGB ints.
     */
    fun renderRows(src: IntArray, w: Int, h: Int, y0: Int, y1: Int, out: IntArray) {
        val outW = w * 2
        val outH = h * 2
        require(y0 in 0..outH && y1 in y0..outH) { "rows $y0..$y1 outside 0..$outH" }
        val rows = y1 - y0
        if (rows == 0) return
        // reconstruction rows y0-1 .. y1 (inclusive) feed the sharpening of rows y0 .. y1-1
        val first = maxOf(0, y0 - 1)
        val last = minOf(outH - 1, y1)
        val n = last - first + 1
        val planes = Array(3) { FloatArray(n * outW) }
        val shifts = intArrayOf(16, 8, 0)
        for (c in 0 until 3) for (i in 0 until n) reconRow(src, w, h, shifts[c], first + i, planes[c], i * outW)

        for (y in y0 until y1) {
            val yTop = if (y > 0) y - 1 else 0
            val yBot = if (y < outH - 1) y + 1 else outH - 1
            val rc = (y - first) * outW
            val rt = (yTop - first) * outW
            val rb = (yBot - first) * outW
            for (x in 0 until outW) {
                val xl = if (x > 0) x - 1 else 0
                val xr = if (x < outW - 1) x + 1 else outW - 1
                var px = 0xFF shl 24
                for (c in 0 until 3) {
                    val p = planes[c]
                    val center = p[rc + x]
                    val cardinals = p[rt + x] + p[rb + x] + p[rc + xl] + p[rc + xr]
                    val diagonals = p[rt + xl] + p[rt + xr] + p[rb + xl] + p[rb + xr]
                    val v = ((16.0f * center - 2.0f * cardinals - 0.25f * diagonals) / 7.0f).toInt().coerceIn(0, 255)
                    px = px or (v shl shifts[c])
                }
                out[(y - y0) * outW + x] = px
            }
        }
    }
}

class TarshSrEngine : SrEngine {
    override val id: String = "tarsh"
    override val label: String = "Tarsh 4-Quadrant Sub-Pixel Reconstructor"
    override val family: ModelFamily = ModelFamily.SR_TARSH
    override val unavailableReason: String? = null

    override fun runWithTelemetry(src: Bitmap, backend: Backend, scale: Int, progress: SrProgress?): SrExecutionResult {
        BackendMatrix.require(family, backend)
        val passes = SrRun.passesFor(scale)
        val monitor = PowerMonitor.get().also { it.start() }
        val token = monitor.begin(ThermalZones.Group.CPU)
        val clock = PhaseClock()

        var current = src
        for (p in 1..passes) {
            if (progress?.cancelled == true) { if (current !== src) current.recycle(); throw SrCancelled() }
            val next = pass(current, p, passes, progress, clock)
            if (current !== src) current.recycle()
            current = next
        }
        return SrExecutionResult(current, monitor.end(token, clock, SrRun.coreLabel(Backend.CPU)))
    }

    private fun pass(src: Bitmap, p: Int, passes: Int, progress: SrProgress?, clock: PhaseClock): Bitmap {
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h)
        clock.prep { src.getPixels(pixels, 0, w, 0, 0, w, h) }
        val outW = w * 2
        val outH = h * 2
        val out = clock.prep { Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888) }
        val band = IntArray(outW * TarshCore.BAND_ROWS)
        var y = 0
        while (y < outH) {
            if (progress?.cancelled == true) { out.recycle(); throw SrCancelled() }
            progress?.report(p, passes, y, outH, "pass $p/$passes row $y/$outH")
            val y1 = minOf(outH, y + TarshCore.BAND_ROWS)
            clock.infer { TarshCore.renderRows(pixels, w, h, y, y1, band) }
            clock.post { out.setPixels(band, 0, outW, 0, y, outW, y1 - y) }
            y = y1
        }
        return out
    }
}
