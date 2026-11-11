package qidk.fix

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import java.util.Random

class TarshCoreTest {

    /** The previous implementation, transcribed verbatim (full-size planes), as the oracle. */
    private fun reference(src: IntArray, w: Int, h: Int): IntArray {
        fun plane(shift: Int) = FloatArray(w * h) { ((src[it] ushr shift) and 0xFF).toFloat() }
        val outW = w * 2; val outH = h * 2
        fun recon(arr: FloatArray): FloatArray {
            val o = FloatArray(outW * outH)
            for (y in 0 until h) {
                val yPrev = if (y > 0) y - 1 else 0; val yNext = if (y < h - 1) y + 1 else h - 1
                for (x in 0 until w) {
                    val xPrev = if (x > 0) x - 1 else 0; val xNext = if (x < w - 1) x + 1 else w - 1
                    val center = arr[y * w + x]; val top = arr[yPrev * w + x]; val bottom = arr[yNext * w + x]
                    val left = arr[y * w + xPrev]; val right = arr[y * w + xNext]
                    val tl = arr[yPrev * w + xPrev]; val tr = arr[yPrev * w + xNext]
                    val bl = arr[yNext * w + xPrev]; val br = arr[yNext * w + xNext]
                    o[(2 * y) * outW + 2 * x] = (4f * center + tl + 2f * top + 2f * left) / 9.0f
                    o[(2 * y) * outW + 2 * x + 1] = (4f * center + tr + 2f * top + 2f * right) / 9.0f
                    o[(2 * y + 1) * outW + 2 * x] = (4f * center + bl + 2f * bottom + 2f * left) / 9.0f
                    o[(2 * y + 1) * outW + 2 * x + 1] = (4f * center + br + 2f * bottom + 2f * right) / 9.0f
                }
            }
            return o
        }
        val rr = recon(plane(16)); val gg = recon(plane(8)); val bb = recon(plane(0))
        fun sharp(p: FloatArray, x: Int, y: Int): Int {
            val yt = if (y > 0) y - 1 else 0; val yb = if (y < outH - 1) y + 1 else outH - 1
            val xl = if (x > 0) x - 1 else 0; val xr = if (x < outW - 1) x + 1 else outW - 1
            val c = p[y * outW + x]
            val e = (16.0f * c - 2.0f * (p[yt * outW + x] + p[yb * outW + x] + p[y * outW + xl] + p[y * outW + xr]) -
                0.25f * (p[yt * outW + xl] + p[yt * outW + xr] + p[yb * outW + xl] + p[yb * outW + xr])) / 7.0f
            return e.toInt().coerceIn(0, 255)
        }
        return IntArray(outW * outH) { i ->
            val x = i % outW; val y = i / outW
            (0xFF shl 24) or (sharp(rr, x, y) shl 16) or (sharp(gg, x, y) shl 8) or sharp(bb, x, y)
        }
    }

    private fun randomImage(w: Int, h: Int, seed: Long): IntArray {
        val rnd = Random(seed)
        return IntArray(w * h) { (0xFF shl 24) or rnd.nextInt(0x1000000) }
    }

    @Test fun bandedRenderingIsBitIdenticalToTheOriginalAlgorithm() {
        for ((w, h) in listOf(1 to 1, 2 to 3, 17 to 11, 64 to 70, 130 to 5)) {
            val src = randomImage(w, h, (w * 31 + h).toLong())
            val oracle = reference(src, w, h)
            val outW = w * 2; val outH = h * 2
            // whole image in one call
            val whole = IntArray(outW * outH)
            TarshCore.renderRows(src, w, h, 0, outH, whole)
            assertArrayEquals("whole ${w}x$h", oracle, whole)
            // in awkward bands of 7 rows
            val banded = IntArray(outW * outH)
            var y = 0
            while (y < outH) {
                val y1 = minOf(outH, y + 7)
                val band = IntArray(outW * (y1 - y))
                TarshCore.renderRows(src, w, h, y, y1, band)
                System.arraycopy(band, 0, banded, y * outW, band.size)
                y = y1
            }
            assertArrayEquals("banded ${w}x$h", oracle, banded)
        }
    }

    @Test fun flatImageStaysFlat() {
        val src = IntArray(20 * 20) { 0xFF336699.toInt() }
        val out = IntArray(40 * 40)
        TarshCore.renderRows(src, 20, 20, 0, 40, out)
        assertEquals(0xFF336699.toInt(), out[0])
        assertEquals(0xFF336699.toInt(), out[40 * 17 + 23])
    }
}
