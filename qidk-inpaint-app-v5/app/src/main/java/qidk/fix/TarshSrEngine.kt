package qidk.fix

import android.graphics.Bitmap
import android.graphics.Color

/**
 * Tarsh's Handbuilt 4-Quadrant Sub-Pixel Reconstructor.
 * Port of meow.py (lines 170-220):
 * 1. 4-quadrant edge-padded sub-kernel accumulation (weights 4, 2, 1 / 9.0).
 * 2. 2x interleaved assembly into 1024x1024 float array.
 * 3. 8-neighbor Laplacian sharpening: (16*center - 2*cardinals - 0.25*diagonals) / 7.0.
 * 4. Value clipping to [0, 255] uint8.
 */
class TarshSrEngine : SrEngine {
    override val id: String = "tarsh"
    override val label: String = "Tarsh 4-Quadrant Sub-Pixel Reconstructor"
    override val family: ModelFamily = ModelFamily.SR_TARSH
    override val unavailableReason: String? = null

    override fun run(src: Bitmap, backend: Backend): Bitmap = run(src, backend, 2)

    override fun run(src: Bitmap, backend: Backend, scale: Int): Bitmap {
        BackendMatrix.require(family, backend)
        require(scale in listOf(2, 4, 8)) { "Scale factor must be 2, 4, or 8 (got $scale)" }

        val passes = when (scale) {
            2 -> 1
            4 -> 2
            8 -> 3
            else -> 1
        }

        var current = src
        for (p in 1..passes) {
            val next = runSingle2xPass(current)
            if (current !== src) current.recycle()
            current = next
        }
        return current
    }

    private fun runSingle2xPass(src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height

        // 1. Extract input RGB channels into float planes
        val srcPixels = IntArray(w * h)
        src.getPixels(srcPixels, 0, w, 0, 0, w, h)

        val rPlane = FloatArray(w * h)
        val gPlane = FloatArray(w * h)
        val bPlane = FloatArray(w * h)

        for (i in srcPixels.indices) {
            val c = srcPixels[i]
            rPlane[i] = ((c ushr 16) and 0xFF).toFloat()
            gPlane[i] = ((c ushr 8) and 0xFF).toFloat()
            bPlane[i] = (c and 0xFF).toFloat()
        }

        // 2. Compute 4-Quadrant Sub-Pixel Assembly for each channel
        val outW = w * 2
        val outH = h * 2

        val rRecon = FloatArray(outW * outH)
        val gRecon = FloatArray(outW * outH)
        val bRecon = FloatArray(outW * outH)

        processChannelRecon(rPlane, w, h, rRecon)
        processChannelRecon(gPlane, w, h, gRecon)
        processChannelRecon(bPlane, w, h, bRecon)

        // 3. 8-Neighbor Kernel Sharpening and pack into result Bitmap
        val outPixels = IntArray(outW * outH)
        for (y in 0 until outH) {
            val yTop = if (y > 0) y - 1 else 0
            val yBot = if (y < outH - 1) y + 1 else outH - 1

            for (x in 0 until outW) {
                val xLeft = if (x > 0) x - 1 else 0
                val xRight = if (x < outW - 1) x + 1 else outW - 1

                val rSharp = sharpenPixel(rRecon, outW, x, y, xLeft, xRight, yTop, yBot)
                val gSharp = sharpenPixel(gRecon, outW, x, y, xLeft, xRight, yTop, yBot)
                val bSharp = sharpenPixel(bRecon, outW, x, y, xLeft, xRight, yTop, yBot)

                outPixels[y * outW + x] = (0xFF shl 24) or (rSharp shl 16) or (gSharp shl 8) or bSharp
            }
        }

        val resultBmp = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        resultBmp.setPixels(outPixels, 0, outW, 0, 0, outW, outH)
        return resultBmp
    }

    private fun processChannelRecon(arr: FloatArray, w: Int, h: Int, outRecon: FloatArray) {
        val outW = w * 2

        for (y in 0 until h) {
            val yPrev = if (y > 0) y - 1 else 0
            val yNext = if (y < h - 1) y + 1 else h - 1

            val rowCurr = y * w
            val rowPrev = yPrev * w
            val rowNext = yNext * w

            for (x in 0 until w) {
                val xPrev = if (x > 0) x - 1 else 0
                val xNext = if (x < w - 1) x + 1 else w - 1

                val center = arr[rowCurr + x]
                val top = arr[rowPrev + x]
                val bottom = arr[rowNext + x]
                val left = arr[rowCurr + xPrev]
                val right = arr[rowCurr + xNext]
                val topLeft = arr[rowPrev + xPrev]
                val topRight = arr[rowPrev + xNext]
                val bottomLeft = arr[rowNext + xPrev]
                val bottomRight = arr[rowNext + xNext]

                val subTL = (4f * center + topLeft + 2f * top + 2f * left) / 9.0f
                val subTR = (4f * center + topRight + 2f * top + 2f * right) / 9.0f
                val subBL = (4f * center + bottomLeft + 2f * bottom + 2f * left) / 9.0f
                val subBR = (4f * center + bottomRight + 2f * bottom + 2f * right) / 9.0f

                val outY = y * 2
                val outX = x * 2

                outRecon[outY * outW + outX] = subTL
                outRecon[outY * outW + (outX + 1)] = subTR
                outRecon[(outY + 1) * outW + outX] = subBL
                outRecon[(outY + 1) * outW + (outX + 1)] = subBR
            }
        }
    }

    private fun sharpenPixel(
        plane: FloatArray, w: Int,
        x: Int, y: Int,
        xLeft: Int, xRight: Int,
        yTop: Int, yBot: Int
    ): Int {
        val center = plane[y * w + x]
        val top = plane[yTop * w + x]
        val bottom = plane[yBot * w + x]
        val left = plane[y * w + xLeft]
        val right = plane[y * w + xRight]

        val topLeft = plane[yTop * w + xLeft]
        val topRight = plane[yTop * w + xRight]
        val bottomLeft = plane[yBot * w + xLeft]
        val bottomRight = plane[yBot * w + xRight]

        val enhanced = (16.0f * center - 2.0f * (top + bottom + left + right) - 0.25f * (topLeft + topRight + bottomLeft + bottomRight)) / 7.0f
        return enhanced.toInt().coerceIn(0, 255)
    }
}
