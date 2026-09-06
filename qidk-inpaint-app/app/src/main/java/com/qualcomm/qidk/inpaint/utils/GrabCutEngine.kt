package com.qualcomm.qidk.inpaint.utils

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

object GrabCutEngine {

    /**
     * Fast CPU Target Auto-Masking (<50ms).
     * Extracts target silhouette from user selection box.
     */
    fun generateBoundingBoxMask(source: Bitmap, bbox: RectF): Pair<Bitmap, Long> {
        val t0 = System.currentTimeMillis()
        val w = 512
        val h = 512
        val maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        maskBmp.eraseColor(Color.TRANSPARENT)

        val left = max(0, bbox.left.toInt())
        val top = max(0, bbox.top.toInt())
        val right = min(w - 1, bbox.right.toInt())
        val bottom = min(h - 1, bbox.bottom.toInt())

        if (right <= left || bottom <= top) {
            return Pair(maskBmp, 0L)
        }

        val boxW = right - left
        val boxH = bottom - top

        // Sample background border around bbox to build reference distribution
        val srcPixels = IntArray(w * h)
        source.getPixels(srcPixels, 0, w, 0, 0, w, h)

        var bgR = 0L
        var bgG = 0L
        var bgB = 0L
        var bgCount = 0

        // Sample 8-pixel border outside the box
        val margin = 8
        val minX = max(0, left - margin)
        val maxX = min(w - 1, right + margin)
        val minY = max(0, top - margin)
        val maxY = min(h - 1, bottom + margin)

        for (y in minY..maxY) {
            for (x in minX..maxX) {
                if (x < left || x > right || y < top || y > bottom) {
                    val c = srcPixels[y * w + x]
                    bgR += Color.red(c)
                    bgG += Color.green(c)
                    bgB += Color.blue(c)
                    bgCount++
                }
            }
        }

        val meanBgR = if (bgCount > 0) (bgR / bgCount).toFloat() else 128f
        val meanBgG = if (bgCount > 0) (bgG / bgCount).toFloat() else 128f
        val meanBgB = if (bgCount > 0) (bgB / bgCount).toFloat() else 128f

        // Foreground silhouette: pixels inside bbox that differ significantly from background mean
        val maskPixels = IntArray(w * h)
        var fgCount = 0

        for (y in top..bottom) {
            val row = y * w
            for (x in left..right) {
                val c = srcPixels[row + x]
                val dr = Color.red(c) - meanBgR
                val dg = Color.green(c) - meanBgG
                val db = Color.blue(c) - meanBgB
                val dist = dr * dr + dg * dg + db * db

                // Significant color difference threshold
                if (dist > 900f) {
                    maskPixels[row + x] = Color.WHITE
                    fgCount++
                }
            }
        }

        // Fallback to solid box if contrast was insufficient
        if (fgCount < (boxW * boxH * 0.05f)) {
            for (y in top..bottom) {
                val row = y * w
                for (x in left..right) {
                    maskPixels[row + x] = Color.WHITE
                }
            }
        }

        maskBmp.setPixels(maskPixels, 0, w, 0, 0, w, h)
        val latency = System.currentTimeMillis() - t0
        return Pair(maskBmp, latency)
    }
}
