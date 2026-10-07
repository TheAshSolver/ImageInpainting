package com.qualcomm.qidk.inpaint.utils

import org.junit.Assert.*
import org.junit.Test

class GrabCutEngineTest {

    private fun rgb(r: Int, g: Int, b: Int): Int {
        return (0xFF shl 24) or ((r and 0xFF) shl 16) or ((g and 0xFF) shl 8) or (b and 0xFF)
    }

    @Test
    fun testSmartObjectExtractionFromSquareBox() {
        val w = 512
        val h = 512
        val src = IntArray(w * h)
        val mask = IntArray(w * h)

        // Background: Beige skin tone (215, 180, 150)
        val bgCol = rgb(215, 180, 150)
        // Foreground object: Dark sunglasses (25, 25, 30) centered at (256, 256) with radius 40
        val fgCol = rgb(25, 25, 30)

        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val dx = x - 256
                val dy = y - 256
                if (dx * dx + dy * dy <= 40 * 40) {
                    src[row + x] = fgCol
                } else {
                    src[row + x] = bgCol
                }
            }
        }

        // Square mask around object: 190 to 320 (130x130 square enclosing the 80px circle)
        val whiteMask = rgb(255, 255, 255)
        var squareMaskCount = 0
        for (y in 190..320) {
            val row = y * w
            for (x in 190..320) {
                mask[row + x] = whiteMask
                squareMaskCount++
            }
        }

        val t0 = System.currentTimeMillis()
        val extracted = GrabCutEngine.extractSmartObjectFromPixels(w, h, src, mask)
        val latency = System.currentTimeMillis() - t0

        println("Smart extraction latency on 512x512: ${latency}ms")

        // Count extracted pixels
        var extractedCount = 0
        for (i in extracted.indices) {
            if (extracted[i] != 0) {
                extractedCount++
            }
        }

        println("Square mask size: $squareMaskCount px")
        println("Extracted object size: $extractedCount px (ideal ~5026 px for radius 40 circle + 2px dilation)")

        // The extracted object must be substantially smaller than the square (tight fit to circle)
        assertTrue("Extracted count should be smaller than box mask", extractedCount < squareMaskCount)
        // The extracted object should accurately capture the circle (~5000-6000 px)
        assertTrue("Extracted count should capture circle silhouette", extractedCount in 3500..8500)
    }

    @Test
    fun testEmptyMaskReturnsEmpty() {
        val w = 512
        val h = 512
        val src = IntArray(w * h)
        val mask = IntArray(w * h)

        val extracted = GrabCutEngine.extractSmartObjectFromPixels(w, h, src, mask)
        var count = 0
        for (p in extracted) {
            if (p != 0) count++
        }
        assertEquals("Empty mask should return 0 pixels", 0, count)
    }
}
