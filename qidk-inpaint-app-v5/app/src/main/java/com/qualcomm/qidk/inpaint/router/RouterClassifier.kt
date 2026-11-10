package com.qualcomm.qidk.inpaint.router

import android.graphics.Bitmap
import android.graphics.Color
import android.media.FaceDetector
import kotlin.math.abs
import kotlin.math.sqrt

data class RoutingDecision(
    val recommendedModel: String,
    val ruleTriggered: String,
    val justification: String,
    val facesDetected: Int,
    val edgeDensity: Float,
    val laplacianVariance: Float,
    val maskAreaRatio: Float,
    val maskLocation: String,
    val latencyMs: Float,
    val energyJoules: Float,
    val thermalRiseC: Float
)

object RouterClassifier {

    fun classifyAndRoute(image: Bitmap, mask: Bitmap): RoutingDecision {
        val w = 512
        val h = 512
        val imgScaled = if (image.width == w && image.height == h) image else Bitmap.createScaledBitmap(image, w, h, true)
        val maskScaled = if (mask.width == w && mask.height == h) mask else Bitmap.createScaledBitmap(mask, w, h, false)

        // 1. Feature Extraction
        val faces = detectFaces(imgScaled)
        val lapVar = computeLaplacianVariance(imgScaled)
        val edgeDensity = computeEdgeDensity(imgScaled)
        val maskStats = analyzeMask(maskScaled)

        val areaRatio = maskStats.areaRatio
        val fragments = maskStats.fragments

        // 2. Heuristic Decision Tree Evaluation
        val (model, rule, justification, latency, energy, dt) = when {
            // Rule 1: Facial Structure Priority
            faces > 0 && areaRatio <= 0.25f -> {
                Tuple6(
                    "MIGAN",
                    "RULE_FACE_PORTRAIT",
                    "High facial geometry ($faces detected). Routed to MIGAN (facial optimization, 216ms, 0.62J).",
                    216.0f, 0.62f, 9.6f
                )
            }
            // Rule 2: Large Void / High Fragmentation
            areaRatio > 0.25f || fragments >= 3 || (maskStats.location == "edge_or_corner" && areaRatio > 0.15f) -> {
                Tuple6(
                    "LAMA",
                    "RULE_LARGE_VOID_FFC",
                    "Large corruption void (${String.format("%.1f", areaRatio * 100)}% > 25%). Routed to LaMa Dilated (global FFT receptive field).",
                    321.0f, 0.99f, 24.6f
                )
            }
            // Rule 3: High Texture / Sharp Geometric Edges
            edgeDensity > 0.08f || lapVar > 500f -> {
                Tuple6(
                    "AOTGAN",
                    "RULE_HIGH_TEXTURE_AOT",
                    "Dense texture / edges (density: ${String.format("%.3f", edgeDensity)}, var: ${String.format("%.1f", lapVar)}). Routed to AOT-GAN.",
                    390.0f, 1.30f, 25.4f
                )
            }
            // Rule 4: Baseline / Smooth Surface Default
            else -> {
                Tuple6(
                    "MIGAN",
                    "RULE_SMOOTH_DEFAULT",
                    "Smooth texture baseline (${String.format("%.1f", areaRatio * 100)}% mask). Routed to MIGAN (ultra-low 0.62J power default).",
                    216.0f, 0.62f, 9.6f
                )
            }
        }

        return RoutingDecision(
            recommendedModel = model,
            ruleTriggered = rule,
            justification = justification,
            facesDetected = faces,
            edgeDensity = edgeDensity,
            laplacianVariance = lapVar,
            maskAreaRatio = areaRatio,
            maskLocation = maskStats.location,
            latencyMs = latency,
            energyJoules = energy,
            thermalRiseC = dt
        )
    }

    private data class Tuple6<A, B, C, D, E, F>(
        val a: A, val b: B, val c: C, val d: D, val e: E, val f: F
    )

    private data class MaskStats(
        val areaRatio: Float,
        val location: String,
        val fragments: Int
    )

    private fun detectFaces(bitmap: Bitmap): Int {
        return try {
            val rgb565 = bitmap.copy(Bitmap.Config.RGB_565, false)
            val detector = FaceDetector(rgb565.width, rgb565.height, 5)
            val faces = arrayOfNulls<FaceDetector.Face>(5)
            detector.findFaces(rgb565, faces)
        } catch (e: Exception) {
            0
        }
    }

    private fun computeLaplacianVariance(bitmap: Bitmap): Float {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        // Convert to luminance
        val gray = FloatArray(w * h)
        for (i in pixels.indices) {
            val c = pixels[i]
            gray[i] = 0.299f * Color.red(c) + 0.587f * Color.green(c) + 0.114f * Color.blue(c)
        }

        // Apply 3x3 discrete Laplacian
        var sum = 0.0
        var sumSq = 0.0
        var count = 0

        for (y in 1 until h - 1) {
            val rowOffset = y * w
            for (x in 1 until w - 1) {
                val idx = rowOffset + x
                val lap = gray[idx - w] + gray[idx + w] + gray[idx - 1] + gray[idx + 1] - 4f * gray[idx]
                sum += lap
                sumSq += (lap * lap)
                count++
            }
        }

        if (count == 0) return 0f
        val mean = sum / count
        val variance = (sumSq / count) - (mean * mean)
        return maxOf(0f, variance.toFloat())
    }

    private fun computeEdgeDensity(bitmap: Bitmap): Float {
        val w = bitmap.width
        val h = bitmap.height
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        val gray = FloatArray(w * h)
        for (i in pixels.indices) {
            val c = pixels[i]
            gray[i] = 0.299f * Color.red(c) + 0.587f * Color.green(c) + 0.114f * Color.blue(c)
        }

        var edgeCount = 0
        var total = 0

        for (y in 1 until h - 1) {
            val rowOffset = y * w
            for (x in 1 until w - 1) {
                val idx = rowOffset + x
                // Sobel kernels
                val gx = (-1f * gray[idx - w - 1] + 1f * gray[idx - w + 1]
                        - 2f * gray[idx - 1] + 2f * gray[idx + 1]
                        - 1f * gray[idx + w - 1] + 1f * gray[idx + w + 1])
                val gy = (1f * gray[idx - w - 1] + 2f * gray[idx - w] + 1f * gray[idx - w + 1]
                        - 1f * gray[idx + w - 1] - 2f * gray[idx + w] - 1f * gray[idx + w + 1])

                val mag = abs(gx) + abs(gy)
                if (mag > 120f) {
                    edgeCount++
                }
                total++
            }
        }

        return if (total > 0) edgeCount.toFloat() / total.toFloat() else 0f
    }

    private fun analyzeMask(mask: Bitmap): MaskStats {
        val w = mask.width
        val h = mask.height
        val pixels = IntArray(w * h)
        mask.getPixels(pixels, 0, w, 0, 0, w, h)

        var holePixels = 0
        var minX = w
        var maxX = 0
        var minY = h
        var maxY = 0

        for (y in 0 until h) {
            val rowOffset = y * w
            for (x in 0 until w) {
                val c = pixels[rowOffset + x]
                val alpha = Color.alpha(c)
                val red = Color.red(c)
                // Hole requires positive mask alpha AND luminance/red > 128
                if (alpha > 50 && red > 128) {
                    holePixels++
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        val total = (w * h).toFloat()
        val areaRatio = holePixels.toFloat() / total

        val location = if (holePixels == 0) {
            "none"
        } else {
            val cx = (minX + maxX) / 2f
            val cy = (minY + maxY) / 2f
            if (cx in (0.33f * w)..(0.66f * w) && cy in (0.33f * h)..(0.66f * h)) {
                "center"
            } else {
                "edge_or_corner"
            }
        }

        val fragments = if (holePixels > 0) 1 else 0
        return MaskStats(areaRatio, location, fragments)
    }
}
