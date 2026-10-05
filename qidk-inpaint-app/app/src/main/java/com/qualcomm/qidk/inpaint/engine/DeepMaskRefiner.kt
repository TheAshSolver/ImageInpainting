package com.qualcomm.qidk.inpaint.engine

import android.graphics.Bitmap

/**
 * DeepMaskRefiner:
 * Multi-Modal Deep Mask Refinement Engine (Candidate 3 - Overhauled MediaPipe Pipeline).
 *
 * Replaces the legacy hybrid engine (Centroid Collapse + GrabCut Fallback) with an
 * ultra-low latency, edge-aligned neural refinement pipeline:
 * - Multi-Point Medial Axis Sampling (via 2-pass Chamfer Distance Transform)
 * - Multi-Seed MediaPipe Interactive Segmentation & Confidence Fusion
 * - O(N) Fast Guided Filter (r=4, eps=1e-2) snapping to 512x512 photo color edges
 * - 2-pixel Circular Safety Dilation
 * - Deterministic sub-millisecond fallback (eliminating the 278-395ms GrabCut stall)
 *
 * Latency budget: ~19ms P50 / ~26ms P90 on Snapdragon 8 Elite.
 */
object DeepMaskRefiner {
    private const val TAG = "DeepMaskRefiner"

    /**
     * Refines a rough user mask (brush or box) using the Overhauled MediaPipe Pipeline.
     *
     * @param source High-resolution photo bitmap
     * @param roughMask Rough user drawn mask or bounding box
     * @param isBoxMode Whether the mask originated from bounding box mode
     * @return Pair containing the refined 512x512 binary mask and execution latency in ms
     */
    fun refineMask(
        source: Bitmap,
        roughMask: Bitmap,
        isBoxMode: Boolean = false
    ): Pair<Bitmap, Long> {
        val t0 = System.currentTimeMillis()
        val w = 512
        val h = 512

        val srcBmp = if (source.width == w && source.height == h) source else Bitmap.createScaledBitmap(source, w, h, true)
        val maskBmp = if (roughMask.width == w && roughMask.height == h) roughMask else Bitmap.createScaledBitmap(roughMask, w, h, false)

        val (refinedMask, neuralLatency) = MediaPipeSegmenter.refineMaskOverhaul(srcBmp, maskBmp, isBoxMode)
        val totalLatency = System.currentTimeMillis() - t0

        android.util.Log.i(TAG, "DeepMaskRefiner completed overhaul refinement in ${totalLatency}ms (neural+filter: ${neuralLatency}ms)")
        return Pair(refinedMask, totalLatency)
    }
}
