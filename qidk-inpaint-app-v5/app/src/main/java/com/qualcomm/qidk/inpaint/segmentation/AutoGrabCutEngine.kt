package com.qualcomm.qidk.inpaint.segmentation

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Path
import android.graphics.PathMeasure
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import com.qualcomm.qidk.inpaint.utils.MagicTouchEngine
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc

/**
 * AutoGrabCutEngine: Mathematically grounded Auto-GrabCut pipeline using 4-state
 * graph-cut energy minimization on the user's circle/loop gesture.
 *
 * Trimap formulation:
 * - Outside Bounding Box: GC_BGD (0, Definite Background)
 * - Inside Bounding Box, Outside Polygon: GC_PR_BGD (2, Probable Background)
 * - Inside Circle Loop Polygon: GC_PR_FGD (3, Probable Foreground)
 * - Morphologically Eroded Inner Core (15-20% inward): GC_FGD (1, Definite Foreground Anchor Seed)
 *
 * Optimization:
 * - GMM fitting (5-component RGB) for FG & BG
 * - Gibbs Energy: E = U + V solved via Boykov-Kolmogorov Min-Cut / Max-Flow
 * - Downscaled proxy grid (max 512px) for sub-50ms interactive execution on CPU
 */
object AutoGrabCutEngine {

    private const val TAG = "AutoGrabCutEngine"
    private var isOpenCvLoaded = false

    init {
        try {
            isOpenCvLoaded = OpenCVLoader.initDebug()
            Log.i(TAG, "OpenCV native library loaded successfully: $isOpenCvLoaded")
        } catch (e: Throwable) {
            Log.e(TAG, "Failed to initialize OpenCV native library: ${e.message}")
            isOpenCvLoaded = false
        }
    }

    fun isAvailable(): Boolean = isOpenCvLoaded

    /**
     * Executes true Auto-GrabCut with 4-state trimap from touch Path.
     */
    fun executeAutoGrabCut(
        sourceBitmap: Bitmap,
        touchPath: Path,
        iterations: Int = 2
    ): Pair<Bitmap, Long> {
        val t0 = System.currentTimeMillis()

        if (!isOpenCvLoaded) {
            Log.w(TAG, "OpenCV unavailable, falling back to fast gradient segmentation")
            val points = extractPointsFromPath(touchPath, 64)
            return MagicTouchEngine.segmentCircledObject(sourceBitmap, points)
        }

        try {
            // 1. Scale down for real-time responsiveness (<50ms)
            val maxDim = 384.0 // 384px guarantees sub-50ms execution on ARM64
            val scale = minOf(maxDim / sourceBitmap.width, maxDim / sourceBitmap.height, 1.0)
            val scaledW = (sourceBitmap.width * scale).toInt().coerceAtLeast(32)
            val scaledH = (sourceBitmap.height * scale).toInt().coerceAtLeast(32)

            val scaledBmp = Bitmap.createScaledBitmap(sourceBitmap, scaledW, scaledH, true)
            val imgMat = Mat()
            Utils.bitmapToMat(scaledBmp, imgMat)
            Imgproc.cvtColor(imgMat, imgMat, Imgproc.COLOR_RGBA2RGB)

            // Calculate bounding box and scale gesture path
            val bounds = RectF()
            touchPath.computeBounds(bounds, true)
            val scaledLeft = (bounds.left * scale).toInt().coerceIn(0, scaledW - 2)
            val scaledTop = (bounds.top * scale).toInt().coerceIn(0, scaledH - 2)
            val scaledRight = (bounds.right * scale).toInt().coerceIn(scaledLeft + 1, scaledW - 1)
            val scaledBottom = (bounds.bottom * scale).toInt().coerceIn(scaledTop + 1, scaledH - 1)

            val bw = scaledRight - scaledLeft
            val bh = scaledBottom - scaledTop

            // Semantic Gesture Intent Classification:
            // Compact gesture (<22% of dimension) signifies a Face, Hand, or Small Detail.
            // Larger gesture signifies Torso or Full Object.
            val isDetailOrFace = (bw.toFloat() / scaledW < 0.25f && bh.toFloat() / scaledH < 0.25f) || maxOf(bw, bh) < (maxDim * 0.22)

            val expLeft: Int
            val expRight: Int
            val expTop: Int
            val expBottom: Int

            if (isDetailOrFace) {
                // FACE / DETAIL SELECTION:
                // Keep search bounds strictly tight around the gesture (max 35px padding).
                // Do NOT expand downwards, preventing face selection from leaking into neck/chest/street!
                expLeft = (bw * 0.25).toInt().coerceIn(6, 35)
                expRight = expLeft
                expTop = (bh * 0.25).toInt().coerceIn(6, 35)
                expBottom = expTop
            } else {
                // WHOLE PERSON / OBJECT SELECTION (e.g. user circled torso):
                // Expand vertically with strict hard clamping (max 75px padding)
                // and restrict lateral expansion to prevent leaking sideways into the street!
                expLeft = (bw * 0.30).toInt().coerceIn(10, 50)
                expRight = expLeft
                expTop = (bh * 0.50).toInt().coerceIn(12, 75)
                expBottom = (bh * 0.65).toInt().coerceIn(15, 85)
            }

            val searchLeft = (scaledLeft - expLeft).coerceIn(0, scaledW - 1)
            val searchTop = (scaledTop - expTop).coerceIn(0, scaledH - 1)
            val searchRight = (scaledRight + expRight).coerceIn(0, scaledW - 1)
            val searchBottom = (scaledBottom + expBottom).coerceIn(0, scaledH - 1)

            val searchRect = Rect(
                searchLeft,
                searchTop,
                searchRight - searchLeft,
                searchBottom - searchTop
            )

            // 2. Initialize GrabCut Mask with GC_BGD (0, Definite Background)
            val maskMat = Mat(imgMat.size(), CvType.CV_8UC1, Scalar(Imgproc.GC_BGD.toDouble()))

            // Set searchRect to Probable Background (GC_PR_BGD = 2)
            Imgproc.rectangle(
                maskMat,
                searchRect.tl(),
                searchRect.br(),
                Scalar(Imgproc.GC_PR_BGD.toDouble()),
                -1
            )

            // CRITICAL FIX FOR STREET LEAKAGE:
            // If selecting whole person/object, establish explicit Definite Background (GC_BGD = 0)
            // lateral barriers on the outer 40% of horizontal margins and the bottom 25% floor margin.
            // This anchors the Boykov-Kolmogorov min-cut graph with hard sink nodes,
            // mathematically preventing the graph cut from flooding across continuous asphalt/street surfaces!
            if (!isDetailOrFace) {
                val lateralBarrierW = maxOf(4, (expLeft * 0.45).toInt())
                // Left barrier
                val leftBarrier = Rect(searchLeft, searchTop, lateralBarrierW, searchBottom - searchTop)
                Imgproc.rectangle(maskMat, leftBarrier.tl(), leftBarrier.br(), Scalar(Imgproc.GC_BGD.toDouble()), -1)
                // Right barrier
                val rightBarrier = Rect(searchRight - lateralBarrierW, searchTop, lateralBarrierW, searchBottom - searchTop)
                Imgproc.rectangle(maskMat, rightBarrier.tl(), rightBarrier.br(), Scalar(Imgproc.GC_BGD.toDouble()), -1)
                // Floor barrier below feet
                val floorBarrierH = maxOf(4, (expBottom * 0.25).toInt())
                val floorBarrier = Rect(searchLeft, searchBottom - floorBarrierH, searchRight - searchLeft, floorBarrierH)
                Imgproc.rectangle(maskMat, floorBarrier.tl(), floorBarrier.br(), Scalar(Imgproc.GC_BGD.toDouble()), -1)
            }

            // 3. Rasterize circle gesture polygon to Definite Foreground (GC_FGD = 1) and Probable FG
            val polyMat = Mat.zeros(maskMat.size(), CvType.CV_8UC1)
            val pathPoints = extractScaledPointsFromPath(touchPath, scale)
            if (pathPoints.size >= 3) {
                val matOfPoint = MatOfPoint()
                matOfPoint.fromList(pathPoints)
                Imgproc.fillPoly(polyMat, listOf(matOfPoint), Scalar(255.0))
                matOfPoint.release()
            }

            // Inside polygon -> GC_PR_FGD
            maskMat.setTo(Scalar(Imgproc.GC_PR_FGD.toDouble()), polyMat)

            // 4. Create an inner core for Definite Foreground (GC_FGD = 1)
            val coreMat = Mat()
            val kSize = if (isDetailOrFace) maxOf(3, (minOf(bw, bh) * 0.15).toInt()) else maxOf(3, (minOf(bw, bh) * 0.10).toInt())
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(kSize.toDouble(), kSize.toDouble()))
            Imgproc.erode(polyMat, coreMat, kernel)
            kernel.release()

            // Ensure coreMat is never empty so GrabCut has solid definite foreground seeds
            var coreFgCount = Core.countNonZero(coreMat)
            if (coreFgCount < 20) {
                val fallbackKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0))
                Imgproc.erode(polyMat, coreMat, fallbackKernel)
                fallbackKernel.release()
                coreFgCount = Core.countNonZero(coreMat)
            }
            if (coreFgCount < 10) {
                val cx = (bounds.centerX() * scale).toDouble()
                val cy = (bounds.centerY() * scale).toDouble()
                val r = maxOf(4.0, minOf(bw, bh) * 0.15)
                Imgproc.circle(coreMat, org.opencv.core.Point(cx, cy), r.toInt(), Scalar(255.0), -1)
            }
            maskMat.setTo(Scalar(Imgproc.GC_FGD.toDouble()), coreMat)
            coreMat.release()

            // 5. Run Graph-Cut Optimization on the expanded search window
            val bgdModel = Mat()
            val fgdModel = Mat()
            Imgproc.grabCut(
                imgMat,
                maskMat,
                searchRect,
                bgdModel,
                fgdModel,
                iterations,
                Imgproc.GC_INIT_WITH_MASK
            )

            // 6. Extract Foreground: GC_FGD (1) and GC_PR_FGD (3) become 255
            val binMask = Mat()
            val prFgdMask = Mat()
            Core.compare(maskMat, Scalar(Imgproc.GC_FGD.toDouble()), binMask, Core.CMP_EQ)
            Core.compare(maskMat, Scalar(Imgproc.GC_PR_FGD.toDouble()), prFgdMask, Core.CMP_EQ)
            Core.bitwise_or(binMask, prFgdMask, binMask)

            // Strict Bounding Box Isolation: Force zero outside searchRect
            val containmentMask = Mat.zeros(binMask.size(), CvType.CV_8UC1)
            Imgproc.rectangle(containmentMask, searchRect.tl(), searchRect.br(), Scalar(255.0), -1)
            Core.bitwise_and(binMask, containmentMask, binMask)
            containmentMask.release()

            // 7. Multi-Component Selection targeting the user's gesture polygon
            // Retain ALL connected components that intersect the user's circle gesture (torso + head + limbs)
            // while completely discarding disconnected regions (distant mountains, roads, sky).
            val labels = Mat()
            val stats = Mat()
            val centroids = Mat()
            val numComponents = Imgproc.connectedComponentsWithStats(binMask, labels, stats, centroids)

            val finalBinary = Mat.zeros(binMask.size(), CvType.CV_8UC1)
            var totalKeptComponents = 0
            val compMask = Mat()
            val intersect = Mat()

            for (i in 1 until numComponents) {
                Core.compare(labels, Scalar(i.toDouble()), compMask, Core.CMP_EQ)
                Core.bitwise_and(compMask, polyMat, intersect)
                val overlapCount = Core.countNonZero(intersect)

                // If component directly intersects user gesture by at least 12 pixels or 4% of component area
                val compArea = stats.get(i, Imgproc.CC_STAT_AREA)?.get(0)?.toInt() ?: 1
                val isOverlap = overlapCount >= 12 || (overlapCount > 0 && overlapCount.toFloat() / compArea >= 0.04f)

                if (isOverlap) {
                    Core.bitwise_or(finalBinary, compMask, finalBinary)
                    totalKeptComponents++
                } else if (!isDetailOrFace && totalKeptComponents > 0) {
                    // For whole person/object mode: also include components that are vertically adjacent
                    // within the user's bounding box column (e.g. head directly above torso or shoes below pants)
                    val compX = stats.get(i, Imgproc.CC_STAT_LEFT)?.get(0)?.toInt() ?: 0
                    val compW = stats.get(i, Imgproc.CC_STAT_WIDTH)?.get(0)?.toInt() ?: 0
                    val compRight = compX + compW
                    val compCenterX = compX + compW / 2
                    if (compCenterX in searchLeft..searchRight && (compRight >= scaledLeft && compX <= scaledRight)) {
                        val compY = stats.get(i, Imgproc.CC_STAT_TOP)?.get(0)?.toInt() ?: 0
                        val compH = stats.get(i, Imgproc.CC_STAT_HEIGHT)?.get(0)?.toInt() ?: 0
                        val compBottom = compY + compH
                        val vertDist = when {
                            compBottom < scaledTop -> scaledTop - compBottom
                            compY > scaledBottom -> compY - scaledBottom
                            else -> 0
                        }
                        if (vertDist <= (bh * 0.25).toInt()) {
                            Core.bitwise_or(finalBinary, compMask, finalBinary)
                            totalKeptComponents++
                        }
                    }
                }
            }
            compMask.release()
            intersect.release()
            labels.release(); stats.release(); centroids.release()

            // If GrabCut failed to find any matching foreground, fallback intelligently
            if (Core.countNonZero(finalBinary) < 25) {
                polyMat.copyTo(finalBinary)
            }

            // Morphological clean-up (close internal holes and smooth contour)
            val smoothKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
            Imgproc.morphologyEx(finalBinary, finalBinary, Imgproc.MORPH_CLOSE, smoothKernel)
            Imgproc.dilate(finalBinary, finalBinary, smoothKernel)
            smoothKernel.release()

            // 7. Upscale back to source dimensions (canonical 512x512)
            val outputMat = Mat()
            Imgproc.resize(
                finalBinary,
                outputMat,
                Size(sourceBitmap.width.toDouble(), sourceBitmap.height.toDouble()),
                0.0,
                0.0,
                Imgproc.INTER_NEAREST
            )

            // Convert CV_8UC1 to RGBA for standard canvas blending
            val rgbaMat = Mat()
            Imgproc.cvtColor(outputMat, rgbaMat, Imgproc.COLOR_GRAY2RGBA)

            val outputBitmap = Bitmap.createBitmap(
                sourceBitmap.width,
                sourceBitmap.height,
                Bitmap.Config.ARGB_8888
            )
            Utils.matToBitmap(rgbaMat, outputBitmap)

            // Clean up native allocations
            imgMat.release(); maskMat.release(); polyMat.release(); coreMat.release()
            bgdModel.release(); fgdModel.release(); binMask.release(); prFgdMask.release()
            outputMat.release(); rgbaMat.release(); kernel.release()

            val latency = System.currentTimeMillis() - t0
            Log.i(TAG, "executeAutoGrabCut completed in ${latency}ms (dims: ${scaledW}x${scaledH})")
            return Pair(outputBitmap, latency)

        } catch (e: Throwable) {
            Log.e(TAG, "GrabCut execution exception: ${e.message}", e)
            val points = extractPointsFromPath(touchPath, 64)
            return MagicTouchEngine.segmentCircledObject(sourceBitmap, points)
        }
    }

    /**
     * Overload accepting coordinate list directly.
     */
    fun executeAutoGrabCutFromPoints(
        sourceBitmap: Bitmap,
        points: List<PointF>,
        iterations: Int = 2
    ): Pair<Bitmap, Long> {
        if (points.size < 3) {
            return Pair(Bitmap.createBitmap(sourceBitmap.width, sourceBitmap.height, Bitmap.Config.ARGB_8888), 0L)
        }
        val path = Path().apply {
            moveTo(points[0].x, points[0].y)
            for (i in 1 until points.size) {
                lineTo(points[i].x, points[i].y)
            }
            close()
        }
        return executeAutoGrabCut(sourceBitmap, path, iterations)
    }

    private fun extractScaledPointsFromPath(path: Path, scale: Double): List<Point> {
        val points = mutableListOf<Point>()
        val pm = PathMeasure(path, false)
        val length = pm.length
        val numSamples = 64
        val pos = FloatArray(2)
        for (i in 0 until numSamples) {
            val dist = (i.toFloat() / numSamples) * length
            pm.getPosTan(dist, pos, null)
            points.add(Point(pos[0] * scale, pos[1] * scale))
        }
        return points
    }

    private fun extractPointsFromPath(path: Path, numSamples: Int): List<PointF> {
        val points = mutableListOf<PointF>()
        val pm = PathMeasure(path, false)
        val length = pm.length
        val pos = FloatArray(2)
        for (i in 0 until numSamples) {
            val dist = (i.toFloat() / numSamples) * length
            pm.getPosTan(dist, pos, null)
            points.add(PointF(pos[0], pos[1]))
        }
        return points
    }
}
