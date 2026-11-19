package qidk.fix

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * "Smart AI Touch": turns a rough finger gesture (a lasso around an object, or a scribble over it) into a mask that
 * follows the object's real contour, using GrabCut.
 *
 *  - seeds: lasso -> filled loop = probable foreground, its innermost part (distance transform > 70% of the maximum)
 *    = definite foreground; scribble -> dilated stroke = probable foreground, stroke eroded by 4 px = definite foreground.
 *  - a 20% margin (at least 12 px) around the gesture is probable background, everything outside it is not part of the
 *    problem, and the outer 2 px of the crop (where it is not the image edge) are definite-background sinks.
 *  - only the ONE connected component that overlaps the gesture most survives; holes are filled; the result may not
 *    grow more than 12% of the gesture size beyond the gesture; a 3 px safety pad is added.
 *  - never a silent fallback: [Result.snapped] is false and [Result.reason] says why (the UI must tell the user).
 *
 * All inputs/outputs are in BITMAP pixel coordinates (use ViewportMapper before calling).
 * Runs on the calling thread (~0.3-0.6 s for a typical object): call it from a background thread.
 */
object SmartTouchSegmenter {
    init {
        try {
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                System.loadLibrary("opencv_java4")
            }
        } catch (_: Throwable) {}
    }
    const val TAG = "QIDK.Smart"
    const val GRABCUT_ITERS = 5
    const val GRABCUT_MAX_SIDE = 256          // GrabCut runs on the crop downscaled to this many px on its long side
    const val MARGIN_FRAC = 0.20
    const val MARGIN_MIN_PX = 12
    const val SCRIBBLE_CORE_ERODE_PX = 4
    const val LASSO_CORE_FRAC = 0.70
    const val LEAK_FRAC = 0.12
    const val EDGE_PAD_PX = 3
    const val MIN_SEED_PX = 12
    const val COLLAPSE_FRAC = 0.02

    class Result(
        /** width*height bytes, 255 = object, 0 = background. */
        val mask: ByteArray, val width: Int, val height: Int,
        /** false = GrabCut could not improve on the gesture; the mask is the gesture region itself. */
        val snapped: Boolean, val reason: String, val ms: Long
    ) {
        val areaPx: Int get() = mask.count { it.toInt() != 0 }

        /** Opaque white where the object is, transparent elsewhere (the convention InpaintCanvasView.applyGrabCutMask reads). */
        fun toBitmap(): Bitmap {
            val px = IntArray(width * height) { if (mask[it].toInt() != 0) -1 else 0 }
            return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { it.setPixels(px, 0, width, 0, 0, width, height) }
        }
    }

    /** A box as a closed stroke, so the old "box -> silhouette" button can use the same engine. */
    fun rectStroke(left: Float, top: Float, right: Float, bottom: Float): List<PointF> =
        listOf(PointF(left, top), PointF(right, top), PointF(right, bottom), PointF(left, bottom), PointF(left, top + 1f))

    private fun ellipse(k: Int): Mat = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))

    fun segment(source: Bitmap, stroke: List<PointF>, brushPx: Float = 14f): Result {
        val t0 = System.nanoTime()
        require(stroke.size >= 2) { "stroke needs at least 2 points" }
        val argb = if (source.config == Bitmap.Config.ARGB_8888) source else source.copy(Bitmap.Config.ARGB_8888, false)
        val w = argb.width
        val h = argb.height
        val n = stroke.size
        val xs = FloatArray(n) { stroke[it].x.coerceIn(0f, w - 1f) }
        val ys = FloatArray(n) { stroke[it].y.coerceIn(0f, h - 1f) }
        val g = StrokeGeometry.analyse(xs, ys)

        val owned = ArrayList<Mat>()
        fun <T : Mat> T.own(): T { owned.add(this); return this }
        try {
            val rgba = Mat().own()
            Utils.bitmapToMat(argb, rgba)
            val rgb = Mat().own()
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)

            // ---- 1. gesture -> probable-foreground region + definite-foreground core --------------------------------
            val brush = max(3, brushPx.roundToInt())
            val pts = Array(n) { Point(xs[it].roundToInt().toDouble(), ys[it].roundToInt().toDouble()) }
            val poly = MatOfPoint(*pts).own()
            val strokeM = Mat.zeros(h, w, CvType.CV_8UC1).own()
            Imgproc.polylines(strokeM, listOf(poly), false, Scalar(255.0), brush)
            val region = Mat.zeros(h, w, CvType.CV_8UC1).own()
            val core = Mat.zeros(h, w, CvType.CV_8UC1).own()
            if (g.isLoop) {
                Imgproc.fillPoly(region, listOf(poly), Scalar(255.0))
                Core.bitwise_or(region, strokeM, region)
                val dist = Mat().own()
                Imgproc.distanceTransform(region, dist, Imgproc.DIST_L2, 5)
                val dmax = Core.minMaxLoc(dist).maxVal
                val thr = Mat().own()
                Imgproc.threshold(dist, thr, LASSO_CORE_FRAC * dmax, 255.0, Imgproc.THRESH_BINARY)
                thr.convertTo(core, CvType.CV_8U)
            } else {
                val r = max(6, (0.10 * max(g.width, g.height)).roundToInt())
                Imgproc.dilate(strokeM, region, ellipse(2 * r + 1).own())
                Imgproc.erode(strokeM, core, ellipse(2 * SCRIBBLE_CORE_ERODE_PX + 1).own())
                if (Core.countNonZero(core) < MIN_SEED_PX) strokeM.copyTo(core)
            }
            if (Core.countNonZero(core) < MIN_SEED_PX) {
                Imgproc.erode(region, core, ellipse(5).own())
                if (Core.countNonZero(core) < MIN_SEED_PX) region.copyTo(core)
            }
            Core.bitwise_and(core, region, core)

            fun finish(mask: Mat, snapped: Boolean, reason: String): Result {
                val bytes = ByteArray(w * h)
                mask.get(0, 0, bytes)
                val ms = (System.nanoTime() - t0) / 1_000_000
                val res = Result(bytes, w, h, snapped, reason, ms)
                Log.i(TAG, "loop=${g.isLoop} compact=${"%.2f".format(g.compactness)} snapped=$snapped reason=$reason area=${res.areaPx}px ms=$ms")
                return res
            }
            fun fallback(reason: String): Result {
                val padded = Mat().own()
                Imgproc.dilate(region, padded, ellipse(2 * EDGE_PAD_PX + 1).own())
                return finish(padded, false, reason)
            }

            // ---- 2. crop = gesture box + margin ------------------------------------------------------------------------
            val nz = Mat().own()
            Core.findNonZero(region, nz)
            if (nz.empty()) return fallback("empty_gesture")
            val bb = Imgproc.boundingRect(nz)
            val margin = max((MARGIN_FRAC * max(bb.width, bb.height)).roundToInt(), MARGIN_MIN_PX)
            val ax0 = max(0, bb.x - margin)
            val ay0 = max(0, bb.y - margin)
            val ax1 = min(w, bb.x + bb.width + margin)
            val ay1 = min(h, bb.y + bb.height + margin)
            val rw = ax1 - ax0
            val rh = ay1 - ay0
            if (rw < 16 || rh < 16) return fallback("crop_too_small")
            val roi = Rect(ax0, ay0, rw, rh)
            val regR = region.submat(roi).clone().own()
            val coreR = core.submat(roi).clone().own()
            if (rw * rh - Core.countNonZero(regR) < 50) return fallback("no_background_context")

            // ---- 3. GrabCut labels --------------------------------------------------------------------------------------
            val gc = Mat(rh, rw, CvType.CV_8UC1, Scalar(Imgproc.GC_PR_BGD.toDouble())).own()
            val bgd = Scalar(Imgproc.GC_BGD.toDouble())
            fun sinks(m: Mat, s: Int) {
                val mw = m.cols(); val mh = m.rows()
                if (ax0 > 0) m.submat(0, mh, 0, s).setTo(bgd)
                if (ay0 > 0) m.submat(0, s, 0, mw).setTo(bgd)
                if (ax1 < w) m.submat(0, mh, mw - s, mw).setTo(bgd)
                if (ay1 < h) m.submat(mh - s, mh, 0, mw).setTo(bgd)
            }
            sinks(gc, 2)
            gc.setTo(Scalar(Imgproc.GC_PR_FGD.toDouble()), regR)
            gc.setTo(Scalar(Imgproc.GC_FGD.toDouble()), coreR)

            val imgR = rgb.submat(roi).clone().own()
            val scale = min(1.0, GRABCUT_MAX_SIDE.toDouble() / max(rw, rh))
            val gw = max(16, (rw * scale).roundToInt())
            val gh = max(16, (rh * scale).roundToInt())
            val imgG: Mat
            val gcG: Mat
            if (scale < 1.0) {
                imgG = Mat().own(); gcG = Mat().own()
                Imgproc.resize(imgR, imgG, Size(gw.toDouble(), gh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                Imgproc.resize(gc, gcG, Size(gw.toDouble(), gh.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
                sinks(gcG, 1)
            } else {
                imgG = imgR; gcG = gc
            }
            try {
                Imgproc.grabCut(imgG, gcG, Rect(), Mat().own(), Mat().own(), GRABCUT_ITERS, Imgproc.GC_INIT_WITH_MASK)
            } catch (e: Exception) {
                Log.w(TAG, "grabCut failed: ${e.message}")
                return fallback("grabcut_error")
            }
            val fgA = Mat().own(); val fgB = Mat().own(); val fgG = Mat().own()
            Core.inRange(gcG, Scalar(Imgproc.GC_FGD.toDouble()), Scalar(Imgproc.GC_FGD.toDouble()), fgA)
            Core.inRange(gcG, Scalar(Imgproc.GC_PR_FGD.toDouble()), Scalar(Imgproc.GC_PR_FGD.toDouble()), fgB)
            Core.bitwise_or(fgA, fgB, fgG)
            val fg = Mat().own()
            if (scale < 1.0) {
                Imgproc.resize(fgG, fg, Size(rw.toDouble(), rh.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
                Imgproc.threshold(fg, fg, 127.0, 255.0, Imgproc.THRESH_BINARY)
            } else {
                fgG.copyTo(fg)
            }

            // ---- 4. keep ONE component: the one overlapping the gesture most ------------------------------------------
            val labels = Mat().own(); val stats = Mat().own(); val cents = Mat().own()
            val count = Imgproc.connectedComponentsWithStats(fg, labels, stats, cents, 8, CvType.CV_32S)
            if (count <= 1) return fallback("empty_result")
            val lab = IntArray(rw * rh)
            labels.get(0, 0, lab)
            val reg = ByteArray(rw * rh)
            regR.get(0, 0, reg)
            val overlap = IntArray(count)
            for (i in lab.indices) if (reg[i].toInt() != 0) overlap[lab[i]]++
            overlap[0] = 0
            var best = 0
            for (l in 1 until count) if (overlap[l] > overlap[best]) best = l
            if (overlap[best] == 0) return fallback("no_overlap")
            val comp = Mat().own()
            Core.compare(labels, Scalar(best.toDouble()), comp, Core.CMP_EQ)

            // ---- 5. fill holes, close, bound by the gesture, pad ------------------------------------------------------
            val contours = ArrayList<MatOfPoint>()
            Imgproc.findContours(comp.clone().own(), contours, Mat().own(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val filled = Mat.zeros(rh, rw, CvType.CV_8UC1).own()
            Imgproc.drawContours(filled, contours, -1, Scalar(255.0), -1)
            contours.forEach { it.release() }
            Imgproc.morphologyEx(filled, filled, Imgproc.MORPH_CLOSE, ellipse(5).own())
            val leak = max(4, (LEAK_FRAC * max(bb.width, bb.height)).roundToInt())
            val limit = Mat().own()
            Imgproc.dilate(regR, limit, ellipse(2 * leak + 1).own())
            Core.bitwise_and(filled, limit, filled)
            Imgproc.dilate(filled, filled, ellipse(2 * EDGE_PAD_PX + 1).own())
            if (Core.countNonZero(filled) < max(MIN_SEED_PX.toDouble(), COLLAPSE_FRAC * Core.countNonZero(regR)))
                return fallback("collapsed")

            val out = Mat.zeros(h, w, CvType.CV_8UC1).own()
            filled.copyTo(out.submat(roi))
            return finish(out, true, "ok")
        } finally {
            owned.forEach { it.release() }
        }
    }
}
