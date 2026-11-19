package qidk.fix

import android.animation.ValueAnimator
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import android.view.MotionEvent
import android.view.animation.LinearInterpolator
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Owns the whole "Smart AI Touch" interaction so the canvas view only has to forward events:
 *
 *   onTouchEvent(e)  -> if (mode == SMART) return controller.onTouch(e)
 *   onDraw(canvas)   -> (after the bitmap and the mask) controller.draw(canvas)
 *
 * While the finger moves it records the stroke in BITMAP coordinates (through [mapper]) and draws an animated, soft
 * white glow in VIEW coordinates. On ACTION_UP it shows a spinner, runs [SmartTouchSegmenter] on a worker thread and
 * delivers the result on the main thread through [onResult] (result.snapped == false means "kept your outline").
 */
class SmartGestureController(
    private val mapper: ViewportMapper,
    private val sourceProvider: () -> Bitmap?,
    private val invalidate: () -> Unit,
    private val postToMain: (Runnable) -> Unit,
    private val onResult: (SmartTouchSegmenter.Result) -> Unit,
    private val onIgnored: (String) -> Unit = {},
    private val brushBitmapPx: () -> Float = { 14f }
) {
    private val tag = "QIDK.Smart"
    private val bmpPts = ArrayList<PointF>()
    private val viewPath = Path()
    private var thinking = false
    private var angle = 0f
    private var pathLenBmp = 0f
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "smart-touch").apply { isDaemon = true } }

    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND; strokeWidth = 30f
    }
    private val corePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; strokeJoin = Paint.Join.ROUND; strokeWidth = 8f
    }
    private val spinPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND; strokeWidth = 7f; color = Color.WHITE
    }
    private val pulse = ValueAnimator.ofFloat(0f, 360f).apply {
        duration = 900; repeatCount = ValueAnimator.INFINITE; interpolator = LinearInterpolator()
        addUpdateListener { angle = it.animatedValue as Float; invalidate() }
    }

    val isBusy: Boolean get() = thinking

    fun onTouch(e: MotionEvent): Boolean {
        if (thinking) return true                       // swallow touches while GrabCut runs
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                bmpPts.clear(); viewPath.reset(); pathLenBmp = 0f
                addPoint(e.x, e.y, first = true)
                if (!pulse.isStarted) pulse.start()
            }
            MotionEvent.ACTION_MOVE -> {
                for (i in 0 until e.historySize) addPoint(e.getHistoricalX(i), e.getHistoricalY(i))
                addPoint(e.x, e.y)
            }
            MotionEvent.ACTION_UP -> {
                addPoint(e.x, e.y)
                finish()
            }
            MotionEvent.ACTION_CANCEL -> { reset() }
        }
        invalidate()
        return true
    }

    private fun addPoint(x: Float, y: Float, first: Boolean = false) {
        val p = PointF(mapper.viewToBitmapX(x), mapper.viewToBitmapY(y))
        val last = bmpPts.lastOrNull()
        if (first) viewPath.moveTo(x, y)
        else {
            if (last != null && hypot(p.x - last.x, p.y - last.y) < 2f) return
            viewPath.lineTo(x, y)
            if (last != null) pathLenBmp += hypot(p.x - last.x, p.y - last.y)
        }
        bmpPts.add(p)
    }

    private fun reset() {
        bmpPts.clear(); viewPath.reset(); thinking = false
        pulse.cancel(); invalidate()
    }

    private fun finish() {
        val src = sourceProvider()
        if (src == null || bmpPts.size < 3 || pathLenBmp < 16f) {
            reset(); onIgnored("Draw around or across the object"); return
        }
        thinking = true
        val pts = ArrayList(bmpPts)
        val brush = brushBitmapPx()
        worker.execute {
            val res = try {
                SmartTouchSegmenter.segment(src, pts, brush)
            } catch (t: Throwable) {
                Log.e(tag, "segmentation crashed", t); null
            }
            postToMain(Runnable {
                reset()
                if (res != null) onResult(res) else onIgnored("Smart selection failed - use Manual Precision")
            })
        }
    }

    fun draw(canvas: Canvas) {
        if (bmpPts.isEmpty()) return
        val a = (0.5f + 0.5f * sin(Math.toRadians(angle.toDouble()).toFloat())) // 0..1 pulse
        glowPaint.color = Color.argb((60 + 50 * a).toInt(), 255, 255, 255)
        corePaint.color = Color.argb((170 + 70 * a).toInt(), 255, 255, 255)
        canvas.drawPath(viewPath, glowPaint)
        canvas.drawPath(viewPath, corePaint)
        if (thinking) {
            val b = RectF()
            viewPath.computeBounds(b, true)
            val cx = b.centerX(); val cy = b.centerY()
            canvas.drawArc(RectF(cx - 26f, cy - 26f, cx + 26f, cy + 26f), angle, 110f, false, spinPaint)
        }
    }

    fun shutdown() { pulse.cancel(); worker.shutdownNow() }
}
