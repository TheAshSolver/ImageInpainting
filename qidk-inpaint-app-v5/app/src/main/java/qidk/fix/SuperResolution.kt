package qidk.fix

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/**
 * One selectable 2x upscaler. [run] receives the WHOLE canvas image (512 x 512) and returns a 1024 x 1024 bitmap.
 * There is no mask anywhere in this path: super-resolution must never be routed through an inpainting call.
 */
interface SrEngine {
    val id: String
    val label: String
    val family: ModelFamily
    /** null = usable. Otherwise why it is disabled; the UI greys the entry out and shows this text. */
    val unavailableReason: String?
    fun run(src: Bitmap, backend: Backend): Bitmap
}

/** Placeholder for an engine whose recipe is not ported yet: listed, greyed out, never runs. */
class UnavailableSrEngine(
    override val id: String, override val label: String, override val family: ModelFamily, reason: String
) : SrEngine {
    override val unavailableReason: String? = reason
    override fun run(src: Bitmap, backend: Backend): Bitmap = error("$label is disabled: $unavailableReason")
}

/** Engine 1: plain bicubic via OpenCV (the baseline in the benchmark). */
class BicubicSrEngine : SrEngine {
    init {
        try {
            if (!org.opencv.android.OpenCVLoader.initDebug()) {
                System.loadLibrary("opencv_java4")
            }
        } catch (_: Throwable) {}
    }
    override val id = "bicubic"
    override val label = "Bicubic Baseline (OpenCV)"
    override val family = ModelFamily.SR_BICUBIC
    override val unavailableReason: String? = null
    override fun run(src: Bitmap, backend: Backend): Bitmap {
        val inM = Mat(); val outM = Mat()
        try {
            Utils.bitmapToMat(src, inM)
            Imgproc.resize(inM, outM, Size(src.width * 2.0, src.height * 2.0), 0.0, 0.0, Imgproc.INTER_CUBIC)
            val out = Bitmap.createBitmap(src.width * 2, src.height * 2, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(outM, out)
            return out
        } finally { inM.release(); outM.release() }
    }
}

/** Before / after viewer: draggable split line, pinch-zoom about the fingers, one-finger pan when zoomed. */
class CompareView(context: Context) : View(context) {
    private val mapper = ViewportMapper()
    private val nearest = Paint().apply { isFilterBitmap = false }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; strokeWidth = 4f }
    private val tag = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = 34f; setShadowLayer(4f, 0f, 0f, Color.BLACK) }
    private var split = 0.5f
    private var draggingSplit = false
    private var lastX = 0f
    private var lastY = 0f
    private var scaling = false

    var before: Bitmap? = null
        set(v) { field = v; invalidate() }
    var after: Bitmap? = null
        set(v) {
            field = v
            v?.let { mapper.setBitmap(it.width, it.height) }
            mapper.setUserTransform(1f, 0f, 0f)
            invalidate()
        }

    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            val fx = d.focusX; val fy = d.focusY
            val bx = mapper.viewToBitmapRawX(fx); val by = mapper.viewToBitmapRawY(fy)
            mapper.setUserTransform(mapper.userScale * d.scaleFactor, mapper.panX, mapper.panY)
            mapper.setUserTransform(mapper.userScale, mapper.panX + (fx - mapper.bitmapToViewX(bx)), mapper.panY + (fy - mapper.bitmapToViewY(by)))
            invalidate()
            return true
        }
    })

    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) { mapper.setView(w, h); invalidate() }

    override fun onTouchEvent(e: MotionEvent): Boolean {
        parent?.requestDisallowInterceptTouchEvent(true)
        scaleDetector.onTouchEvent(e)
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                draggingSplit = abs(e.x - split * width) < 56f * resources.displayMetrics.density
                lastX = e.x; lastY = e.y; scaling = false
            }
            MotionEvent.ACTION_POINTER_DOWN -> { scaling = true; draggingSplit = false }
            MotionEvent.ACTION_MOVE -> {
                if (e.pointerCount > 1 || scaleDetector.isInProgress) scaling = true
                else if (!scaling) {
                    if (draggingSplit) split = (e.x / width).coerceIn(0.02f, 0.98f)
                    else if (mapper.userScale > 1f) mapper.setUserTransform(mapper.userScale, mapper.panX + e.x - lastX, mapper.panY + e.y - lastY)
                    lastX = e.x; lastY = e.y
                    invalidate()
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { draggingSplit = false; scaling = false }
        }
        return true
    }

    override fun onDraw(c: Canvas) {
        c.drawColor(Color.rgb(24, 24, 28))
        val a = after ?: return
        val b = before ?: a
        val dst = RectF(mapper.destLeft, mapper.destTop, mapper.destRight, mapper.destBottom)
        val sx = split * width
        c.save(); c.clipRect(0f, 0f, sx, height.toFloat()); c.drawBitmap(b, null, dst, nearest); c.restore()
        c.save(); c.clipRect(sx, 0f, width.toFloat(), height.toFloat()); c.drawBitmap(a, null, dst, nearest); c.restore()
        c.drawLine(sx, 0f, sx, height.toFloat(), line)
        c.drawCircle(sx, height / 2f, 22f, line)
        c.drawText("Before ${b.width} x ${b.height}", 16f, 44f, tag)
        val t = "After ${a.width} x ${a.height}"
        c.drawText(t, width - tag.measureText(t) - 16f, 44f, tag)
        c.drawText("x%.1f".format(mapper.userScale), 16f, height - 16f, tag)
    }
}

/** The "Super Resolution" screen: engine + backend pickers, Upscale button, compare viewer, 1024 x 1024 / latency readout. */
class SuperResolutionView(
    context: Context,
    private val engines: List<SrEngine>,
    private val sourceProvider: () -> Bitmap?
) : LinearLayout(context) {
    private val tag = "QIDK.SR"
    private val engineSpinner = Spinner(context)
    private val backendSpinner = Spinner(context)
    private val runButton = Button(context).apply { text = "Upscale 2x" }
    private val compare = CompareView(context)
    private val info = TextView(context).apply { setTextColor(Color.WHITE); textSize = 14f }
    private var busy = false

    init {
        orientation = VERTICAL
        setBackgroundColor(Color.rgb(24, 24, 28))
        val pad = (8 * resources.displayMetrics.density).toInt()
        val row = LinearLayout(context).apply { orientation = HORIZONTAL; setPadding(pad, pad, pad, pad); gravity = Gravity.CENTER_VERTICAL }
        row.addView(engineSpinner, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1.4f))
        row.addView(backendSpinner, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
        row.addView(runButton, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT))
        addView(row, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(compare, LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f))
        info.setPadding(pad, pad, pad, pad)
        addView(info, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))

        engineSpinner.adapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item,
            engines.map { if (it.unavailableReason == null) it.label else "${it.label}  (unavailable)" })
        engineSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) { onEngineChanged() }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
        runButton.setOnClickListener { runSelected() }
        onEngineChanged()
        info.text = "Pick an engine and press Upscale 2x. The whole canvas is upscaled; no mask is needed."
    }

    private fun engine() = engines[engineSpinner.selectedItemPosition.coerceIn(0, engines.size - 1)]

    private fun onEngineChanged() {
        val e = engine()
        val allowed = BackendMatrix.allowed(e.family)
        backendSpinner.adapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, allowed.map { it.label })
        backendSpinner.isEnabled = allowed.size > 1
        runButton.isEnabled = e.unavailableReason == null && !busy
        if (e.unavailableReason != null) info.text = "${e.label} is disabled: ${e.unavailableReason}"
    }

    /** Re-reads the canvas image (call when this page becomes visible). */
    fun refreshSource() { compare.before = sourceProvider() }

    private fun runSelected() {
        val src = sourceProvider()
        if (src == null) { info.text = "Load an image on the Inpaint tab first."; return }
        val e = engine()
        val backend = BackendMatrix.allowed(e.family)[backendSpinner.selectedItemPosition.coerceIn(0, BackendMatrix.allowed(e.family).size - 1)]
        busy = true; runButton.isEnabled = false; info.text = "Running ${e.label} on ${backend.label} ..."
        compare.before = src
        Thread {
            val t0 = System.nanoTime()
            val result = try { Result.success(e.run(src, backend)) } catch (t: Throwable) { Result.failure(t) }
            val ms = (System.nanoTime() - t0) / 1_000_000
            post {
                busy = false; runButton.isEnabled = true
                result.onSuccess { out ->
                    val ok = out.width == src.width * 2 && out.height == src.height * 2
                    compare.after = out
                    info.text = (if (ok) "Output ${out.width} x ${out.height}" else "WRONG SIZE ${out.width} x ${out.height} (expected ${src.width * 2} x ${src.height * 2})") +
                        "  |  $ms ms  |  ${e.label}  |  ${backend.label}"
                    Log.i(tag, "engine=${e.id} backend=$backend out=${out.width}x${out.height} ms=$ms ok=$ok")
                }.onFailure { t ->
                    info.text = "Failed: ${t.message}"
                    Log.e(tag, "engine=${e.id} failed", t)
                }
            }
        }.start()
    }

    /** Synchronous variant for the adb debug hook: returns the log line. */
    fun runForTest(engineId: String, backend: Backend): String {
        val e = engines.firstOrNull { it.id == engineId } ?: return "unknown engine $engineId"
        e.unavailableReason?.let { return "engine disabled: $it" }
        BackendMatrix.require(e.family, backend)
        val src = sourceProvider() ?: return "no source image"
        val t0 = System.nanoTime()
        val out = e.run(src, backend)
        return "engine=$engineId backend=$backend out=${out.width}x${out.height} ms=${(System.nanoTime() - t0) / 1_000_000}"
    }
}

/**
 * Gives an activity that has no navigation a two-button tab bar: "Inpaint" (the existing screen, untouched) and
 * "Super Resolution" (the page). Call once from onCreate after setContentView. If the activity already has a bottom
 * navigation / TabLayout / ViewPager, add a "Super Resolution" item there instead and show [page] for it.
 */
object SuperResolutionTab {
    fun install(activity: Activity, page: View) {
        val content = activity.findViewById<ViewGroup>(android.R.id.content)
        val original = content.getChildAt(0) ?: error("setContentView must be called before SuperResolutionTab.install")
        content.removeView(original)
        val match = ViewGroup.LayoutParams.MATCH_PARENT
        val root = LinearLayout(activity).apply { orientation = LinearLayout.VERTICAL }
        val bar = LinearLayout(activity).apply { orientation = LinearLayout.HORIZONTAL }
        val inpaint = Button(activity).apply { text = "Inpaint" }
        val sr = Button(activity).apply { text = "Super Resolution" }
        bar.addView(inpaint, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        bar.addView(sr, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        val body = FrameLayout(activity)
        body.addView(original, FrameLayout.LayoutParams(match, match))
        body.addView(page, FrameLayout.LayoutParams(match, match))
        page.visibility = View.GONE
        inpaint.setOnClickListener { original.visibility = View.VISIBLE; page.visibility = View.GONE }
        sr.setOnClickListener { original.visibility = View.GONE; page.visibility = View.VISIBLE; (page as? SuperResolutionView)?.refreshSource() }
        root.addView(bar, LinearLayout.LayoutParams(match, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(body, LinearLayout.LayoutParams(match, 0, 1f))
        content.addView(root, ViewGroup.LayoutParams(match, match))
    }
}
