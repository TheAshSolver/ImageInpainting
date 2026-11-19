package qidk.fix

import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.drawable.GradientDrawable
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
import com.qualcomm.qidk.inpaint.engine.OnDeviceProcessDriver
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/**
 * Super-Resolution engine interface.
 * Supports progressive multi-scale upscaling (2x, 4x, 8x).
 */
interface SrEngine {
    val id: String
    val label: String
    val family: ModelFamily
    /** null = usable. Otherwise why it is disabled; the UI greys the entry out and shows this text. */
    val unavailableReason: String?
    fun run(src: Bitmap, backend: Backend): Bitmap = run(src, backend, 2)
    fun run(src: Bitmap, backend: Backend, scale: Int): Bitmap
}

/** Placeholder for an engine whose recipe is not ported yet. */
class UnavailableSrEngine(
    override val id: String, override val label: String, override val family: ModelFamily, reason: String
) : SrEngine {
    override val unavailableReason: String? = reason
    override fun run(src: Bitmap, backend: Backend, scale: Int): Bitmap = error("$label is disabled: $unavailableReason")
}

/** Plain bicubic via OpenCV. */
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

    override fun run(src: Bitmap, backend: Backend): Bitmap = run(src, backend, 2)

    override fun run(src: Bitmap, backend: Backend, scale: Int): Bitmap {
        val targetW = src.width * scale
        val targetH = src.height * scale
        val inM = Mat(); val outM = Mat()
        try {
            Utils.bitmapToMat(src, inM)
            Imgproc.resize(inM, outM, Size(targetW.toDouble(), targetH.toDouble()), 0.0, 0.0, Imgproc.INTER_CUBIC)
            val out = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(outM, out)
            return out
        } finally { inM.release(); outM.release() }
    }
}

/** High-contrast dark-mode adapter to eliminate unreadable black-on-grey text on Android 15. */
class HighContrastSpinnerAdapter(
    context: Context,
    items: List<String>
) : ArrayAdapter<String>(context, android.R.layout.simple_spinner_item, items) {

    init {
        setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
    }

    override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
        val v = super.getView(position, convertView, parent)
        if (v is TextView) {
            v.setTextColor(Color.parseColor("#F8FAFC"))
            v.textSize = 13f
            v.setPadding(12, 10, 12, 10)
        }
        return v
    }

    override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
        val v = super.getDropDownView(position, convertView, parent)
        if (v is TextView) {
            v.setTextColor(Color.parseColor("#FFFFFF"))
            v.setBackgroundColor(Color.parseColor("#2D2D38"))
            v.textSize = 14f
            val padH = (16 * parent.resources.displayMetrics.density).toInt()
            val padV = (12 * parent.resources.displayMetrics.density).toInt()
            v.setPadding(padH, padV, padH, padV)
        }
        return v
    }
}

/** Before / after viewer: draggable split line, pinch-zoom about the fingers, one-finger pan when zoomed. */
class CompareView(context: Context) : View(context) {
    private val mapper = ViewportMapper()
    private val nearest = Paint().apply { isFilterBitmap = false }
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.parseColor("#38BDF8"); strokeWidth = 5f }
    private val tag = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#F8FAFC")
        textSize = 34f
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }
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
        c.drawColor(Color.parseColor("#121216"))
        val a = after ?: return
        val b = before ?: a
        val dst = RectF(mapper.destLeft, mapper.destTop, mapper.destRight, mapper.destBottom)
        val sx = split * width
        c.save(); c.clipRect(0f, 0f, sx, height.toFloat()); c.drawBitmap(b, null, dst, nearest); c.restore()
        c.save(); c.clipRect(sx, 0f, width.toFloat(), height.toFloat()); c.drawBitmap(a, null, dst, nearest); c.restore()
        c.drawLine(sx, 0f, sx, height.toFloat(), line)
        c.drawCircle(sx, height / 2f, 22f, line)
        c.drawText("Before ${b.width} x ${b.height}", 20f, 48f, tag)
        val t = "After ${a.width} x ${a.height}"
        c.drawText(t, width - tag.measureText(t) - 20f, 48f, tag)
        c.drawText("x%.1f".format(mapper.userScale), 20f, height - 20f, tag)
    }
}

/**
 * Redesigned Super-Resolution screen with dark palette:
 * - Background: #121216
 * - Containers: #1E1E26 (8dp rounded corners)
 * - Headers: #F8FAFC
 * - Secondary labels: #94A3B8
 * - Spinners: High-contrast dropdowns (#FFFFFF on #2D2D38)
 * - Multi-scale factor selector: 2x, 4x, 8x with thermal guardrail
 */
class SuperResolutionView(
    context: Context,
    private val engines: List<SrEngine>,
    private val sourceProvider: () -> Bitmap?
) : LinearLayout(context) {
    private val tag = "QIDK.SR"

    private val telemetryBar = TelemetryBarView(context)
    private val controlCard: LinearLayout
    private val statusCard: LinearLayout
    private val cardHeader: TextView
    private val engineSpinner = Spinner(context)
    private val backendSpinner = Spinner(context)
    private val scaleSpinner = Spinner(context)
    private val runButton = Button(context)
    private val previewBadge = TextView(context)
    private val thermalWarning = TextView(context)
    private val compare = CompareView(context)
    private val info = TextView(context)
    private var busy = false

    private val scaleFactors = listOf(2, 4, 8)

    init {
        orientation = VERTICAL
        setBackgroundColor(Color.parseColor("#0B0F19"))
        val density = resources.displayMetrics.density
        val pad = (12 * density).toInt()
        val cornerRadius = 8 * density

        // Persistent Telemetry HUD Bar
        val lpTelemetry = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
            setMargins(pad, (8 * density).toInt(), pad, (4 * density).toInt())
        }
        addView(telemetryBar, lpTelemetry)

        // Card Container for Execution Controls
        controlCard = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(pad, pad, pad, pad)
            val gd = GradientDrawable().apply {
                setColor(Color.parseColor("#161F30"))
                this.cornerRadius = cornerRadius
            }
            background = gd
            val lp = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(pad, (4 * density).toInt(), pad, (6 * density).toInt())
            }
            layoutParams = lp
        }

        // Header label
        cardHeader = TextView(context).apply {
            text = "⚡ Neural Super-Resolution Engine"
            setTextColor(Color.parseColor("#F8FAFC"))
            textSize = 15f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            setPadding(0, 0, 0, (6 * density).toInt())
        }
        controlCard.addView(cardHeader)

        // Row 1: Engine + Backend pickers
        val row1 = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, (6 * density).toInt())
        }

        val gdEngine = GradientDrawable().apply {
            setColor(Color.parseColor("#2D2D38"))
            this.cornerRadius = 6 * density
        }
        engineSpinner.background = gdEngine
        val gdBackend = GradientDrawable().apply {
            setColor(Color.parseColor("#2D2D38"))
            this.cornerRadius = 6 * density
        }
        backendSpinner.background = gdBackend

        row1.addView(engineSpinner, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1.5f).apply { marginEnd = (6 * density).toInt() })
        row1.addView(backendSpinner, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1.0f))
        controlCard.addView(row1)

        // Row 2: Scale selector + Action button
        val row2 = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val gdScale = GradientDrawable().apply {
            setColor(Color.parseColor("#2D2D38"))
            this.cornerRadius = 6 * density
        }
        scaleSpinner.background = gdScale

        val scaleLabels = listOf(
            "2x (1024²) • ~4 MB RAM",
            "4x (2048²) • ~16 MB RAM",
            "8x (4096²) • ~64 MB RAM"
        )
        scaleSpinner.adapter = HighContrastSpinnerAdapter(context, scaleLabels)
        scaleSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                updateScalePreview()
            }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }

        runButton.apply {
            text = "Upscale 2x"
            setTextColor(Color.WHITE)
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            val gdBtn = GradientDrawable().apply {
                setColor(Color.parseColor("#10B981"))
                this.cornerRadius = 6 * density
            }
            background = gdBtn
            setPadding((16 * density).toInt(), (8 * density).toInt(), (16 * density).toInt(), (8 * density).toInt())
            setOnClickListener { runSelected() }
        }

        row2.addView(scaleSpinner, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1.5f).apply { marginEnd = (6 * density).toInt() })
        row2.addView(runButton, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT))
        controlCard.addView(row2)

        // Preview & Guardrail indicators
        previewBadge.apply {
            setTextColor(Color.parseColor("#94A3B8"))
            textSize = 12f
            setPadding(0, (6 * density).toInt(), 0, 0)
        }
        controlCard.addView(previewBadge)

        thermalWarning.apply {
            setTextColor(Color.parseColor("#F87171"))
            textSize = 12f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            visibility = View.GONE
            setPadding(0, (4 * density).toInt(), 0, 0)
        }
        controlCard.addView(thermalWarning)

        addView(controlCard)

        // Comparison Viewport
        addView(compare, LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f))

        // Status / Telemetry Footer Card
        statusCard = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(pad, (8 * density).toInt(), pad, (8 * density).toInt())
            val gd = GradientDrawable().apply {
                setColor(Color.parseColor("#161F30"))
                this.cornerRadius = cornerRadius
            }
            background = gd
            val lp = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(pad, (4 * density).toInt(), pad, pad)
            }
            layoutParams = lp
        }

        info.apply {
            setTextColor(Color.parseColor("#F8FAFC"))
            textSize = 13f
            text = "Pick an engine, backend, and scale factor. No mask required."
        }
        statusCard.addView(info)
        addView(statusCard)

        // Setup Engine Spinner adapter
        engineSpinner.adapter = HighContrastSpinnerAdapter(context,
            engines.map { if (it.unavailableReason == null) it.label else "${it.label}  (unavailable)" })
        engineSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) { onEngineChanged() }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }

        onEngineChanged()
        updateScalePreview()
    }

    private fun selectedScale(): Int {
        val idx = scaleSpinner.selectedItemPosition.coerceIn(0, scaleFactors.size - 1)
        return scaleFactors[idx]
    }

    private fun updateScalePreview() {
        val s = selectedScale()
        runButton.text = "Upscale ${s}x"
        val outDim = 512 * s
        val ramMb = when (s) {
            2 -> 4
            4 -> 16
            8 -> 64
            else -> 4
        }
        val passes = when (s) {
            2 -> 1
            4 -> 2
            8 -> 3
            else -> 1
        }
        previewBadge.text = "Output: ${outDim}x${outDim} ($s×) • Peak RAM: ~${ramMb} MB • Passes: $passes"

        // Thermal Guardrail Check
        val peakTemp = OnDeviceProcessDriver.readPeakTemperatureC()
        if (s == 8 && peakTemp > 65.0f) {
            thermalWarning.text = "⚠️ Thermal Guardrail: Device peak temperature is %.1f°C (>65°C). 8x upscaling may cause throttling.".format(peakTemp)
            thermalWarning.visibility = View.VISIBLE
        } else {
            thermalWarning.visibility = View.GONE
        }
    }

    private fun engine() = engines[engineSpinner.selectedItemPosition.coerceIn(0, engines.size - 1)]

    private fun onEngineChanged() {
        val e = engine()
        val allowed = BackendMatrix.allowed(e.family)
        backendSpinner.adapter = HighContrastSpinnerAdapter(context, allowed.map { it.label })
        backendSpinner.isEnabled = allowed.size > 1
        runButton.isEnabled = e.unavailableReason == null && !busy
        if (e.unavailableReason != null) {
            info.text = "${e.label} is disabled: ${e.unavailableReason}"
            info.setTextColor(Color.parseColor("#F87171"))
        } else {
            info.text = "Ready to upscale using ${e.label}."
            info.setTextColor(Color.parseColor("#F8FAFC"))
        }
    }

    /** Re-reads the canvas image (call when this page becomes visible). */
    fun refreshSource() {
        compare.before = sourceProvider()
        updateScalePreview()
    }

    private fun runSelected() {
        val src = sourceProvider()
        if (src == null) {
            info.text = "Load an image on the Inpaint tab first."
            info.setTextColor(Color.parseColor("#F87171"))
            return
        }
        val e = engine()
        val allowed = BackendMatrix.allowed(e.family)
        val backend = allowed[backendSpinner.selectedItemPosition.coerceIn(0, allowed.size - 1)]
        val scale = selectedScale()

        busy = true
        runButton.isEnabled = false
        info.text = "Running ${e.label} on ${backend.label} (${scale}x)..."
        info.setTextColor(Color.parseColor("#38BDF8"))
        compare.before = src

        Thread {
            val t0 = System.nanoTime()
            val result = try {
                Result.success(e.run(src, backend, scale))
            } catch (t: Throwable) {
                Result.failure(t)
            }
            val ms = (System.nanoTime() - t0) / 1_000_000

            post {
                busy = false
                runButton.isEnabled = true
                result.onSuccess { out ->
                    val expectedW = src.width * scale
                    val expectedH = src.height * scale
                    val ok = out.width == expectedW && out.height == expectedH
                    compare.after = out
                    info.text = (if (ok) "Output ${out.width} x ${out.height}" else "WRONG SIZE ${out.width} x ${out.height} (expected $expectedW x $expectedH)") +
                        "  |  $ms ms  |  ${e.label}  |  ${backend.label}"
                    info.setTextColor(if (ok) Color.parseColor("#10B981") else Color.parseColor("#F87171"))
                    Log.i(tag, "engine=${e.id} backend=$backend scale=${scale}x out=${out.width}x${out.height} ms=$ms ok=$ok")
                }.onFailure { t ->
                    info.text = "Failed: ${t.message}"
                    info.setTextColor(Color.parseColor("#F87171"))
                    Log.e(tag, "engine=${e.id} failed", t)
                }
            }
        }.start()
    }

    /** Synchronous variant for the adb debug hook: returns the log line. */
    fun runForTest(engineId: String, backend: Backend, scale: Int = 2): String {
        val e = engines.firstOrNull { it.id == engineId } ?: return "unknown engine $engineId"
        e.unavailableReason?.let { return "engine disabled: $it" }
        BackendMatrix.require(e.family, backend)
        val src = sourceProvider() ?: return "no source image"
        val t0 = System.nanoTime()
        val out = e.run(src, backend, scale)
        return "engine=$engineId backend=$backend scale=${scale}x out=${out.width}x${out.height} ms=${(System.nanoTime() - t0) / 1_000_000}"
    }

    fun setThemeMode(isDark: Boolean) {
        val density = resources.displayMetrics.density
        val cornerRadius = 8 * density
        val bgCol = if (isDark) Color.parseColor("#0B0F19") else Color.parseColor("#F8FAFC")
        val cardCol = if (isDark) Color.parseColor("#161F30") else Color.parseColor("#FFFFFF")
        val textCol = if (isDark) Color.parseColor("#F8FAFC") else Color.parseColor("#0F172A")
        val borderCol = if (isDark) Color.parseColor("#1E293B") else Color.parseColor("#CBD5E1")
        val subtextCol = if (isDark) Color.parseColor("#94A3B8") else Color.parseColor("#64748B")

        setBackgroundColor(bgCol)
        telemetryBar.setThemeColors(cardCol, borderCol, textCol)

        controlCard.background = GradientDrawable().apply {
            setColor(cardCol)
            this.cornerRadius = cornerRadius
            if (!isDark) setStroke((1 * density).toInt(), borderCol)
        }
        statusCard.background = GradientDrawable().apply {
            setColor(cardCol)
            this.cornerRadius = cornerRadius
            if (!isDark) setStroke((1 * density).toInt(), borderCol)
        }
        cardHeader.setTextColor(textCol)
        previewBadge.setTextColor(subtextCol)
        info.setTextColor(textCol)
    }
}

/**
 * Tab bar navigation installer: "Inpaint" and "Super Resolution" with sleek dark theme styling.
 */
object SuperResolutionTab {
    private var rootLayout: LinearLayout? = null
    private var barLayout: LinearLayout? = null
    private var btnInpaintRef: Button? = null
    private var btnSrRef: Button? = null
    private var isDarkState: Boolean = true
    private var isSrActive: Boolean = false

    fun setThemeMode(isDark: Boolean) {
        isDarkState = isDark
        val density = rootLayout?.resources?.displayMetrics?.density ?: 2f
        val bgCol = if (isDark) Color.parseColor("#0B0F19") else Color.parseColor("#F8FAFC")
        val inactiveCol = if (isDark) Color.parseColor("#161F30") else Color.parseColor("#E2E8F0")
        val activeCol = Color.parseColor("#2563EB")
        val inactiveText = if (isDark) Color.parseColor("#94A3B8") else Color.parseColor("#475569")
        val activeText = Color.WHITE

        rootLayout?.setBackgroundColor(bgCol)
        barLayout?.setBackgroundColor(bgCol)

        btnInpaintRef?.let { btn ->
            btn.setTextColor(if (!isSrActive) activeText else inactiveText)
            btn.background = GradientDrawable().apply {
                setColor(if (!isSrActive) activeCol else inactiveCol)
                cornerRadius = 8 * density
            }
        }
        btnSrRef?.let { btn ->
            btn.setTextColor(if (isSrActive) activeText else inactiveText)
            btn.background = GradientDrawable().apply {
                setColor(if (isSrActive) activeCol else inactiveCol)
                cornerRadius = 8 * density
            }
        }
    }

    fun install(activity: Activity, page: View) {
        val content = activity.findViewById<ViewGroup>(android.R.id.content)
        val original = content.getChildAt(0) ?: error("setContentView must be called before SuperResolutionTab.install")
        content.removeView(original)

        val match = ViewGroup.LayoutParams.MATCH_PARENT
        val density = activity.resources.displayMetrics.density

        val root = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#0B0F19"))
        }
        rootLayout = root

        val bar = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#0B0F19"))
            setPadding((8 * density).toInt(), (6 * density).toInt(), (8 * density).toInt(), (6 * density).toInt())
        }
        barLayout = bar

        fun createTabBtn(label: String, active: Boolean) = Button(activity).apply {
            text = label
            setTextColor(Color.WHITE)
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            textSize = 13f
            val gd = GradientDrawable().apply {
                setColor(if (active) Color.parseColor("#2563EB") else Color.parseColor("#161F30"))
                cornerRadius = 8 * density
            }
            background = gd
        }

        val inpaint = createTabBtn("🎨 Inpaint", true)
        val sr = createTabBtn("⚡ Super Resolution", false)
        btnInpaintRef = inpaint
        btnSrRef = sr

        val btnLp1 = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
            marginEnd = (4 * density).toInt()
        }
        val btnLp2 = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
            marginStart = (4 * density).toInt()
        }

        bar.addView(inpaint, btnLp1)
        bar.addView(sr, btnLp2)

        val body = FrameLayout(activity)
        body.addView(original, FrameLayout.LayoutParams(match, match))
        body.addView(page, FrameLayout.LayoutParams(match, match))
        page.visibility = View.GONE

        inpaint.setOnClickListener {
            isSrActive = false
            original.visibility = View.VISIBLE
            page.visibility = View.GONE
            setThemeMode(isDarkState)
        }

        sr.setOnClickListener {
            isSrActive = true
            original.visibility = View.GONE
            page.visibility = View.VISIBLE
            setThemeMode(isDarkState)
            (page as? SuperResolutionView)?.refreshSource()
        }

        root.addView(bar, LinearLayout.LayoutParams(match, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(body, LinearLayout.LayoutParams(match, 0, 1f))
        content.addView(root, ViewGroup.LayoutParams(match, match))
    }
}
