package qidk.fix

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.LinearLayout
import androidx.core.content.ContextCompat
import com.google.android.material.button.MaterialButton
import com.qualcomm.qidk.inpaint.R
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Down-sampled copies (1/2, 1/4, ...) of one bitmap. A 4096 px output drawn into a 330 px pane with plain bilinear sampling
 * aliases badly (every 12th source pixel is read); drawing from the level whose scale is >= 0.5..1 keeps it clean.
 */
object MipPyramid {
    /** Level whose pixels are closest to 1:1 on screen without magnifying: scale = view pixels per bitmap pixel. */
    fun levelFor(scale: Float, levels: Int): Int {
        if (levels <= 1 || scale >= 0.75f || scale <= 0f) return 0
        return min(levels - 1, floor(ln(1.0 / scale) / ln(2.0)).toInt())
    }

    /** Level 0 is [src] itself (not copied); the pyramid stops when the shorter side would drop below 256 px. */
    fun build(src: Bitmap): List<Bitmap> {
        val out = ArrayList<Bitmap>()
        out += src
        var cur = src
        while (min(cur.width, cur.height) >= 512) {
            cur = Bitmap.createScaledBitmap(cur, cur.width / 2, cur.height / 2, true)
            out += cur
        }
        return out
    }
}

/** One finger pans, two fingers pinch and pan, double tap toggles fit / 4x. All of it goes into the shared state. */
class ViewportTouchController(
    context: Context,
    private val shared: SharedViewport,
    private val geometry: SharedViewport.Geometry
) {
    private var lastX = 0f
    private var lastY = 0f
    private var tracking = false

    private val scale = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            shared.zoomAbout(geometry, d.scaleFactor, d.focusX, d.focusY)
            return true
        }
    })
    private val taps = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDoubleTap(e: MotionEvent): Boolean { shared.toggleZoomAt(geometry, e.x, e.y); return true }
    })

    fun onTouchEvent(view: View, ev: MotionEvent): Boolean {
        if (ev.actionMasked == MotionEvent.ACTION_DOWN) view.parent?.requestDisallowInterceptTouchEvent(true)
        scale.onTouchEvent(ev)
        taps.onTouchEvent(ev)
        // Mean position of the pointers that stay down: when a finger lifts, the reference is re-anchored to the others,
        // so the picture never jumps (the old per-view handler kept the lifted finger as reference).
        val skip = if (ev.actionMasked == MotionEvent.ACTION_POINTER_UP) ev.actionIndex else -1
        var sx = 0f; var sy = 0f; var n = 0
        for (i in 0 until ev.pointerCount) if (i != skip) { sx += ev.getX(i); sy += ev.getY(i); n++ }
        if (n == 0) { tracking = false; return true }
        val x = sx / n; val y = sy / n
        when (ev.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN, MotionEvent.ACTION_POINTER_UP -> { lastX = x; lastY = y; tracking = true }
            MotionEvent.ACTION_MOVE -> if (tracking) { shared.panBy(geometry, x - lastX, y - lastY); lastX = x; lastY = y }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> { tracking = false; view.parent?.requestDisallowInterceptTouchEvent(false) }
        }
        return true
    }
}

/** Draws one image through the [SharedViewport]; owns no transform of its own. */
class ImagePaneView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {
    private var shared: SharedViewport? = null
    private var geometry: SharedViewport.Geometry? = null
    private var controller: ViewportTouchController? = null
    private var levels: List<Bitmap> = emptyList()
    private val listener = SharedViewport.Listener { _, _, _ -> postInvalidateOnAnimation() }

    var badge: String = ""
        set(v) { field = v; invalidate() }
    var badgeAccent: Int = ContextCompat.getColor(context, R.color.qidk_cyan)

    private val density = resources.displayMetrics.density
    private val wellPaint = Paint().apply { color = ContextCompat.getColor(context, R.color.qidk_well) }
    private val imagePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val badgeBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = ContextCompat.getColor(context, R.color.qidk_badge_bg) }
    private val badgeText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.qidk_badge_text); textSize = 12f * density; typeface = Typeface.DEFAULT_BOLD
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val dest = FloatArray(4)
    private val srcRect = Rect()
    private val dstRect = RectF()
    private val badgeRect = RectF()

    fun bind(sv: SharedViewport) {
        geometry?.let { g -> shared?.let { it.unregister(g); it.removeListener(listener) } }
        shared = sv
        val g = sv.register()
        geometry = g
        g.viewW = width.toFloat(); g.viewH = height.toFloat()
        bitmap()?.let { g.bmpW = it.width.toFloat(); g.bmpH = it.height.toFloat() }
        sv.addListener(listener)
        controller = ViewportTouchController(context, sv, g)
    }

    private fun bitmap(): Bitmap? = levels.firstOrNull()

    /** Verification aid: [x0, y0, x1, y1] of the part of the image this pane shows, in normalised image coordinates (0..1 = whole image). */
    fun visibleNorm(): FloatArray? {
        val sv = shared ?: return null
        val g = geometry ?: return null
        if (!g.valid) return null
        return floatArrayOf(sv.toNormX(g, 0f), sv.toNormY(g, 0f), sv.toNormX(g, width.toFloat()), sv.toNormY(g, height.toFloat()))
    }

    fun setImage(bitmap: Bitmap?, pyramid: List<Bitmap>? = null) {
        levels = if (bitmap == null) emptyList() else (pyramid ?: MipPyramid.build(bitmap))
        geometry?.let { if (bitmap != null) { it.bmpW = bitmap.width.toFloat(); it.bmpH = bitmap.height.toFloat() } }
        shared?.let { it.set(it.zoom, it.centerX, it.centerY) }      // re-clamp for the new aspect ratio
        invalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, ow: Int, oh: Int) {
        geometry?.let { it.viewW = w.toFloat(); it.viewH = h.toFloat() }
        shared?.let { it.set(it.zoom, it.centerX, it.centerY) }
    }

    override fun onDetachedFromWindow() {
        geometry?.let { g -> shared?.let { it.unregister(g); it.removeListener(listener) } }
        geometry = null; shared = null
        super.onDetachedFromWindow()
    }

    override fun onTouchEvent(e: MotionEvent): Boolean = controller?.onTouchEvent(this, e) ?: super.onTouchEvent(e)

    override fun onDraw(c: Canvas) {
        c.drawRect(0f, 0f, width.toFloat(), height.toFloat(), wellPaint)
        val sv = shared; val g = geometry; val base = bitmap()
        if (sv != null && g != null && base != null && g.valid) {
            sv.destRect(g, dest)
            val scale = sv.pixelScale(g)
            val lvl = MipPyramid.levelFor(scale, levels.size)
            val src = levels[lvl]
            val full = dest[2] - dest[0]; val fullH = dest[3] - dest[1]
            val vl = max(0f, dest[0]); val vt = max(0f, dest[1]); val vr = min(width.toFloat(), dest[2]); val vb = min(height.toFloat(), dest[3])
            if (vr > vl && vb > vt && full > 0f && fullH > 0f) {
                val x0 = floor((vl - dest[0]) / full * src.width).toInt().coerceIn(0, src.width - 1)
                val y0 = floor((vt - dest[1]) / fullH * src.height).toInt().coerceIn(0, src.height - 1)
                val x1 = ceil((vr - dest[0]) / full * src.width).toInt().coerceIn(x0 + 1, src.width)
                val y1 = ceil((vb - dest[1]) / fullH * src.height).toInt().coerceIn(y0 + 1, src.height)
                srcRect.set(x0, y0, x1, y1)
                // destination of exactly that source rectangle: no sub-pixel shift between zoom levels
                dstRect.set(dest[0] + x0.toFloat() / src.width * full, dest[1] + y0.toFloat() / src.height * fullH,
                    dest[0] + x1.toFloat() / src.width * full, dest[1] + y1.toFloat() / src.height * fullH)
                imagePaint.isFilterBitmap = scale * (1 shl lvl) < 2.5f     // real pixels when magnified, smooth when shrunk
                c.drawBitmap(src, srcRect, dstRect, imagePaint)
            }
            if (sv.zoom > 1.01f) {
                val t = String.format(java.util.Locale.US, "x%.1f", sv.zoom)
                val w = badgeText.measureText(t)
                badgeRect.set(width - w - 22 * density, height - 34 * density, width - 8 * density, height - 8 * density)
                c.drawRoundRect(badgeRect, 10 * density, 10 * density, badgeBg)
                c.drawText(t, badgeRect.left + 7 * density, badgeRect.bottom - 8 * density, badgeText)
            }
        }
        if (badge.isNotEmpty()) {
            val w = badgeText.measureText(badge)
            badgeRect.set(10 * density, 10 * density, 10 * density + w + 36 * density, 10 * density + 28 * density)
            c.drawRoundRect(badgeRect, 10 * density, 10 * density, badgeBg)
            dotPaint.color = badgeAccent
            c.drawCircle(badgeRect.left + 12 * density, badgeRect.centerY(), 4 * density, dotPaint)
            c.drawText(badge, badgeRect.left + 24 * density, badgeRect.centerY() + 4.2f * density, badgeText)
        }
    }
}

/**
 * Two vertically stacked viewports (input on top, output below) on ONE [SharedViewport], plus the floating "1:1 Reset".
 * A gesture on either pane moves both; the output may be 2, 4 or 8 times larger than the input, the scene region stays identical.
 */
class DualViewport @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null) : FrameLayout(context, attrs) {
    val shared = SharedViewport()
    val top = ImagePaneView(context)
    val bottom = ImagePaneView(context)
    private lateinit var reset: MaterialButton
    private val syncListener = SharedViewport.Listener { _, _, _ -> reset.visibility = if (shared.isTransformed) VISIBLE else GONE }

    init {
        val d = resources.displayMetrics.density
        val column = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
        column.addView(top, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        column.addView(View(context), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (6 * d).toInt()))
        column.addView(bottom, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        addView(column, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        top.bind(shared); bottom.bind(shared)
        top.badgeAccent = ContextCompat.getColor(context, R.color.qidk_cyan)
        bottom.badgeAccent = ContextCompat.getColor(context, R.color.qidk_emerald)
        reset = MaterialButton(context).apply {
            text = "1:1 Reset"
            isAllCaps = false
            textSize = 12f
            insetTop = 0; insetBottom = 0
            backgroundTintList = ContextCompat.getColorStateList(context, R.color.qidk_primary_action)
            setTextColor(ContextCompat.getColor(context, R.color.qidk_on_accent))
            visibility = GONE
            setOnClickListener { shared.reset() }
        }
        addView(reset, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply {
            gravity = android.view.Gravity.BOTTOM or android.view.Gravity.END
            setMargins(0, 0, (12 * d).toInt(), (12 * d).toInt())
        })
        shared.addListener(syncListener)
    }

    /** Verification aid: shared state, what each pane shows (normalised) and where the top pane is on screen (for adb input swipes). */
    fun describe(): String {
        fun f(a: FloatArray?) = if (a == null) "none" else String.format(java.util.Locale.US, "[%.4f,%.4f,%.4f,%.4f]", a[0], a[1], a[2], a[3])
        val loc = IntArray(2)
        top.getLocationOnScreen(loc)
        return String.format(java.util.Locale.US, "zoom=%.3f center=(%.4f,%.4f) top=%s bottom=%s topScreen=[%d,%d,%d,%d] topImg=%s bottomImg=%s",
            shared.zoom, shared.centerX, shared.centerY, f(top.visibleNorm()), f(bottom.visibleNorm()),
            loc[0], loc[1], loc[0] + top.width, loc[1] + top.height, top.badge.replace("  ", " "), bottom.badge.replace("  ", " "))
    }

    fun setImages(input: Bitmap?, output: Bitmap?, outputPyramid: List<Bitmap>? = null) {
        top.badge = if (input != null) "INPUT  ${input.width}x${input.height}" else "INPUT"
        bottom.badge = if (output != null) "OUTPUT  ${output.width}x${output.height}" else "OUTPUT"
        top.setImage(input)
        bottom.setImage(output, outputPyramid)
        shared.reset()
    }
}
