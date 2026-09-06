package com.qualcomm.qidk.inpaint.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class InpaintCanvasView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    enum class Mode {
        BRUSH,
        BOUNDING_BOX
    }

    private var currentMode = Mode.BRUSH
    private var sourceBitmap: Bitmap? = null

    // 512x512 Internal Canonical Resolution Mask
    private val canonicalSize = 512
    private var maskBitmap: Bitmap = Bitmap.createBitmap(canonicalSize, canonicalSize, Bitmap.Config.ARGB_8888)
    private var maskCanvas: Canvas = Canvas(maskBitmap)

    // History for Undo
    private val undoStack = mutableListOf<Bitmap>()
    private val maxUndoSteps = 10

    // Drawing Paints
    private val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        strokeWidth = 30f
    }

    private val maskFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    // On-Screen Translucent Overlay Paint (Red Tint)
    private val overlayPaint = Paint().apply {
        colorFilter = PorterDuffColorFilter(Color.argb(120, 255, 40, 40), PorterDuff.Mode.SRC_IN)
    }

    private val bboxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 5f
        pathEffect = DashPathEffect(floatArrayOf(15f, 15f), 0f)
    }

    private val bboxFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(60, 0, 255, 255)
        style = Paint.Style.FILL
    }

    private val cornerHandlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.FILL
    }

    private val cornerHandleStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val firstCornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val firstCornerFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.FILL
    }

    private val bannerBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(210, 15, 23, 42) // Dark Slate
        style = Paint.Style.FILL
    }

    private val bannerStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(160, 56, 189, 248) // Light Cyan
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }

    private val bannerTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 28f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    // Touch Tracking
    private var lastX = 0f
    private var lastY = 0f
    private val currentPath = Path()
    private val canonicalPath = Path()

    // Two-Tap Bounding Box Selection
    private var firstCorner: PointF? = null
    private var currentBBox: RectF? = null
    var onBoxSelectionStatusChanged: ((String) -> Unit)? = null

    init {
        clearMask(saveUndo = false)
    }

    fun resetBoxSelection() {
        firstCorner = null
        currentBBox = null
        invalidate()
    }

    fun setMode(mode: Mode) {
        currentMode = mode
        resetBoxSelection()
        invalidate()
    }

    fun getMode(): Mode = currentMode

    fun setBrushRadius(radius: Float) {
        maskPaint.strokeWidth = radius * 2f
    }

    fun setSourceBitmap(bitmap: Bitmap) {
        sourceBitmap = Bitmap.createScaledBitmap(bitmap, canonicalSize, canonicalSize, true)
        clearMask(saveUndo = false)
        invalidate()
    }

    fun getSourceBitmap(): Bitmap? = sourceBitmap

    fun getMaskBitmap(): Bitmap {
        // Returns 512x512 mask where 255=hole, 0=background
        val outBitmap = Bitmap.createBitmap(canonicalSize, canonicalSize, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(canonicalSize * canonicalSize)
        maskBitmap.getPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        for (i in pixels.indices) {
            val c = pixels[i]
            val a = Color.alpha(c)
            val r = Color.red(c)
            val isHole = (a > 50 && r > 128)
            pixels[i] = if (isHole) Color.WHITE else Color.BLACK
        }
        outBitmap.setPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        return outBitmap
    }

    fun getSelectionBox(): RectF? = currentBBox

    fun applyGrabCutMask(binaryMask: Bitmap) {
        saveUndoState()
        val scaled = Bitmap.createScaledBitmap(binaryMask, canonicalSize, canonicalSize, false)
        val pixels = IntArray(canonicalSize * canonicalSize)
        scaled.getPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)

        for (i in pixels.indices) {
            val c = pixels[i]
            val a = Color.alpha(c)
            val r = Color.red(c)
            val isHole = (a > 50 && r > 128)
            pixels[i] = if (isHole) Color.WHITE else Color.TRANSPARENT
        }
        maskBitmap.setPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        resetBoxSelection()
        invalidate()
    }

    fun clearMask(saveUndo: Boolean = true) {
        if (saveUndo) saveUndoState()
        maskBitmap.eraseColor(Color.TRANSPARENT)
        resetBoxSelection()
        invalidate()
    }

    fun undo() {
        if (undoStack.isNotEmpty()) {
            val previous = undoStack.removeAt(undoStack.size - 1)
            maskBitmap = previous.copy(Bitmap.Config.ARGB_8888, true)
            maskCanvas = Canvas(maskBitmap)
            resetBoxSelection()
            invalidate()
        }
    }

    private fun saveUndoState() {
        if (undoStack.size >= maxUndoSteps) {
            undoStack.removeAt(0)
        }
        undoStack.add(maskBitmap.copy(Bitmap.Config.ARGB_8888, false))
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x
        val y = event.y

        // Map View coordinates to 512x512 canonical coordinates
        val scaleX = canonicalSize.toFloat() / width.toFloat()
        val scaleY = canonicalSize.toFloat() / height.toFloat()
        val cx = x * scaleX
        val cy = y * scaleY

        when (currentMode) {
            Mode.BRUSH -> {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        saveUndoState()
                        currentPath.reset()
                        canonicalPath.reset()
                        currentPath.moveTo(x, y)
                        canonicalPath.moveTo(cx, cy)
                        lastX = cx
                        lastY = cy
                        maskCanvas.drawCircle(cx, cy, maskPaint.strokeWidth / 2f, maskFillPaint)
                        invalidate()
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val dx = abs(cx - lastX)
                        val dy = abs(cy - lastY)
                        if (dx >= 2f || dy >= 2f) {
                            canonicalPath.quadTo(lastX, lastY, (cx + lastX) / 2f, (cy + lastY) / 2f)
                            lastX = cx
                            lastY = cy
                            maskCanvas.drawPath(canonicalPath, maskPaint)
                            invalidate()
                        }
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        canonicalPath.lineTo(cx, cy)
                        maskCanvas.drawPath(canonicalPath, maskPaint)
                        currentPath.reset()
                        canonicalPath.reset()
                        invalidate()
                        return true
                    }
                }
            }
            Mode.BOUNDING_BOX -> {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        if (firstCorner == null) {
                            firstCorner = PointF(cx, cy)
                            currentBBox = null
                            onBoxSelectionStatusChanged?.invoke("Tap opposite corner to complete box")
                            invalidate()
                        } else {
                            val p1 = firstCorner!!
                            val x1 = p1.x
                            val y1 = p1.y
                            val x2 = cx
                            val y2 = cy

                            val left = min(x1, x2)
                            val top = min(y1, y2)
                            val right = max(x1, x2)
                            val bottom = max(y1, y2)

                            if (abs(right - left) >= 8f && abs(bottom - top) >= 8f) {
                                currentBBox = RectF(left, top, right, bottom)
                                firstCorner = null
                                onBoxSelectionStatusChanged?.invoke("Box selected (${(right - left).toInt()}x${(bottom - top).toInt()} px). Tap 'Auto GrabCut'")
                                invalidate()
                            } else {
                                firstCorner = PointF(cx, cy)
                                onBoxSelectionStatusChanged?.invoke("Tap opposite corner to complete box")
                                invalidate()
                            }
                        }
                        return true
                    }
                }
            }
        }
        return super.onTouchEvent(event)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val dstRect = Rect(0, 0, width, height)

        // 1. Draw source background photo
        sourceBitmap?.let {
            canvas.drawBitmap(it, null, dstRect, null)
        } ?: run {
            canvas.drawColor(Color.DKGRAY)
        }

        // 2. Draw translucent red overlay mask
        canvas.drawBitmap(maskBitmap, null, dstRect, overlayPaint)

        // 3. Draw Two-Tap Bounding Box UI if active
        if (currentMode == Mode.BOUNDING_BOX) {
            val viewScaleX = width.toFloat() / canonicalSize.toFloat()
            val viewScaleY = height.toFloat() / canonicalSize.toFloat()

            // A. Point 1 set: draw cyan crosshair & circle
            firstCorner?.let { p ->
                val fx = p.x * viewScaleX
                val fy = p.y * viewScaleY
                val crossSize = 28f
                canvas.drawLine(fx - crossSize, fy, fx + crossSize, fy, firstCornerPaint)
                canvas.drawLine(fx, fy - crossSize, fx, fy + crossSize, firstCornerPaint)
                canvas.drawCircle(fx, fy, 14f, firstCornerPaint)
                canvas.drawCircle(fx, fy, 5f, firstCornerFill)

                drawBanner(canvas, "📍 Corner 1 set. Tap opposite corner to complete box")
            }

            // B. Full box formed: draw translucent fill, dashed outline, and corner handles
            currentBBox?.let { box ->
                val viewRect = RectF(
                    box.left * viewScaleX,
                    box.top * viewScaleY,
                    box.right * viewScaleX,
                    box.bottom * viewScaleY
                )
                canvas.drawRect(viewRect, bboxFillPaint)
                canvas.drawRect(viewRect, bboxPaint)

                // Draw corner handles
                val handleRadius = 10f
                val corners = arrayOf(
                    PointF(viewRect.left, viewRect.top),
                    PointF(viewRect.right, viewRect.top),
                    PointF(viewRect.left, viewRect.bottom),
                    PointF(viewRect.right, viewRect.bottom)
                )
                for (pt in corners) {
                    canvas.drawCircle(pt.x, pt.y, handleRadius, cornerHandlePaint)
                    canvas.drawCircle(pt.x, pt.y, handleRadius, cornerHandleStroke)
                }

                drawBanner(canvas, "✅ Box: ${box.width().toInt()}×${box.height().toInt()} px — Tap '⚡ Auto GrabCut'")
            }

            // C. Initial prompt if no tap recorded yet
            if (firstCorner == null && currentBBox == null) {
                drawBanner(canvas, "🔲 Box Mode: Tap 1st corner of unwanted object")
            }
        }
    }

    private fun drawBanner(canvas: Canvas, text: String) {
        val bannerHeight = 56f
        val margin = 16f
        val bannerRect = RectF(margin, margin, width - margin, margin + bannerHeight)
        canvas.drawRoundRect(bannerRect, 14f, 14f, bannerBgPaint)
        canvas.drawRoundRect(bannerRect, 14f, 14f, bannerStrokePaint)
        canvas.drawText(text, width / 2f, margin + 38f, bannerTextPaint)
    }
}
