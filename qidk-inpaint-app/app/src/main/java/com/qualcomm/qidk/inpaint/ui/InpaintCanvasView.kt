package com.qualcomm.qidk.inpaint.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
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

    private val zoomBadgeBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(200, 30, 41, 59)
        style = Paint.Style.FILL
    }

    private val zoomBadgeText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#38BDF8")
        textSize = 24f
        textAlign = Paint.Align.RIGHT
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    // Touch Tracking
    private var lastX = 0f
    private var lastY = 0f
    private val canonicalPath = Path()

    // Bounding Box Selection (Supports both Two-Tap and Drag-to-Box)
    private var firstCorner: PointF? = null
    private var currentBBox: RectF? = null
    private var boxTouchDown: PointF? = null
    private var isDraggingBox = false
    private var dragPreviewBox: RectF? = null

    // Viewport Zoom & Pan
    private var zoomScale = 1.0f
    private var panX = 0f
    private var panY = 0f
    private var isMultiTouchZooming = false
    private var lastMultiTouchMidX = 0f
    private var lastMultiTouchMidY = 0f

    var onBoxSelectionStatusChanged: ((String) -> Unit)? = null
    var onMaskChanged: (() -> Unit)? = null

    private val scaleListener = object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            val prevScale = zoomScale
            zoomScale *= detector.scaleFactor
            zoomScale = zoomScale.coerceIn(1.0f, 6.0f)

            // Focal zoom anchor adjustment
            val fx = detector.focusX
            val fy = detector.focusY
            val factor = zoomScale / prevScale
            panX = (panX - fx) * factor + fx
            panY = (panY - fy) * factor + fy
            clampPan()

            invalidate()
            return true
        }
    }
    private val scaleDetector = ScaleGestureDetector(context, scaleListener)

    private val gestureDetector = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onDoubleTap(e: MotionEvent): Boolean {
            // Double-tap resets zoom & pan back to 1.0x
            resetZoom()
            return true
        }
    })

    init {
        clearMask(saveUndo = false)
    }

    fun resetZoom() {
        zoomScale = 1.0f
        panX = 0f
        panY = 0f
        invalidate()
    }

    fun zoomIn() {
        zoomScale = (zoomScale * 1.35f).coerceIn(1.0f, 6.0f)
        clampPan()
        invalidate()
    }

    fun zoomOut() {
        zoomScale = (zoomScale / 1.35f).coerceIn(1.0f, 6.0f)
        clampPan()
        invalidate()
    }

    fun getZoomLevel(): Float = zoomScale

    private fun clampPan() {
        val maxPanX = max(0f, (width * (zoomScale - 1f)) / 2f)
        val maxPanY = max(0f, (height * (zoomScale - 1f)) / 2f)
        if (zoomScale <= 1.0f) {
            panX = 0f
            panY = 0f
        } else {
            panX = panX.coerceIn(-maxPanX, maxPanX)
            panY = panY.coerceIn(-maxPanY, maxPanY)
        }
    }

    fun resetBoxSelection() {
        firstCorner = null
        currentBBox = null
        boxTouchDown = null
        isDraggingBox = false
        dragPreviewBox = null
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
        resetZoom()
        invalidate()
    }

    fun getSourceBitmap(): Bitmap? = sourceBitmap

    fun getMaskBitmap(): Bitmap {
        val outBitmap = Bitmap.createBitmap(canonicalSize, canonicalSize, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(canonicalSize * canonicalSize)
        maskBitmap.getPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        for (i in pixels.indices) {
            val c = pixels[i]
            val a = (c ushr 24) and 0xFF
            val r = (c ushr 16) and 0xFF
            val isHole = (a > 50 && r > 128)
            pixels[i] = if (isHole) Color.WHITE else Color.BLACK
        }
        outBitmap.setPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        return outBitmap
    }

    fun hasMask(): Boolean {
        val pixels = IntArray(canonicalSize * canonicalSize)
        maskBitmap.getPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        for (i in pixels.indices) {
            val c = pixels[i]
            val a = (c ushr 24) and 0xFF
            val r = (c ushr 16) and 0xFF
            if (a > 50 && r > 128) {
                return true
            }
        }
        return false
    }

    fun getSelectionBox(): RectF? = currentBBox

    fun applyGrabCutMask(binaryMask: Bitmap) {
        saveUndoState()
        val scaled = Bitmap.createScaledBitmap(binaryMask, canonicalSize, canonicalSize, false)
        val pixels = IntArray(canonicalSize * canonicalSize)
        scaled.getPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)

        for (i in pixels.indices) {
            val c = pixels[i]
            val a = (c ushr 24) and 0xFF
            val r = (c ushr 16) and 0xFF
            val isHole = (a > 50 && r > 128)
            pixels[i] = if (isHole) Color.WHITE else Color.TRANSPARENT
        }
        maskBitmap.setPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        resetBoxSelection()
        invalidate()
        onMaskChanged?.invoke()
    }

    fun clearMask(saveUndo: Boolean = true) {
        if (saveUndo) saveUndoState()
        maskBitmap.eraseColor(Color.TRANSPARENT)
        resetBoxSelection()
        invalidate()
        onMaskChanged?.invoke()
    }

    fun undo() {
        if (undoStack.isNotEmpty()) {
            val previous = undoStack.removeAt(undoStack.size - 1)
            maskBitmap = previous.copy(Bitmap.Config.ARGB_8888, true)
            maskCanvas = Canvas(maskBitmap)
            resetBoxSelection()
            invalidate()
            onMaskChanged?.invoke()
        }
    }

    private fun saveUndoState() {
        if (undoStack.size >= maxUndoSteps) {
            undoStack.removeAt(0)
        }
        undoStack.add(maskBitmap.copy(Bitmap.Config.ARGB_8888, false))
    }

    /**
     * Map View Screen coordinates (x, y) to 512x512 Canonical Image coordinates
     * taking into account dynamic zoom level and pan offsets.
     */
    private fun screenToCanonical(screenX: Float, screenY: Float): PointF {
        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW <= 0f || viewH <= 0f) return PointF(0f, 0f)

        val viewCenterX = viewW / 2f
        val viewCenterY = viewH / 2f

        // Invert pan and focal zoom
        val unscaledX = (screenX - panX - viewCenterX) / zoomScale + viewCenterX
        val unscaledY = (screenY - panY - viewCenterY) / zoomScale + viewCenterY

        val baseScaleX = canonicalSize.toFloat() / viewW
        val baseScaleY = canonicalSize.toFloat() / viewH

        val cx = unscaledX * baseScaleX
        val cy = unscaledY * baseScaleY

        return PointF(
            cx.coerceIn(0f, (canonicalSize - 1).toFloat()),
            cy.coerceIn(0f, (canonicalSize - 1).toFloat())
        )
    }

    /**
     * Prevents parent ScrollView from intercepting vertical touch drags.
     */
    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                if (event.pointerCount <= 1) {
                    parent?.requestDisallowInterceptTouchEvent(false)
                }
            }
        }
        return super.dispatchTouchEvent(event)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        gestureDetector.onTouchEvent(event)

        // 1. Two-finger Pinch-to-Zoom and Pan
        if (event.pointerCount >= 2) {
            isMultiTouchZooming = true
            scaleDetector.onTouchEvent(event)

            val p1x = event.getX(0); val p1y = event.getY(0)
            val p2x = event.getX(1); val p2y = event.getY(1)
            val midX = (p1x + p2x) / 2f
            val midY = (p1y + p2y) / 2f

            if (event.actionMasked == MotionEvent.ACTION_MOVE) {
                if (lastMultiTouchMidX != 0f && lastMultiTouchMidY != 0f) {
                    panX += (midX - lastMultiTouchMidX)
                    panY += (midY - lastMultiTouchMidY)
                    clampPan()
                    invalidate()
                }
            }
            lastMultiTouchMidX = midX
            lastMultiTouchMidY = midY
            return true
        }

        lastMultiTouchMidX = 0f
        lastMultiTouchMidY = 0f

        // If user just lifted a 2-finger gesture, don't draw with remaining finger
        if (isMultiTouchZooming) {
            if (event.actionMasked == MotionEvent.ACTION_UP || event.actionMasked == MotionEvent.ACTION_CANCEL) {
                isMultiTouchZooming = false
            }
            return true
        }

        // 2. Single-finger Brush and Box creation
        val pt = screenToCanonical(event.x, event.y)
        val cx = pt.x
        val cy = pt.y

        when (currentMode) {
            Mode.BRUSH -> {
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        saveUndoState()
                        canonicalPath.reset()
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
                        canonicalPath.reset()
                        invalidate()
                        onMaskChanged?.invoke()
                        return true
                    }
                }
            }
            Mode.BOUNDING_BOX -> {
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        boxTouchDown = PointF(cx, cy)
                        isDraggingBox = false
                        dragPreviewBox = null
                        if (firstCorner != null) {
                            val fc = firstCorner!!
                            dragPreviewBox = RectF(
                                min(fc.x, cx), min(fc.y, cy),
                                max(fc.x, cx), max(fc.y, cy)
                            )
                            invalidate()
                        }
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val down = boxTouchDown
                        if (firstCorner != null) {
                            val fc = firstCorner!!
                            dragPreviewBox = RectF(
                                min(fc.x, cx), min(fc.y, cy),
                                max(fc.x, cx), max(fc.y, cy)
                            )
                            invalidate()
                        } else if (down != null) {
                            val dx = abs(cx - down.x)
                            val dy = abs(cy - down.y)
                            if (dx >= 8f || dy >= 8f) {
                                isDraggingBox = true
                                dragPreviewBox = RectF(
                                    min(down.x, cx), min(down.y, cy),
                                    max(down.x, cx), max(down.y, cy)
                                )
                                invalidate()
                            }
                        }
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        val down = boxTouchDown
                        if (isDraggingBox && down != null) {
                            val left = min(down.x, cx)
                            val top = min(down.y, cy)
                            val right = max(down.x, cx)
                            val bottom = max(down.y, cy)

                            if (abs(right - left) >= 6f && abs(bottom - top) >= 6f) {
                                saveUndoState()
                                maskCanvas.drawRect(left, top, right, bottom, maskFillPaint)
                                currentBBox = RectF(left, top, right, bottom)
                                onBoxSelectionStatusChanged?.invoke("Square mask created (${(right - left).toInt()}x${(bottom - top).toInt()} px)")
                                onMaskChanged?.invoke()
                            }
                            resetBoxSelection()
                        } else if (firstCorner == null) {
                            firstCorner = PointF(cx, cy)
                            dragPreviewBox = null
                            isDraggingBox = false
                            onBoxSelectionStatusChanged?.invoke("Tap opposite corner to complete square mask")
                            invalidate()
                        } else {
                            val p1 = firstCorner!!
                            val left = min(p1.x, cx)
                            val top = min(p1.y, cy)
                            val right = max(p1.x, cx)
                            val bottom = max(p1.y, cy)

                            if (abs(right - left) >= 6f && abs(bottom - top) >= 6f) {
                                saveUndoState()
                                maskCanvas.drawRect(left, top, right, bottom, maskFillPaint)
                                currentBBox = RectF(left, top, right, bottom)
                                onBoxSelectionStatusChanged?.invoke("Square mask created (${(right - left).toInt()}x${(bottom - top).toInt()} px)")
                                onMaskChanged?.invoke()
                            }
                            resetBoxSelection()
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

        // 1. Save and apply Viewport Zoom & Pan transformations
        canvas.save()
        val viewCenterX = width / 2f
        val viewCenterY = height / 2f
        canvas.translate(panX, panY)
        canvas.scale(zoomScale, zoomScale, viewCenterX, viewCenterY)

        val dstRect = Rect(0, 0, width, height)

        // Source background photo
        sourceBitmap?.let {
            canvas.drawBitmap(it, null, dstRect, null)
        } ?: run {
            canvas.drawColor(Color.DKGRAY)
        }

        // Translucent red overlay mask
        canvas.drawBitmap(maskBitmap, null, dstRect, overlayPaint)

        // Bounding Box selection elements (rendered inside zoom space)
        if (currentMode == Mode.BOUNDING_BOX) {
            val viewScaleX = width.toFloat() / canonicalSize.toFloat()
            val viewScaleY = height.toFloat() / canonicalSize.toFloat()

            dragPreviewBox?.let { box ->
                val viewRect = RectF(
                    box.left * viewScaleX, box.top * viewScaleY,
                    box.right * viewScaleX, box.bottom * viewScaleY
                )
                canvas.drawRect(viewRect, bboxFillPaint)
                canvas.drawRect(viewRect, bboxPaint)
            }

            firstCorner?.let { p ->
                val fx = p.x * viewScaleX
                val fy = p.y * viewScaleY
                val crossSize = 28f
                canvas.drawLine(fx - crossSize, fy, fx + crossSize, fy, firstCornerPaint)
                canvas.drawLine(fx, fy - crossSize, fx, fy + crossSize, firstCornerPaint)
                canvas.drawCircle(fx, fy, 14f, firstCornerPaint)
                canvas.drawCircle(fx, fy, 5f, firstCornerFill)
            }
        }

        canvas.restore()

        // 2. Screen-space HUD & Banners (drawn without zoom distortion)
        if (zoomScale > 1.05f) {
            val zoomText = String.format("🔍 %.1fx • 2-finger pan • Double-tap reset", zoomScale)
            val badgeW = 460f
            val badgeH = 46f
            val badgeRect = RectF(width - badgeW - 16f, 16f, width - 16f, 16f + badgeH)
            canvas.drawRoundRect(badgeRect, 12f, 12f, zoomBadgeBg)
            canvas.drawText(zoomText, width - 30f, 48f, zoomBadgeText)
        }

        if (currentMode == Mode.BOUNDING_BOX) {
            if (firstCorner != null) {
                drawBanner(canvas, "📍 Corner 1 set. Tap opposite corner to create square mask")
            } else if (dragPreviewBox == null) {
                drawBanner(canvas, "🔲 Box Mode: Tap 2 corners or drag to create square mask")
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
