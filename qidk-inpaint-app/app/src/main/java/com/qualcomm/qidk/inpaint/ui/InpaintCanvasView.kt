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

    // Two-Tap and Drag Bounding Box Selection
    private var firstCorner: PointF? = null
    private var currentBBox: RectF? = null
    private var dragPreviewBox: RectF? = null
    private var boxTouchDown: PointF? = null
    var onBoxSelectionStatusChanged: ((String) -> Unit)? = null
    var onMaskChanged: (() -> Unit)? = null

    init {
        clearMask(saveUndo = false)
    }

    fun resetBoxSelection() {
        firstCorner = null
        currentBBox = null
        dragPreviewBox = null
        boxTouchDown = null
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

    private var cachedProcessedMask: Bitmap? = null
    private var isMaskDirty: Boolean = true

    fun setSourceBitmap(bitmap: Bitmap) {
        sourceBitmap = if (bitmap.width == canonicalSize && bitmap.height == canonicalSize) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, canonicalSize, canonicalSize, true)
        }
        clearMask(saveUndo = false)
        invalidate()
    }

    fun getSourceBitmap(): Bitmap? = sourceBitmap

    fun getMaskBitmap(): Bitmap {
        if (!isMaskDirty && cachedProcessedMask != null) {
            return cachedProcessedMask!!
        }
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
        cachedProcessedMask = outBitmap
        isMaskDirty = false
        return outBitmap
    }

    fun getSelectionBox(): RectF? = currentBBox

    fun getRawMaskBitmap(): Bitmap {
        return maskBitmap.copy(Bitmap.Config.ARGB_8888, false)
    }

    fun setRawMaskBitmap(bitmap: Bitmap) {
        val scaled = Bitmap.createScaledBitmap(bitmap, canonicalSize, canonicalSize, false)
        maskBitmap = scaled.copy(Bitmap.Config.ARGB_8888, true)
        maskCanvas = Canvas(maskBitmap)
        resetBoxSelection()
        isMaskDirty = true
        invalidate()
        onMaskChanged?.invoke()
    }

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
        isMaskDirty = true
        invalidate()
        onMaskChanged?.invoke()
    }

    fun clearMask(saveUndo: Boolean = true) {
        if (saveUndo) saveUndoState()
        maskBitmap.eraseColor(Color.TRANSPARENT)
        resetBoxSelection()
        isMaskDirty = true
        invalidate()
        onMaskChanged?.invoke()
    }

    fun undo() {
        if (undoStack.isNotEmpty()) {
            val previous = undoStack.removeAt(undoStack.size - 1)
            maskBitmap = previous.copy(Bitmap.Config.ARGB_8888, true)
            maskCanvas = Canvas(maskBitmap)
            resetBoxSelection()
            isMaskDirty = true
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

    override fun dispatchTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                parent?.requestDisallowInterceptTouchEvent(true)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                parent?.requestDisallowInterceptTouchEvent(false)
            }
        }
        return super.dispatchTouchEvent(event)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        parent?.requestDisallowInterceptTouchEvent(true)
        val x = event.x
        val y = event.y

        // Map View coordinates to 512x512 canonical coordinates
        val scaleX = canonicalSize.toFloat() / width.toFloat()
        val scaleY = canonicalSize.toFloat() / height.toFloat()
        val cx = (x * scaleX).coerceIn(0f, (canonicalSize - 1).toFloat())
        val cy = (y * scaleY).coerceIn(0f, (canonicalSize - 1).toFloat())

        when (currentMode) {
            Mode.BRUSH -> {
                when (event.actionMasked) {
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
                        isMaskDirty = true
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
                        if (firstCorner != null) {
                            val fc = firstCorner!!
                            dragPreviewBox = RectF(
                                min(fc.x, cx), min(fc.y, cy),
                                max(fc.x, cx), max(fc.y, cy)
                            )
                            invalidate()
                        } else if (boxTouchDown != null) {
                            val down = boxTouchDown!!
                            val dx = abs(cx - down.x)
                            val dy = abs(cy - down.y)
                            if (dx >= 8f || dy >= 8f) {
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
                        val preview = dragPreviewBox
                        if (preview != null && preview.width() >= 10f && preview.height() >= 10f) {
                            currentBBox = RectF(preview)
                            saveUndoState()
                            maskCanvas.drawRect(currentBBox!!, maskFillPaint)
                            isMaskDirty = true
                            firstCorner = null
                            dragPreviewBox = null
                            boxTouchDown = null
                            val bw = currentBBox!!.width().toInt()
                            val bh = currentBBox!!.height().toInt()
                            onBoxSelectionStatusChanged?.invoke("Box masked (${bw}x${bh} px). Ready to inpaint")
                            onMaskChanged?.invoke()
                            invalidate()
                            return true
                        }

                        dragPreviewBox = null
                        boxTouchDown = null

                        if (firstCorner == null) {
                            firstCorner = PointF(cx, cy)
                            currentBBox = null
                            onBoxSelectionStatusChanged?.invoke("Corner 1 set. Tap opposite corner to complete box")
                            invalidate()
                        } else {
                            val p1 = firstCorner!!
                            val left = min(p1.x, cx)
                            val top = min(p1.y, cy)
                            val right = max(p1.x, cx)
                            val bottom = max(p1.y, cy)

                            if (abs(right - left) >= 10f && abs(bottom - top) >= 10f) {
                                currentBBox = RectF(left, top, right, bottom)
                                saveUndoState()
                                maskCanvas.drawRect(currentBBox!!, maskFillPaint)
                                isMaskDirty = true
                                firstCorner = null
                                val bw = (right - left).toInt()
                                val bh = (bottom - top).toInt()
                                onBoxSelectionStatusChanged?.invoke("Box masked (${bw}x${bh} px). Ready to inpaint")
                                onMaskChanged?.invoke()
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

        // 3. Draw Bounding Box UI if in Bounding Box mode
        if (currentMode == Mode.BOUNDING_BOX) {
            val viewScaleX = width.toFloat() / canonicalSize.toFloat()
            val viewScaleY = height.toFloat() / canonicalSize.toFloat()

            val activeBox = dragPreviewBox ?: currentBBox
            if (activeBox != null) {
                // Draw full box: translucent fill, dashed outline, and corner handles
                val viewRect = RectF(
                    activeBox.left * viewScaleX,
                    activeBox.top * viewScaleY,
                    activeBox.right * viewScaleX,
                    activeBox.bottom * viewScaleY
                )
                canvas.drawRect(viewRect, bboxFillPaint)
                canvas.drawRect(viewRect, bboxPaint)

                // Corner handles
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

                if (dragPreviewBox != null) {
                    drawBanner(canvas, "📐 Sizing box: ${activeBox.width().toInt()}×${activeBox.height().toInt()} px")
                } else {
                    drawBanner(canvas, "✅ Box masked: ${activeBox.width().toInt()}×${activeBox.height().toInt()} px — Ready to inpaint")
                }
            } else if (firstCorner != null) {
                // Corner 1 set: draw cyan crosshair & circle
                val p = firstCorner!!
                val fx = p.x * viewScaleX
                val fy = p.y * viewScaleY
                val crossSize = 28f
                canvas.drawLine(fx - crossSize, fy, fx + crossSize, fy, firstCornerPaint)
                canvas.drawLine(fx, fy - crossSize, fx, fy + crossSize, firstCornerPaint)
                canvas.drawCircle(fx, fy, 14f, firstCornerPaint)
                canvas.drawCircle(fx, fy, 5f, firstCornerFill)

                drawBanner(canvas, "📍 Corner 1 set. Tap opposite corner to complete box")
            } else {
                // Initial prompt
                drawBanner(canvas, "🔲 Box Mode: Drag across object or tap 2 corners")
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
