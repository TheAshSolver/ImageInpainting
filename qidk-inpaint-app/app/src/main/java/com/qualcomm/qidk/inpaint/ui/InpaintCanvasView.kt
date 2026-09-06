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

    // Touch Tracking
    private var lastX = 0f
    private var lastY = 0f
    private val currentPath = Path()
    private val canonicalPath = Path()

    // Bounding Box Selection
    private var bboxStart = PointF()
    private var currentBBox: RectF? = null

    init {
        clearMask(saveUndo = false)
    }

    fun setMode(mode: Mode) {
        currentMode = mode
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
        // Returns 512x512 single-channel / grayscale mask where 255=hole, 0=background
        val outBitmap = Bitmap.createBitmap(canonicalSize, canonicalSize, Bitmap.Config.ALPHA_8)
        val canvas = Canvas(outBitmap)
        val paint = Paint()
        canvas.drawBitmap(maskBitmap, 0f, 0f, paint)
        return outBitmap
    }

    fun getSelectionBox(): RectF? = currentBBox

    fun applyGrabCutMask(binaryMask: Bitmap) {
        saveUndoState()
        val scaled = Bitmap.createScaledBitmap(binaryMask, canonicalSize, canonicalSize, false)
        maskCanvas.drawBitmap(scaled, 0f, 0f, null)
        currentBBox = null
        invalidate()
    }

    fun clearMask(saveUndo: Boolean = true) {
        if (saveUndo) saveUndoState()
        maskBitmap.eraseColor(Color.TRANSPARENT)
        currentBBox = null
        invalidate()
    }

    fun undo() {
        if (undoStack.isNotEmpty()) {
            val previous = undoStack.removeAt(undoStack.size - 1)
            maskBitmap = previous.copy(Bitmap.Config.ARGB_8888, true)
            maskCanvas = Canvas(maskBitmap)
            currentBBox = null
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
                        bboxStart.set(cx, cy)
                        currentBBox = RectF(cx, cy, cx, cy)
                        invalidate()
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val left = min(bboxStart.x, cx)
                        val top = min(bboxStart.y, cy)
                        val right = max(bboxStart.x, cx)
                        val bottom = max(bboxStart.y, cy)
                        currentBBox = RectF(left, top, right, bottom)
                        invalidate()
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        invalidate()
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

        // 3. Draw Bounding Box Selection if active
        currentBBox?.let { box ->
            val viewScaleX = width.toFloat() / canonicalSize.toFloat()
            val viewScaleY = height.toFloat() / canonicalSize.toFloat()
            val viewRect = RectF(
                box.left * viewScaleX,
                box.top * viewScaleY,
                box.right * viewScaleX,
                box.bottom * viewScaleY
            )
            canvas.drawRect(viewRect, bboxFillPaint)
            canvas.drawRect(viewRect, bboxPaint)
        }
    }
}
