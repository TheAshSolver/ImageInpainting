package com.qualcomm.qidk.inpaint.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import com.qualcomm.qidk.inpaint.segmentation.AutoGrabCutEngine
import com.qualcomm.qidk.inpaint.segmentation.MediaPipeSegmenter
import com.qualcomm.qidk.inpaint.utils.MagicTouchEngine
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class InpaintCanvasView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    enum class Mode {
        CIRCLE_SELECT,
        BRUSH,
        BOUNDING_BOX
    }

    enum class SegmentationEngine {
        AUTO_GRABCUT,     // 4-state Trimap GMM Graph-Cut (Boykov-Kolmogorov) - Expands to whole object
        AI_MAGIC_TOUCH,   // Promptable Vision Segmentation (MediaPipe Interactive Segmenter)
        EXACT_SELECT      // Exact Area: Strictly masks precisely the drawn circle/lasso interior (no expansion)
    }

    private var currentMode = Mode.CIRCLE_SELECT
    private var activeSegmentationEngine = SegmentationEngine.AUTO_GRABCUT
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

    // Circle to Select (Magic Touch) State & Paints
    private val circlePoints = mutableListOf<PointF>()
    private val circlePath = Path()
    private var isDrawingCircle = false

    // Affine Transformation Matrices for strict 512x512 Viewport Mapping
    private val bitmapToViewMatrix = Matrix()
    private val viewToBitmapMatrix = Matrix()

    // Visual Touch Feedback Trail (Glowing path with smooth fade-out)
    data class TrailPoint(val x: Float, val y: Float, val timestamp: Long)
    private val touchTrail = mutableListOf<TrailPoint>()

    private val trailGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(120, 255, 255, 255) // Semi-opaque white glow
        style = Paint.Style.STROKE
        strokeWidth = 18f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val trailCorePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 7f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    // Inline Thinking Pulse State
    private var isThinkingPulse = false
    private var thinkingPulseStartTime = 0L
    private var thinkingPulseCenter: PointF? = null

    private val circleStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#C084FC") // Neon Purple / Lavender
        style = Paint.Style.STROKE
        strokeWidth = 4f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        pathEffect = DashPathEffect(floatArrayOf(12f, 8f), 0f)
    }

    private val circleGlowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#44A855F7") // Translucent Neon Purple Glow
        style = Paint.Style.STROKE
        strokeWidth = 10f
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    private val circleFillPreviewPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.argb(45, 168, 85, 247) // Gentle translucent fill preview
        style = Paint.Style.FILL
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
    var onCircleSelectionStatusChanged: ((String) -> Unit)? = null
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

    fun resetCircleSelection() {
        circlePoints.clear()
        circlePath.reset()
        isDrawingCircle = false
        invalidate()
    }

    fun setMode(mode: Mode) {
        currentMode = mode
        resetBoxSelection()
        resetCircleSelection()
        invalidate()
    }

    fun getMode(): Mode = currentMode

    fun setSegmentationEngine(engine: SegmentationEngine) {
        activeSegmentationEngine = engine
        invalidate()
    }

    fun getSegmentationEngine(): SegmentationEngine = activeSegmentationEngine

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

    fun getRawMaskBitmap(): Bitmap {
        return maskBitmap.copy(Bitmap.Config.ARGB_8888, false)
    }

    fun setRawMaskBitmap(bitmap: Bitmap) {
        val scaled = Bitmap.createScaledBitmap(bitmap, canonicalSize, canonicalSize, false)
        maskBitmap = scaled.copy(Bitmap.Config.ARGB_8888, true)
        maskCanvas = Canvas(maskBitmap)
        resetBoxSelection()
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
            val a = (c ushr 24) and 0xFF
            val r = (c ushr 16) and 0xFF
            val isHole = (a > 50 && r > 128)
            pixels[i] = if (isHole) Color.WHITE else Color.TRANSPARENT
        }
        maskBitmap.setPixels(pixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        resetBoxSelection()
        resetCircleSelection()
        invalidate()
        onMaskChanged?.invoke()
    }

    fun mergeMaskIntoCanvas(newMask: Bitmap) {
        saveUndoState()
        val scaled = Bitmap.createScaledBitmap(newMask, canonicalSize, canonicalSize, false)
        val newPixels = IntArray(canonicalSize * canonicalSize)
        val currentPixels = IntArray(canonicalSize * canonicalSize)
        scaled.getPixels(newPixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        maskBitmap.getPixels(currentPixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)

        for (i in newPixels.indices) {
            val np = newPixels[i]
            val isNewHole = ((np ushr 24) > 50 && ((np ushr 16) and 0xFF) > 128)
            if (isNewHole) {
                currentPixels[i] = Color.WHITE
            }
        }
        maskBitmap.setPixels(currentPixels, 0, canonicalSize, 0, 0, canonicalSize, canonicalSize)
        resetBoxSelection()
        resetCircleSelection()
        invalidate()
        onMaskChanged?.invoke()
    }

    fun clearMask(saveUndo: Boolean = true) {
        if (saveUndo) saveUndoState()
        maskBitmap.eraseColor(Color.TRANSPARENT)
        resetBoxSelection()
        resetCircleSelection()
        invalidate()
        onMaskChanged?.invoke()
    }

    fun undo() {
        if (undoStack.isNotEmpty()) {
            val previous = undoStack.removeAt(undoStack.size - 1)
            maskBitmap = previous.copy(Bitmap.Config.ARGB_8888, true)
            maskCanvas = Canvas(maskBitmap)
            resetBoxSelection()
            resetCircleSelection()
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
     * Viewport-to-Bitmap Affine Matrix Transforms:
     * Calculates uniform aspect-ratio scale ('fitCenter') into intrinsic 512x512 space,
     * coupled with focal zoom and pan, and inverts the matrix to map screen touch coordinates
     * precisely to underlying bitmap pixels.
     */
    private fun updateTransformMatrices() {
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0f || vh <= 0f) return

        val scale = minOf(vw / canonicalSize.toFloat(), vh / canonicalSize.toFloat())
        val dx = (vw - canonicalSize * scale) / 2f
        val dy = (vh - canonicalSize * scale) / 2f

        bitmapToViewMatrix.reset()
        bitmapToViewMatrix.postScale(scale, scale)
        bitmapToViewMatrix.postTranslate(dx, dy)
        bitmapToViewMatrix.postScale(zoomScale, zoomScale, vw / 2f, vh / 2f)
        bitmapToViewMatrix.postTranslate(panX, panY)

        bitmapToViewMatrix.invert(viewToBitmapMatrix)
    }

    fun mapTouchToBitmap(touchX: Float, touchY: Float): PointF {
        updateTransformMatrices()
        val pts = floatArrayOf(touchX, touchY)
        viewToBitmapMatrix.mapPoints(pts)
        val bitmapX = pts[0].coerceIn(0f, (canonicalSize - 1).toFloat())
        val bitmapY = pts[1].coerceIn(0f, (canonicalSize - 1).toFloat())
        return PointF(bitmapX, bitmapY)
    }

    private fun screenToCanonical(screenX: Float, screenY: Float): PointF {
        return mapTouchToBitmap(screenX, screenY)
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

        // Add to visual glowing feedback trail
        touchTrail.add(TrailPoint(event.x, event.y, System.currentTimeMillis()))

        when (currentMode) {
            Mode.CIRCLE_SELECT -> {
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        circlePoints.clear()
                        circlePath.reset()
                        circlePoints.add(PointF(cx, cy))
                        circlePath.moveTo(cx, cy)
                        lastX = cx
                        lastY = cy
                        isDrawingCircle = true
                        invalidate()
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val dx = abs(cx - lastX)
                        val dy = abs(cy - lastY)
                        if (dx >= 3f || dy >= 3f) {
                            circlePath.quadTo(lastX, lastY, (cx + lastX) / 2f, (cy + lastY) / 2f)
                            circlePoints.add(PointF(cx, cy))
                            lastX = cx
                            lastY = cy
                            invalidate()
                        }
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        if (isDrawingCircle && circlePoints.isNotEmpty()) {
                            val src = sourceBitmap
                            if (src == null) {
                                resetCircleSelection()
                                return true
                            }

                            // Determine if gesture was a single tap or an enclosing circle
                            var minX = cx; var maxX = cx
                            var minY = cy; var maxY = cy
                            for (p in circlePoints) {
                                if (p.x < minX) minX = p.x
                                if (p.x > maxX) maxX = p.x
                                if (p.y < minY) minY = p.y
                                if (p.y > maxY) maxY = p.y
                            }
                            val spanW = maxX - minX
                            val spanH = maxY - minY

                            if (circlePoints.size < 6 || (spanW < 18f && spanH < 18f)) {
                                // Single Tap
                                if (activeSegmentationEngine == SegmentationEngine.EXACT_SELECT) {
                                    saveUndoState()
                                    maskCanvas.drawCircle(cx, cy, 18f, maskFillPaint)
                                    invalidate()
                                    onCircleSelectionStatusChanged?.invoke("🎯 Exact Area: Point painted")
                                } else {
                                    thinkingPulseCenter = PointF(cx, cy)
                                    thinkingPulseStartTime = System.currentTimeMillis()
                                    isThinkingPulse = true
                                    invalidate()

                                    val normX = cx / src.width.toFloat()
                                    val normY = cy / src.height.toFloat()
                                    val mpRes = if (MediaPipeSegmenter.isAvailable()) MediaPipeSegmenter.segmentObject(src, normX, normY) else null
                                    isThinkingPulse = false
                                    if (mpRes != null) {
                                        mergeMaskIntoCanvas(mpRes.first)
                                        onCircleSelectionStatusChanged?.invoke("✨ Magic Touch: Object selected via tap (${mpRes.second}ms)")
                                    } else {
                                        val (mask, latency) = MagicTouchEngine.tapToSelect(src, PointF(cx, cy))
                                        mergeMaskIntoCanvas(mask)
                                        onCircleSelectionStatusChanged?.invoke("✨ Magic Touch: Object selected via tap (${latency}ms)")
                                    }
                                }
                            } else {
                                // Circle Gesture -> Dispatch according to selected segmentation engine
                                circlePath.lineTo(cx, cy)
                                circlePoints.add(PointF(cx, cy))
                                when (activeSegmentationEngine) {
                                    SegmentationEngine.AUTO_GRABCUT -> {
                                        thinkingPulseCenter = PointF((minX + maxX) / 2f, (minY + maxY) / 2f)
                                        thinkingPulseStartTime = System.currentTimeMillis()
                                        isThinkingPulse = true
                                        invalidate()

                                        val (mask, latency) = AutoGrabCutEngine.executeAutoGrabCut(src, circlePath, iterations = 2)
                                        isThinkingPulse = false
                                        mergeMaskIntoCanvas(mask)
                                        onCircleSelectionStatusChanged?.invoke("⚡ Auto-GrabCut (Expanded Object): Segmented in ${latency}ms")
                                    }
                                    SegmentationEngine.AI_MAGIC_TOUCH -> {
                                        thinkingPulseCenter = PointF((minX + maxX) / 2f, (minY + maxY) / 2f)
                                        thinkingPulseStartTime = System.currentTimeMillis()
                                        isThinkingPulse = true
                                        invalidate()

                                        // Build rough circle mask to extract medial axis seeds
                                        val roughMask = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
                                        val cCanvas = Canvas(roughMask)
                                        val cPaint = Paint().apply {
                                            style = Paint.Style.FILL
                                            color = Color.WHITE
                                        }
                                        cCanvas.drawPath(circlePath, cPaint)

                                        val mpRes = if (MediaPipeSegmenter.isAvailable()) {
                                             MediaPipeSegmenter.refineMaskOverhaul(src, roughMask)
                                        } else {
                                            AutoGrabCutEngine.executeAutoGrabCut(src, circlePath, iterations = 2)
                                        }
                                        isThinkingPulse = false
                                        mergeMaskIntoCanvas(mpRes.first)
                                        onCircleSelectionStatusChanged?.invoke("🤖 AI Magic Touch: Full object segmented in ${mpRes.second}ms")
                                    }
                                    SegmentationEngine.EXACT_SELECT -> {
                                        val exactMask = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
                                        val cCanvas = Canvas(exactMask)
                                        val cPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                                            style = Paint.Style.FILL
                                            color = Color.WHITE
                                        }
                                        cCanvas.drawPath(circlePath, cPaint)
                                        mergeMaskIntoCanvas(exactMask)
                                        onCircleSelectionStatusChanged?.invoke("🎯 Exact Area: Masked precisely what you circled")
                                    }
                                }
                            }
                            resetCircleSelection()
                            onMaskChanged?.invoke()
                        }
                        return true
                    }
                }
            }
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
        updateTransformMatrices()

        // 1. Draw Intrinsic 512x512 Bitmap Space via Viewport Affine Matrix
        canvas.save()
        canvas.concat(bitmapToViewMatrix)

        val canonicalRect = Rect(0, 0, canonicalSize, canonicalSize)

        // Source background photo
        sourceBitmap?.let {
            canvas.drawBitmap(it, null, canonicalRect, null)
        } ?: run {
            canvas.drawColor(Color.DKGRAY)
        }

        // Translucent red overlay mask
        canvas.drawBitmap(maskBitmap, null, canonicalRect, overlayPaint)

        // Circle to Select (Magic Touch) active lasso trail (rendered directly in 512x512 intrinsic space)
        if (currentMode == Mode.CIRCLE_SELECT && isDrawingCircle && circlePoints.size >= 2) {
            val closedPreview = Path(circlePath)
            closedPreview.close()
            canvas.drawPath(closedPreview, circleFillPreviewPaint)
            canvas.drawPath(circlePath, circleGlowPaint)
            canvas.drawPath(circlePath, circleStrokePaint)

            val startX = circlePoints[0].x
            val startY = circlePoints[0].y
            canvas.drawCircle(startX, startY, 8f, circleGlowPaint)
            canvas.drawCircle(startX, startY, 4f, circleStrokePaint)
        }

        // Bounding Box selection elements (rendered directly in 512x512 intrinsic space)
        if (currentMode == Mode.BOUNDING_BOX) {
            dragPreviewBox?.let { box ->
                canvas.drawRect(box, bboxFillPaint)
                canvas.drawRect(box, bboxPaint)
            }

            firstCorner?.let { p ->
                val crossSize = 18f
                canvas.drawLine(p.x - crossSize, p.y, p.x + crossSize, p.y, firstCornerPaint)
                canvas.drawLine(p.x, p.y - crossSize, p.x, p.y + crossSize, firstCornerPaint)
                canvas.drawCircle(p.x, p.y, 10f, firstCornerPaint)
                canvas.drawCircle(p.x, p.y, 3.5f, firstCornerFill)
            }
        }

        canvas.restore()

        // 2. Screen-Space Visual Touch Feedback Trail (Glowing white path with smooth fade-out)
        val now = System.currentTimeMillis()
        val trailIter = touchTrail.iterator()
        while (trailIter.hasNext()) {
            val tp = trailIter.next()
            if (now - tp.timestamp > 400) {
                trailIter.remove()
            }
        }
        if (touchTrail.size >= 2) {
            val trailPath = Path()
            trailPath.moveTo(touchTrail[0].x, touchTrail[0].y)
            for (i in 1 until touchTrail.size) {
                trailPath.lineTo(touchTrail[i].x, touchTrail[i].y)
            }
            canvas.drawPath(trailPath, trailGlowPaint)
            canvas.drawPath(trailPath, trailCorePaint)
            postInvalidateOnAnimation()
        }

        // 3. Inline Thinking Pulse Indicator
        if (isThinkingPulse) {
            val center = thinkingPulseCenter
            if (center != null) {
                val screenPts = floatArrayOf(center.x, center.y)
                bitmapToViewMatrix.mapPoints(screenPts)
                val pulseAge = now - thinkingPulseStartTime
                val phase = ((pulseAge % 1000) / 1000f)
                val radius = 20f + phase * 65f
                val alpha = ((1.0f - phase) * 220).toInt().coerceIn(0, 255)

                val pulsePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                    color = Color.argb(alpha, 168, 85, 247)
                    style = Paint.Style.STROKE
                    strokeWidth = 6f
                }
                canvas.drawCircle(screenPts[0], screenPts[1], radius, pulsePaint)
                postInvalidateOnAnimation()
            }
        }

        // 4. Screen-space HUD & Banners (drawn without zoom distortion)
        if (zoomScale > 1.05f) {
            val zoomText = String.format("🔍 %.1fx • 2-finger pan • Double-tap reset", zoomScale)
            val badgeW = 460f
            val badgeH = 46f
            val badgeRect = RectF(width - badgeW - 16f, 16f, width - 16f, 16f + badgeH)
            canvas.drawRoundRect(badgeRect, 12f, 12f, zoomBadgeBg)
            canvas.drawText(zoomText, width - 30f, 48f, zoomBadgeText)
        }

        if (currentMode == Mode.CIRCLE_SELECT) {
            val bannerMsg = when (activeSegmentationEngine) {
                SegmentationEngine.AUTO_GRABCUT -> "⚡ Auto-GrabCut: Draw loop to expand & segment whole object"
                SegmentationEngine.AI_MAGIC_TOUCH -> "🤖 AI Magic Touch: Tap or loop for promptable segmentation"
                SegmentationEngine.EXACT_SELECT -> "🎯 Exact Area: Masks precisely what you circle (no expansion)"
            }
            drawBanner(canvas, bannerMsg)
        } else if (currentMode == Mode.BOUNDING_BOX) {
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
