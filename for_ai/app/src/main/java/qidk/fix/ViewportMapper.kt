package qidk.fix

import kotlin.math.max
import kotlin.math.min

/**
 * Single source of truth for view <-> bitmap coordinates. The canvas must DRAW the bitmap/mask/strokes through
 * [bitmapToViewX]/[bitmapToViewY] (or [destLeft]..[destBottom]) and READ touches through [viewToBitmapX]/[viewToBitmapY];
 * a screen that draws with one mapping and reads with another is what selects "the boat on the other side".
 *
 * Base fit: uniform letter-boxed "fit centre" (default) or, with [stretch] = true, the old non-uniform fill.
 * Pinch-zoom is a user transform applied about the view centre after the base fit.
 * Pure Kotlin on purpose: no android.* import, so it is unit-testable on the JVM.
 */
class ViewportMapper(var stretch: Boolean = false) {
    var viewW = 0f; private set
    var viewH = 0f; private set
    var bmpW = 512f; private set
    var bmpH = 512f; private set
    var userScale = 1f; private set
    private var userPanX = 0f
    private var userPanY = 0f

    /** bitmap -> view:  v = b * s + t, per axis. */
    var sx = 1f; private set
    var sy = 1f; private set
    var tx = 0f; private set
    var ty = 0f; private set

    fun setView(w: Int, h: Int) { viewW = w.toFloat(); viewH = h.toFloat(); recompute() }
    fun setBitmap(w: Int, h: Int) { bmpW = max(1, w).toFloat(); bmpH = max(1, h).toFloat(); recompute() }
    fun setUserTransform(scale: Float, panX: Float, panY: Float) {
        userScale = scale.coerceIn(1f, 16f); userPanX = panX; userPanY = panY; recompute()
    }
    val panX get() = userPanX
    val panY get() = userPanY

    private fun recompute() {
        if (viewW <= 0f || viewH <= 0f) return
        val bsx: Float; val bsy: Float; val btx: Float; val bty: Float
        if (stretch) {
            bsx = viewW / bmpW; bsy = viewH / bmpH; btx = 0f; bty = 0f
        } else {
            val s = min(viewW / bmpW, viewH / bmpH)
            bsx = s; bsy = s; btx = (viewW - bmpW * s) / 2f; bty = (viewH - bmpH * s) / 2f
        }
        val cx = viewW / 2f
        val cy = viewH / 2f
        sx = bsx * userScale; sy = bsy * userScale
        tx = cx + (btx - cx) * userScale + userPanX
        ty = cy + (bty - cy) * userScale + userPanY
    }

    val destLeft get() = tx
    val destTop get() = ty
    val destRight get() = tx + bmpW * sx
    val destBottom get() = ty + bmpH * sy

    /** Touch -> bitmap pixel, clamped to [0, size-1] (a touch in the letter-box margin selects the nearest edge pixel). */
    fun viewToBitmapX(x: Float): Float = ((x - tx) / sx).coerceIn(0f, bmpW - 1f)
    fun viewToBitmapY(y: Float): Float = ((y - ty) / sy).coerceIn(0f, bmpH - 1f)

    /** Unclamped inverse, for tests and diagnostics. */
    fun viewToBitmapRawX(x: Float): Float = (x - tx) / sx
    fun viewToBitmapRawY(y: Float): Float = (y - ty) / sy

    fun bitmapToViewX(bx: Float): Float = bx * sx + tx
    fun bitmapToViewY(by: Float): Float = by * sy + ty

    fun isInsideImage(x: Float, y: Float): Boolean = x >= destLeft && x <= destRight && y >= destTop && y <= destBottom

    override fun toString() = "view=${viewW.toInt()}x${viewH.toInt()} bmp=${bmpW.toInt()}x${bmpH.toInt()} " +
        "s=($sx,$sy) t=($tx,$ty) zoom=$userScale stretch=$stretch"
}
