package qidk.fix

import java.util.concurrent.CopyOnWriteArrayList
import kotlin.math.min

/**
 * ONE zoom / pan state for every image view of the dual viewport (input on top, output below).
 *
 * Why one object and not "view A tells view B": two views that each own a transform and mirror each other's callbacks
 * drift apart (rounding, different letter-boxing, a gesture that reaches only one of them). Here the views own no
 * transform at all; they only draw what this object says. Pure Kotlin on purpose (no android.* import): the maths is
 * unit-tested on the JVM.
 *
 * State (resolution independent, so a 512 px input and a 4096 px output show the same scene region):
 *   zoom z >= 1 (1 = whole image fitted into the view), centre (cx, cy) in NORMALISED image coordinates [0,1].
 *
 * For a view k with letter-boxed fit scale s_k = min(Vw/bw, Vh/bh) the full image is Lx = s_k*bw by Ly = s_k*bh pixels at
 * z = 1, and a normalised image point (u, v) lands on screen at
 *     x = Vw/2 + (u - cx) * Lx * z          y = Vh/2 + (v - cy) * Ly * z
 * Focal zoom keeps the image point under the fingers fixed:   c' = p - (f - V/2) / (L * z')
 * Dragging by d pixels moves the centre by                    c' = c - d / (L * z)
 * and the centre is clamped so that no empty margin appears (visible fraction = V / (L * z), centre in
 * [frac/2, 1 - frac/2], or exactly 0.5 when the whole axis is visible). With several views the SMALLEST visible
 * fraction is used, so every view can still reach the whole image.
 */
class SharedViewport(val maxZoom: Float = 32f) {

    /** Pixel geometry of one participating view. The view keeps it up to date from onSizeChanged / setBitmap. */
    class Geometry {
        var viewW = 0f
        var viewH = 0f
        var bmpW = 1f
        var bmpH = 1f

        /** View pixels per bitmap pixel at zoom 1 (letter-boxed fit). 0 while the view has no size. */
        val fit: Float
            get() = if (viewW <= 0f || viewH <= 0f || bmpW <= 0f || bmpH <= 0f) 0f else min(viewW / bmpW, viewH / bmpH)
        /** Width / height of the whole image in view pixels at zoom 1. */
        val lx: Float get() = fit * bmpW
        val ly: Float get() = fit * bmpH
        val valid: Boolean get() = fit > 0f
    }

    fun interface Listener {
        fun onViewportChanged(zoom: Float, centerX: Float, centerY: Float)
    }

    var zoom = 1f
        private set
    var centerX = 0.5f
        private set
    var centerY = 0.5f
        private set

    private val geometries = CopyOnWriteArrayList<Geometry>()
    private val listeners = CopyOnWriteArrayList<Listener>()

    fun register(): Geometry = Geometry().also { geometries.add(it) }
    fun unregister(g: Geometry) { geometries.remove(g) }
    fun addListener(l: Listener) { listeners.addIfAbsent(l) }
    fun removeListener(l: Listener) { listeners.remove(l) }

    /** "1:1 Reset": whole image fitted and centred in every view. */
    fun reset() = set(1f, 0.5f, 0.5f)

    val isTransformed: Boolean get() = zoom > 1.001f || kotlin.math.abs(centerX - 0.5f) > 0.001f || kotlin.math.abs(centerY - 0.5f) > 0.001f

    fun set(newZoom: Float, cx: Float, cy: Float) {
        zoom = newZoom.coerceIn(1f, maxZoom)
        centerX = cx
        centerY = cy
        clampCenter()
        notifyChanged()
    }

    /** Pinch: multiply the zoom by [factor] about the focal point ([fx], [fy]) given in pixels of view [g]. */
    fun zoomAbout(g: Geometry, factor: Float, fx: Float, fy: Float) {
        if (!g.valid || factor <= 0f || factor.isNaN()) return
        val z2 = (zoom * factor).coerceIn(1f, maxZoom)
        // normalised image point under the fingers
        val nx = centerX + (fx - g.viewW / 2f) / (g.lx * zoom)
        val ny = centerY + (fy - g.viewH / 2f) / (g.ly * zoom)
        zoom = z2
        centerX = nx - (fx - g.viewW / 2f) / (g.lx * z2)
        centerY = ny - (fy - g.viewH / 2f) / (g.ly * z2)
        clampCenter()
        notifyChanged()
    }

    /** Drag: the image follows the finger, [dx], [dy] are view-pixel deltas in view [g]. */
    fun panBy(g: Geometry, dx: Float, dy: Float) {
        if (!g.valid) return
        centerX -= dx / (g.lx * zoom)
        centerY -= dy / (g.ly * zoom)
        clampCenter()
        notifyChanged()
    }

    /** Double tap: fit <-> [zoomIn] about the tapped point. */
    fun toggleZoomAt(g: Geometry, fx: Float, fy: Float, zoomIn: Float = 4f) {
        if (zoom > 1.01f) reset() else zoomAbout(g, zoomIn / zoom, fx, fy)
    }

    // ------------------------------------------------------------------------------------------ mapping helpers

    fun toViewX(g: Geometry, nx: Float): Float = g.viewW / 2f + (nx - centerX) * g.lx * zoom
    fun toViewY(g: Geometry, ny: Float): Float = g.viewH / 2f + (ny - centerY) * g.ly * zoom
    fun toNormX(g: Geometry, x: Float): Float = centerX + (x - g.viewW / 2f) / (g.lx * zoom)
    fun toNormY(g: Geometry, y: Float): Float = centerY + (y - g.viewH / 2f) / (g.ly * zoom)

    /** Where the whole image is drawn in view [g]: [left, top, right, bottom] in pixels (may extend beyond the view). */
    fun destRect(g: Geometry, out: FloatArray = FloatArray(4)): FloatArray {
        out[0] = toViewX(g, 0f)
        out[1] = toViewY(g, 0f)
        out[2] = toViewX(g, 1f)
        out[3] = toViewY(g, 1f)
        return out
    }

    /** View pixels per bitmap pixel of view [g] at the current zoom (> 1 means magnified). */
    fun pixelScale(g: Geometry): Float = g.fit * zoom

    // ------------------------------------------------------------------------------------------ internals

    private fun clampCenter() {
        var fracX = Float.MAX_VALUE
        var fracY = Float.MAX_VALUE
        var any = false
        for (g in geometries) {
            if (!g.valid) continue
            any = true
            fracX = min(fracX, g.viewW / (g.lx * zoom))
            fracY = min(fracY, g.viewH / (g.ly * zoom))
        }
        if (!any) {
            centerX = centerX.coerceIn(0f, 1f)
            centerY = centerY.coerceIn(0f, 1f)
            return
        }
        centerX = clampAxis(centerX, fracX)
        centerY = clampAxis(centerY, fracY)
    }

    private fun clampAxis(c: Float, frac: Float): Float =
        if (frac >= 1f) 0.5f else c.coerceIn(frac / 2f, 1f - frac / 2f)

    private fun notifyChanged() {
        val z = zoom; val x = centerX; val y = centerY
        for (l in listeners) l.onViewportChanged(z, x, y)
    }
}
