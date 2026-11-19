package qidk.fix

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** Decides whether a finger stroke is a lasso (closed loop around an object) or a scribble (drawn over an object). */
object StrokeGeometry {
    class Info(
        val isLoop: Boolean,
        val minX: Float, val minY: Float, val maxX: Float, val maxY: Float,
        val pathLen: Float, val gap: Float, val areaRatio: Float, val compactness: Float
    ) {
        val width get() = max(1f, maxX - minX)
        val height get() = max(1f, maxY - minY)
        val diag get() = hypot(width, height)
    }

    /**
     * Loop  <=> the end of the stroke comes back near its start (gap <= 0.6 * box diagonal, so a 3/4 circle still
     * counts) AND the implicitly closed polygon is compact: 4*pi*area / perimeter^2 >= 0.15 (circle 1.0, thin diagonal
     * ellipse ~0.5, hand-drawn blob 0.5-0.9; a straight line or a back-and-forth scribble is below 0.1).
     */
    fun analyse(xs: FloatArray, ys: FloatArray, n: Int = min(xs.size, ys.size)): Info {
        require(n >= 2) { "a stroke needs at least 2 points" }
        var minX = xs[0]; var maxX = xs[0]; var minY = ys[0]; var maxY = ys[0]
        var len = 0f
        var twiceArea = 0f
        for (i in 0 until n) {
            minX = min(minX, xs[i]); maxX = max(maxX, xs[i]); minY = min(minY, ys[i]); maxY = max(maxY, ys[i])
            val j = (i + 1) % n
            twiceArea += xs[i] * ys[j] - xs[j] * ys[i]
            if (i + 1 < n) len += hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        }
        val gap = hypot(xs[n - 1] - xs[0], ys[n - 1] - ys[0])
        val w = max(1f, maxX - minX)
        val h = max(1f, maxY - minY)
        val areaRatio = abs(twiceArea) / 2f / (w * h)
        val diag = hypot(w, h)
        val area = abs(twiceArea) / 2f
        val perimeter = len + gap
        val compactness = if (perimeter > 0f) (4f * Math.PI.toFloat() * area) / (perimeter * perimeter) else 0f
        val loop = gap <= 0.6f * diag && compactness >= 0.15f
        return Info(loop, minX, minY, maxX, maxY, len, gap, areaRatio, compactness)
    }
}
