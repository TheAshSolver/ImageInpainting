package qidk.fix

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random
import kotlin.math.abs

class SharedViewportTest {

    private fun geo(sv: SharedViewport, vw: Int, vh: Int, bw: Int, bh: Int) =
        sv.register().apply { viewW = vw.toFloat(); viewH = vh.toFloat(); bmpW = bw.toFloat(); bmpH = bh.toFloat() }

    @Test fun focalPointStaysUnderTheFingers() {
        val sv = SharedViewport()
        val g = geo(sv, 1080, 900, 512, 512)
        val rnd = Random(3)
        sv.set(6f, 0.5f, 0.5f)
        repeat(300) {
            val fx = rnd.nextFloat() * 1080f
            val fy = rnd.nextFloat() * 900f
            val nx = sv.toNormX(g, fx)
            val ny = sv.toNormY(g, fy)
            sv.zoomAbout(g, 0.9f + rnd.nextFloat() * 0.2f, fx, fy)   // small steps stay inside the free range
            assertEquals(nx, sv.toNormX(g, fx), 1e-3f)
            assertEquals(ny, sv.toNormY(g, fy), 1e-3f)
            sv.set(6f, 0.5f, 0.5f)
        }
    }

    @Test fun exactFocalInvarianceInTheFreeRange() {
        val sv = SharedViewport()
        val g = geo(sv, 1000, 1000, 512, 512)
        sv.set(4f, 0.5f, 0.5f)
        val fx = 640f; val fy = 380f
        val nx = sv.toNormX(g, fx); val ny = sv.toNormY(g, fy)
        sv.zoomAbout(g, 1.5f, fx, fy)
        assertEquals(6f, sv.zoom, 1e-4f)
        assertEquals(nx, sv.toNormX(g, fx), 1e-5f)
        assertEquals(ny, sv.toNormY(g, fy), 1e-5f)
    }

    @Test fun bothViewsShowTheSameSceneRegion() {
        val sv = SharedViewport()
        val top = geo(sv, 1080, 700, 512, 512)       // input 512 x 512
        val bottom = geo(sv, 1080, 700, 4096, 4096)  // output 4096 x 4096 in an identical viewport
        sv.set(3.7f, 0.31f, 0.62f)
        val rnd = Random(9)
        repeat(100) {
            val u = rnd.nextFloat(); val v = rnd.nextFloat()
            assertEquals(sv.toViewX(top, u), sv.toViewX(bottom, u), 1e-3f)
            assertEquals(sv.toViewY(top, v), sv.toViewY(bottom, v), 1e-3f)
        }
        // a gesture made on the bottom view moves the top view by the same amount
        val before = sv.toViewX(top, 0.5f)
        sv.panBy(bottom, 25f, 0f)
        assertEquals(before + 25f, sv.toViewX(top, 0.5f), 1e-2f)
    }

    @Test fun dragMovesTheImageByExactlyTheDraggedPixels() {
        val sv = SharedViewport()
        val g = geo(sv, 1000, 1000, 800, 600)
        sv.set(5f, 0.5f, 0.5f)
        val x0 = sv.toViewX(g, 0.4f); val y0 = sv.toViewY(g, 0.7f)
        sv.panBy(g, -37f, 22f)
        assertEquals(x0 - 37f, sv.toViewX(g, 0.4f), 1e-3f)
        assertEquals(y0 + 22f, sv.toViewY(g, 0.7f), 1e-3f)
    }

    @Test fun neverShowsAnEmptyMargin() {
        val sv = SharedViewport()
        val g = geo(sv, 1080, 900, 1024, 1024)
        sv.set(2f, 0.5f, 0.5f)
        repeat(50) { sv.panBy(g, 400f, 400f) }
        var r = sv.destRect(g)
        assertEquals(0f, r[0], 1e-2f)            // left edge of the image sits at the left edge of the view
        assertEquals((900f - 900f), r[1], 1e-2f) // fit is height-limited: top edge flush at zoom 2 only after clamp
        repeat(100) { sv.panBy(g, -400f, -400f) }
        r = sv.destRect(g)
        assertEquals(g.viewW, r[2], 1e-2f)
        assertEquals(g.viewH, r[3], 1e-2f)
    }

    @Test fun fittedImageCannotBePannedAndStaysCentred() {
        val sv = SharedViewport()
        val g = geo(sv, 1080, 1920, 512, 512)
        sv.panBy(g, 300f, -200f)
        assertEquals(0.5f, sv.centerX, 1e-6f)
        assertEquals(0.5f, sv.centerY, 1e-6f)
        assertFalse(sv.isTransformed)
        // letter-boxed axis: at zoom 1 the image is centred vertically
        val r = sv.destRect(g)
        assertEquals((1920f - 1080f) / 2f, r[1], 1e-3f)
    }

    @Test fun zoomIsLimitedAndResetRecentresBothViews() {
        val sv = SharedViewport(maxZoom = 16f)
        val a = geo(sv, 1000, 600, 512, 512)
        val b = geo(sv, 1000, 600, 1024, 1024)
        sv.zoomAbout(a, 100f, 500f, 300f)
        assertEquals(16f, sv.zoom, 1e-6f)
        sv.zoomAbout(a, 0.0001f, 500f, 300f)
        assertEquals(1f, sv.zoom, 1e-6f)
        sv.set(8f, 0.2f, 0.9f)
        assertTrue(sv.isTransformed)
        sv.reset()
        assertEquals(1f, sv.zoom, 0f)
        assertEquals(0.5f, sv.centerX, 0f)
        assertEquals(sv.toViewX(a, 0.25f), sv.toViewX(b, 0.25f), 1e-4f)
    }

    @Test fun listenersAreToldOnceAndCanBeRemoved() {
        val sv = SharedViewport()
        val g = geo(sv, 500, 500, 100, 100)
        var calls = 0
        val l = SharedViewport.Listener { _, _, _ -> calls++ }
        sv.addListener(l)
        sv.zoomAbout(g, 2f, 250f, 250f)
        sv.panBy(g, 5f, 5f)
        assertEquals(2, calls)
        sv.removeListener(l)
        sv.reset()
        assertEquals(2, calls)
    }

    @Test fun differentViewSizesUseTheSmallestVisibleFraction() {
        val sv = SharedViewport()
        val wide = geo(sv, 1000, 300, 512, 512)    // height limited: image 300 px wide in a 1000 px view
        val tall = geo(sv, 400, 400, 512, 512)     // full fit
        sv.set(2f, 0.5f, 0.5f)
        sv.panBy(tall, -10000f, 0f)
        // the narrow-fraction view (tall) decides the clamp; the wide view just shows a margin
        val rTall = sv.destRect(tall)
        assertEquals(tall.viewW, rTall[2], 1e-2f)
        assertTrue(sv.centerX in 0f..1f)
        assertTrue(wide.valid)
    }
}
