package qidk.fix

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class PureLogicTest {

    // ---------------------------------------------------------------- ViewportMapper
    @Test fun letterboxedMappingInvertsTheDisplayMatrix() {
        val m = ViewportMapper().apply { setView(1080, 2000); setBitmap(512, 512) }
        // 1080 / 512 = 2.109375, bitmap is centred vertically: top margin (2000 - 1080) / 2 = 460
        assertEquals(2.109375f, m.sx, 1e-6f)
        assertEquals(460f, m.ty, 1e-3f)
        assertEquals(256f, m.viewToBitmapRawX(540f), 1e-3f)
        assertEquals(256f, m.viewToBitmapRawY(1000f), 1e-3f)
        // a point the old "x * 512 / viewW, y * 512 / viewH" code got wrong: view y = 600
        assertEquals((600f - 460f) / 2.109375f, m.viewToBitmapRawY(600f), 1e-3f)
        assertTrue(Math.abs(m.viewToBitmapRawY(600f) - 600f * 512f / 2000f) > 50f)
    }

    @Test fun touchesInTheMarginClampToTheEdgePixel() {
        val m = ViewportMapper().apply { setView(1080, 2000); setBitmap(512, 512) }
        assertEquals(0f, m.viewToBitmapY(10f), 0f)
        assertEquals(511f, m.viewToBitmapY(1990f), 0f)
        assertEquals(511f, m.viewToBitmapX(5000f), 0f)
        assertFalse(m.isInsideImage(500f, 10f))
        assertTrue(m.isInsideImage(500f, 1000f))
    }

    @Test fun roundTripWithZoomAndPan() {
        val m = ViewportMapper().apply { setView(1200, 800); setBitmap(1024, 1024); setUserTransform(3.2f, -140f, 55f) }
        for ((x, y) in listOf(0f to 0f, 600f to 400f, 1199f to 799f, 321.5f to 77.25f)) {
            val bx = m.viewToBitmapRawX(x); val by = m.viewToBitmapRawY(y)
            assertEquals(x, m.bitmapToViewX(bx), 1e-2f)
            assertEquals(y, m.bitmapToViewY(by), 1e-2f)
        }
        // zoom is about the view centre: with no pan the centre keeps its bitmap pixel
        m.setUserTransform(4f, 0f, 0f)
        assertEquals(512f, m.viewToBitmapRawX(600f), 1e-3f)
    }

    @Test fun stretchModeMatchesTheLegacyFormula() {
        val m = ViewportMapper(stretch = true).apply { setView(1080, 2000); setBitmap(512, 512) }
        assertEquals(600f * 512f / 2000f, m.viewToBitmapRawY(600f), 1e-3f)
    }

    // ---------------------------------------------------------------- StrokeGeometry
    private fun arc(cx: Float, cy: Float, rx: Float, ry: Float, sweep: Double, n: Int = 60): Pair<FloatArray, FloatArray> {
        val xs = FloatArray(n) { (cx + rx * cos(sweep * it / (n - 1))).toFloat() }
        val ys = FloatArray(n) { (cy + ry * sin(sweep * it / (n - 1))).toFloat() }
        return xs to ys
    }

    @Test fun circleAndOpenCAreLoops_lineAndZigzagAreNot() {
        arc(200f, 200f, 90f, 50f, 2.1 * PI).let { assertTrue(StrokeGeometry.analyse(it.first, it.second).isLoop) }
        arc(200f, 200f, 90f, 90f, 1.5 * PI).let { assertTrue("75% circle is still a lasso", StrokeGeometry.analyse(it.first, it.second).isLoop) }
        val lx = FloatArray(30) { 10f + it * 10f }; val ly = FloatArray(30) { 50f + it * 0.5f }
        assertFalse(StrokeGeometry.analyse(lx, ly).isLoop)
        val zx = FloatArray(40) { if ((it / 10) % 2 == 0) 10f + (it % 10) * 20f else 190f - (it % 10) * 20f }
        val zy = FloatArray(40) { 10f + (it / 10) * 6f }
        assertFalse("flat zig-zag scribble", StrokeGeometry.analyse(zx, zy).isLoop)
        // thin ellipse rotated 45 degrees (a lasso around a diagonal aircraft): bounding box is much bigger than the loop
        val ex = FloatArray(60) { (300 + 100 * cos(2.1 * PI * it / 59) * 0.7071 - 30 * sin(2.1 * PI * it / 59) * 0.7071).toFloat() }
        val ey = FloatArray(60) { (300 + 100 * cos(2.1 * PI * it / 59) * 0.7071 + 30 * sin(2.1 * PI * it / 59) * 0.7071).toFloat() }
        assertTrue("diagonal thin lasso", StrokeGeometry.analyse(ex, ey).isLoop)
    }

    // ---------------------------------------------------------------- BackendMatrix
    @Test fun backendFeasibility() {
        assertEquals(listOf(Backend.NPU), BackendMatrix.allowed(ModelFamily.SD15))
        assertEquals(listOf(Backend.NPU, Backend.GPU, Backend.CPU), BackendMatrix.allowed(ModelFamily.MIGAN))
        assertTrue(Backend.GPU in BackendMatrix.allowed(ModelFamily.LAMA))
        assertTrue(Backend.GPU in BackendMatrix.allowed(ModelFamily.AOT_GAN))
        assertEquals(listOf(Backend.CPU), BackendMatrix.allowed(ModelFamily.CLASSICAL))
        assertEquals(Backend.NPU, BackendMatrix.coerce(ModelFamily.SD15, Backend.GPU))
        assertEquals(Backend.GPU, BackendMatrix.coerce(ModelFamily.MIGAN, Backend.GPU))
        try { BackendMatrix.require(ModelFamily.SD15, Backend.CPU); fail("SD on CPU must be refused") } catch (_: IllegalArgumentException) {}
    }

    @Test fun spinnerTextToFamily() {
        assertEquals(ModelFamily.SD15, BackendMatrix.familyOf("Stable Diffusion 1.5 (Inpainting - DPM 12-step)"))
        assertEquals(ModelFamily.SD15, BackendMatrix.familyOf("Smart SD"))
        assertEquals(ModelFamily.MIGAN, BackendMatrix.familyOf("MIGAN (Portraits & Speed)"))
        assertEquals(ModelFamily.LAMA, BackendMatrix.familyOf("LaMa Dilated (Large Voids)"))
        assertEquals(ModelFamily.AOT_GAN, BackendMatrix.familyOf("AOT-GAN (Dense Texture)"))
        for (s in listOf("Biharmonic Poisson", "Criminisi Exemplar", "Fast Marching (Telea)", "Fluid Streamlines (Navier-Stokes)"))
            assertEquals(s, ModelFamily.CLASSICAL, BackendMatrix.familyOf(s))
        assertNull(BackendMatrix.familyOf("Auto (Router Recommended)"))
    }

    // ---------------------------------------------------------------- MiniJson
    @Test fun jsonBasics() {
        val v = MiniJson.parse("""{"a":[1,2.5e1,-3],"s":"x\"yA","n":null,"t":true,"nan":NaN,"inf":-Infinity,"o":{}}""") as Map<*, *>
        assertEquals(listOf(1.0, 25.0, -3.0), v["a"])
        assertEquals("x\"yA", v["s"])
        assertNull(v["n"]); assertEquals(true, v["t"])
        assertTrue((v["nan"] as Double).isNaN()); assertEquals(Double.NEGATIVE_INFINITY, v["inf"])
        try { MiniJson.parse("{} x"); fail() } catch (_: IllegalArgumentException) {}
    }

    // ---------------------------------------------------------------- LinUcbModel
    @Test fun linucbFromAandB_listLayout() {
        val m = LinUcbModel.parse("""{"alpha":0.5,"feature_names":["f0","f1"],
            "scaler":{"mean":[1,1],"std":[2,2]},
            "arms":[{"name":"max_quality","A":[[2,0],[0,2]],"b":[1,0]},
                    {"name":"balanced","A":[[2,0],[0,2]],"b":[0,1]},
                    {"name":"thermal_safe","A":[[2,0],[0,2]],"b":[-1,-1]}]}""")
        assertEquals(listOf("max_quality", "balanced", "thermal_safe"), m.armNames)
        assertEquals(2, m.dim); assertEquals(listOf("f0", "f1"), m.featureNames)
        val z = m.standardise(doubleArrayOf(3.0, 5.0))
        assertEquals(1.0, z[0], 1e-12); assertEquals(2.0, z[1], 1e-12)
        // theta0 = A^-1 b = (0.5, 0); x = (1,1): 0.5 + alpha * sqrt(0.5 + 0.5) = 1.0
        assertEquals(1.0, m.scores(doubleArrayOf(1.0, 1.0))[0], 1e-12)
        assertEquals(0.5, m.scores(doubleArrayOf(1.0, 1.0), alphaOverride = 0.0)[0], 1e-12)
        assertEquals(0, m.select(doubleArrayOf(1.0, -1.0), 0.0))
        assertEquals(1, m.select(doubleArrayOf(-1.0, 1.0), 0.0))
        assertEquals(2, m.select(doubleArrayOf(-1.0, -1.0), 0.0))
    }

    @Test fun linucbFromObjectLayoutWithThetaAndAinv() {
        val m = LinUcbModel.parse("""{"policy":{"alpha_serve":0.0,"arms":{
            "q":{"theta":[1,0],"A_inv":[[1,0],[0,1]]}, "b":{"theta":[0,1],"A_inv":[[1,0],[0,1]]}}}}""")
        assertEquals(listOf("q", "b"), m.armNames)
        assertEquals(0, m.select(doubleArrayOf(2.0, 1.0)))
        assertEquals(1, m.select(doubleArrayOf(1.0, 2.0)))
    }

    @Test fun linucbFromStackedArrays() {
        val m = LinUcbModel.parse("""{"arms":["a","b"],"alpha":1.0,
            "theta":[[1,0],[0,1]],"A_inv":[[[1,0],[0,1]],[[4,0],[0,4]]]}""")
        assertEquals(listOf("a", "b"), m.armNames)
        // x = (1,1): arm a -> 1 + sqrt(2) ; arm b -> 1 + sqrt(8)
        val s = m.scores(doubleArrayOf(1.0, 1.0))
        assertEquals(1 + Math.sqrt(2.0), s[0], 1e-12); assertEquals(1 + Math.sqrt(8.0), s[1], 1e-12)
        assertEquals(1, m.select(doubleArrayOf(1.0, 1.0)))
    }

    @Test fun linucbRejectsUnknownSchema() {
        try { LinUcbModel.parse("""{"weights_blob":"abc"}"""); fail() } catch (e: IllegalArgumentException) {
            assertTrue(e.message!!.contains("weights_blob"))
        }
    }

    @Test fun linucbIsFastEnoughOnDevice() {
        val d = 20; val k = 3
        val rnd = java.util.Random(7)
        fun mat() = (0 until d).joinToString(",", "[", "]") { i -> (0 until d).joinToString(",", "[", "]") { j -> if (i == j) "1.0" else "0.01" } }
        val arms = (0 until k).joinToString(",") { a -> """{"name":"m$a","A_inv":${mat()},"theta":[${(0 until d).joinToString(",") { rnd.nextGaussian().toString() }}]}""" }
        val m = LinUcbModel.parse("""{"alpha":0.3,"arms":[$arms]}""")
        val x = DoubleArray(d) { rnd.nextGaussian() }
        val out = DoubleArray(k)
        repeat(20000) { m.scores(x, null, out) }            // JIT warm-up
        val t0 = System.nanoTime(); val n = 20000
        repeat(n) { m.scores(x, null, out) }
        val us = (System.nanoTime() - t0) / 1000.0 / n
        println("LinUCB d=$d K=$k: %.2f us per decision".format(us))
        assertTrue("decision took $us us", us < 100.0)
    }
}
