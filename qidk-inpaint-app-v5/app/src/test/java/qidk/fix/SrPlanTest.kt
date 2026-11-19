package qidk.fix

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SrPlanTest {

    // ------------------------------------------------------------------------------------------------ windows

    @Test fun windowCoresTileTheOutputExactly() {
        for ((w, h) in listOf(1024 to 1024, 1030 to 770, 2048 to 2048, 2 to 2)) {
            val seen = BooleanArray(w * h)
            val wins = SrWindows.plan(w, h)
            for (win in wins) {
                assertEquals(0, win.x0 % 32); assertEquals(0, win.y0 % 32)
                assertTrue(win.coreX + win.coreW <= SrWindows.WIN && win.coreY + win.coreH <= SrWindows.WIN)
                assertEquals(win.dstX, win.x0 + win.coreX); assertEquals(win.dstY, win.y0 + win.coreY)
                for (y in 0 until win.coreH) for (x in 0 until win.coreW) {
                    val i = (win.dstY + y) * w + (win.dstX + x)
                    assertFalse("pixel written twice", seen[i])
                    seen[i] = true
                }
            }
            assertTrue("gap in ${w}x$h", seen.all { it })
        }
        assertEquals(9, SrWindows.plan(1024, 1024).size)
        assertEquals(36, SrWindows.plan(2048, 2048).size)
        assertEquals(121, SrWindows.plan(4096, 4096).size)
    }

    // ------------------------------------------------------------------------------------------------ contract probe

    @Test fun candidateOrderStartsWithTheInpaintValidatedContract() {
        assertEquals(ModelContract(0f, false), ContractCandidates.forKey("migan").first())
        assertEquals(ModelContract(1f, false), ContractCandidates.forKey("lama").first())
        assertEquals(ModelContract(1f, false), ContractCandidates.forKey("aotgan").first())
        assertEquals(4, ContractCandidates.forKey("lama").toSet().size)
    }

    /** An ideal inpainter that really expects [trueC], driven through the wrapper that applies the CANDIDATE contract. */
    private fun net(trueC: ModelContract, cand: ModelContract): TileModel = object : TileModel {
        val truth = ContractProbe.truth()
        override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock): Boolean {
            for (i in image.indices) {
                val px = i / 3
                // wrapper: canonical -> candidate form
                val x = if (cand.minusOneToOne) image[i] * 2f - 1f else image[i]
                val mask = if (hole[px] == 1f) cand.holeValue else 1f - cand.holeValue
                // network: reads that tensor under ITS contract
                val canon = if (trueC.minusOneToOne) (x + 1f) / 2f else x
                val isHole = mask == trueC.holeValue
                val o = if (isHole) truth[i] else canon
                val netOut = if (trueC.minusOneToOne) o * 2f - 1f else o
                // wrapper: candidate form -> canonical
                out[i] = if (cand.minusOneToOne) (netOut + 1f) / 2f else netOut
            }
            return true
        }
    }

    @Test fun probePicksTheContractTheNetworkReallyHas() {
        val all = ContractCandidates.forKey("lama")
        for (trueC in all) {
            val r = ContractProbe.run(all, { c -> net(trueC, c) })
            assertEquals("true=${trueC.describe()} lines=${r.lines}", trueC, r.contract)
        }
        val m = ContractProbe.run(ContractCandidates.forKey("migan"), { c -> net(ModelContract(0f, false), c) })
        assertEquals(ModelContract(0f, false), m.contract)
        assertEquals(1, m.lines.size)                                // the first candidate already passes: no wasted calls
    }

    @Test fun probeRejectsIdentityNonFiniteAndFailingNetworks() {
        val cands = ContractCandidates.forKey("lama")
        val identity = ContractProbe.run(cands, { object : TileModel {
            override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock): Boolean { image.copyInto(out); return true }
        } })
        assertNull("identity must not pass (this is the MIGAN / AOT-GAN failure of the lattice SR)", identity.contract)
        val nan = ContractProbe.run(cands, { object : TileModel {
            override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock): Boolean { out.fill(Float.NaN); return true }
        } })
        assertNull(nan.contract)
        val down = ContractProbe.run(cands, { object : TileModel {
            override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock) = false
        } })
        assertNull(down.contract)
        assertTrue(down.lines.all { it.contains("model call failed") })
        val black = ContractProbe.run(cands, { object : TileModel {
            override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock): Boolean { out.fill(0f); return true }
        } })
        assertNull(black.contract)
    }

    // ------------------------------------------------------------------------------------------------ gates

    private fun agg(
        samples: Double = 1e6, abs: Double = 3.0 / 255.0, nyqShare: Double = 0.33, dcLevels: Double = 0.5,
        pure: Double = 1e5, ident: Double = 0.0, clamped: Double = 0.0, used: Double = 18.0, rejected: Double = 0.0, windows: Int = 9
    ): SrAggregate {
        val a = SrAggregate()
        val per = DoubleArray(SrStat.SIZE)
        per[SrStat.SAMPLES] = samples / windows
        per[SrStat.SUM_ABS_D] = abs * samples / windows
        per[SrStat.SUM_HAD2] = 1000.0 / windows
        per[SrStat.SUM_NYQ2] = 1000.0 * nyqShare / windows
        val dc = dcLevels / 255.0
        per[SrStat.SUM_DC2] = dc * dc * (samples / 4.0) / windows
        per[SrStat.PURE_HOLE] = pure / windows
        per[SrStat.IDENTITY_HOLE] = ident * pure / windows
        per[SrStat.CLAMPED] = clamped * samples / windows
        per[SrStat.PHASES_USED] = used / windows
        per[SrStat.PHASES_REJECTED] = rejected / windows
        repeat(windows) { a.add(per) }
        return a
    }

    @Test fun gatesAcceptARealisticPassAndNameEveryFailure() {
        assertEquals(SrVerdict.ACCEPT, SrGates.evaluate(agg()).verdict)
        assertEquals(SrVerdict.NO_DETAIL, SrGates.evaluate(agg(ident = 1.0)).verdict)        // MIGAN / AOT-GAN "output == input"
        assertEquals(SrVerdict.NO_DETAIL, SrGates.evaluate(agg(abs = 0.0)).verdict)
        assertEquals(SrVerdict.MESH, SrGates.evaluate(agg(nyqShare = 0.9)).verdict)          // LaMa "screen door"
        assertEquals(SrVerdict.UNSTABLE, SrGates.evaluate(agg(used = 10.0, rejected = 8.0)).verdict)
        assertEquals(SrVerdict.LR_BROKEN, SrGates.evaluate(agg(dcLevels = 6.0)).verdict)
        assertEquals(SrVerdict.UNSTABLE, SrGates.evaluate(agg(clamped = 0.7)).verdict)
        assertEquals(SrVerdict.ACCEPT, SrGates.evaluate(agg(ident = 1.0), early = true).verdict)   // early check ignores NO_DETAIL
        assertEquals(SrVerdict.MESH, SrGates.evaluate(agg(nyqShare = 0.9), early = true).verdict)
        assertEquals(SrVerdict.UNSTABLE, SrGates.evaluate(SrAggregate()).verdict)
        assertTrue(SrGates.evaluate(agg(nyqShare = 0.9)).metrics.contains("mesh 0.90"))
    }

    @Test fun cappedAlphaOnlyAppliesToStrongDetail() {
        val s = DoubleArray(SrStat.SIZE)
        s[SrStat.SAMPLES] = 1000.0
        s[SrStat.SUM_ABS_D] = 12.0 / 255.0 * 1000.0                      // 12 levels
        assertEquals(0.3f, SrGates.cappedAlpha(0.6f, s)!!, 1e-4f)
        s[SrStat.SUM_ABS_D] = 3.0 / 255.0 * 1000.0
        assertNull(SrGates.cappedAlpha(0.6f, s))
    }

    // ------------------------------------------------------------------------------------------------ pass runner

    private class Img(override val width: Int, override val height: Int) : SrImage

    private class FakeKernels(val statsFor: (Int, Float) -> DoubleArray) : SrKernels {
        var windowIndex = -1
        val composeAlphas = ArrayList<Float>()
        val composeCoreSizes = ArrayList<Pair<Int, Int>>()
        var nullO1 = 0
        var baseWrites = 0
        var lastSampleOrigin = 0 to 0
        override fun sampleBase(src: SrImage, x0: Int, y0: Int, win: Int, out: FloatArray): Boolean { lastSampleOrigin = x0 to y0; windowIndex++; return true }
        override fun holeMask(x0: Int, y0: Int, win: Int, cell: Int, phase: Int, out: FloatArray) = true
        override fun compose(u: FloatArray, o0: FloatArray?, o1: FloatArray?, win: Int, winX0: Int, winY0: Int, cell: Int, margin: Int,
                             alpha: Float, tau: Float, driftLimit: Float, cx0: Int, cy0: Int, cw: Int, ch: Int,
                             dst: SrImage, dstX: Int, dstY: Int): DoubleArray? {
            composeAlphas += alpha; composeCoreSizes += (cw to ch)
            if (o1 == null) nullO1++
            return statsFor(windowIndex, alpha)
        }
        override fun writeBase(u: FloatArray, win: Int, cx0: Int, cy0: Int, cw: Int, ch: Int, dst: SrImage, dstX: Int, dstY: Int): Boolean { baseWrites++; return true }
    }

    private fun stats(levels: Double = 3.0, nyqShare: Double = 0.33, ident: Double = 0.0): DoubleArray {
        val s = DoubleArray(SrStat.SIZE)
        s[SrStat.SAMPLES] = 4000.0
        s[SrStat.SUM_ABS_D] = levels / 255.0 * 4000.0
        s[SrStat.SUM_HAD2] = 100.0; s[SrStat.SUM_NYQ2] = 100.0 * nyqShare
        s[SrStat.PURE_HOLE] = 1000.0; s[SrStat.IDENTITY_HOLE] = 1000.0 * ident
        s[SrStat.PHASES_USED] = 2.0
        return s
    }

    private class Model(val failFrom: Int = Int.MAX_VALUE, val failPhase1Only: Boolean = false) : TileModel {
        var calls = 0
        override fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock): Boolean {
            val idx = calls++
            val window = idx / 2
            if (window >= failFrom) return false
            if (failPhase1Only && idx % 2 == 1) return false
            return true
        }
    }

    @Test fun passVisitsEveryWindowTwiceAndAggregates() {
        val k = FakeKernels { _, _ -> stats() }
        val m = Model()
        val out = NsdtPass(k, m, PhaseClock()).run(Img(512, 512), Img(1024, 1024), 1, 1, null)
        assertTrue(out.gate.reason, out.gate.accepted)
        assertEquals(9, out.windows); assertEquals(18, m.calls); assertEquals(9, k.composeAlphas.size)
        assertEquals(1024 * 1024, k.composeCoreSizes.sumOf { it.first * it.second })
        assertEquals(0, k.nullO1)
    }

    @Test fun strongWindowsAreRecomposedWithASmallerAlpha() {
        val k = FakeKernels { _, alpha -> stats(levels = 12.0 * alpha / 0.6) }
        NsdtPass(k, Model(), PhaseClock(), NsdtParams(alpha = 0.6f)).run(Img(512, 512), Img(1024, 1024), 1, 1, null)
        assertEquals(18, k.composeAlphas.size)                        // every window composed twice
        assertEquals(0.6f, k.composeAlphas[0], 1e-6f)
        assertEquals(0.3f, k.composeAlphas[1], 1e-4f)
    }

    @Test fun aLatticeIsCaughtAfterSixWindowsNotAfterAllOfThem() {
        val k = FakeKernels { _, _ -> stats(nyqShare = 0.95) }
        val m = Model()
        val out = NsdtPass(k, m, PhaseClock()).run(Img(1024, 1024), Img(2048, 2048), 1, 2, null)
        assertTrue(out.earlyAbort)
        assertEquals(SrVerdict.MESH, out.gate.verdict)
        assertEquals(6, out.windows)
        assertEquals(12, m.calls)                                     // 6 of 36 windows: the rest of the NPU time is saved
    }

    @Test fun cancellationStopsBetweenWindows() {
        val k = FakeKernels { _, _ -> stats() }
        var seen = 0
        val progress = object : SrProgress {
            override val cancelled: Boolean get() = seen >= 3
            override fun report(pass: Int, passes: Int, done: Int, total: Int, note: String) { seen = done }
        }
        val out = NsdtPass(k, Model(), PhaseClock()).run(Img(512, 512), Img(1024, 1024), 1, 1, progress)
        assertTrue(out.cancelled)
        assertTrue("stops at the first window boundary after the request, not at the end of the pass: ${out.windows}", out.windows in 3..4)
    }

    @Test fun threeFailedWindowsInARowAbortThePass() {
        val k = FakeKernels { _, _ -> stats() }
        val out = NsdtPass(k, Model(failFrom = 2), PhaseClock()).run(Img(512, 512), Img(1024, 1024), 1, 1, null)
        assertEquals(SrVerdict.UNSTABLE, out.gate.verdict)
        assertTrue(out.earlyAbort)
        assertEquals(5, out.windows)                                  // 2 good + 3 failed
        assertEquals(3, out.modelFailures)
    }

    @Test fun aSingleFailedPhaseStillComposesWithTheOtherOne() {
        val k = FakeKernels { _, _ -> stats() }
        val out = NsdtPass(k, Model(failPhase1Only = true), PhaseClock()).run(Img(512, 512), Img(1024, 1024), 1, 1, null)
        assertEquals(9, out.windows)
        assertEquals(9, k.nullO1)
    }

    @Test fun classicalFallbackWritesEveryWindowWithoutTheModel() {
        val k = FakeKernels { _, _ -> stats() }
        val m = Model()
        NsdtPass(k, m, PhaseClock()).runClassical(Img(512, 512), Img(1024, 1024), null, 1, 1)
        assertEquals(9, k.baseWrites)
        assertEquals(0, m.calls)
        assertNotNull(k.lastSampleOrigin)
    }
}
