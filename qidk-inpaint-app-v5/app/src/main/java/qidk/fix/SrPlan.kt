package qidk.fix

import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Pure (JVM-testable) half of the neural super-resolution path: window geometry, the model I/O contract probe, the quality
 * gates and the per-pass runner. The pixel kernels live in native code (SrNative / sr_core.cpp) behind [SrKernels].
 *
 * Method = Null-Space Detail Transfer (NSDT), see SUPER_STUDIO_REPAIR_GUIDE.md section 1:
 *   X = U + alpha * softclamp( sum_p w_p * P(O_p - U) ),   U = classical 2x base, P = I - UP(DOWN(.)).
 * The base U carries every low-frequency fact of the picture, so the network cannot move geometry; it only contributes
 * the detail the low-resolution grid cannot represent, and every pass is checked by gates before it is accepted.
 */
interface SrImage {
    val width: Int
    val height: Int
}

object SrWindows {
    const val WIN = 512      // network input
    const val CORE = 384     // pixels of each window that are kept
    const val PAD = 64       // context around the core
    const val CELL = 32      // checkerboard cell (pixels of the 2x grid)
    const val MARGIN = 1     // blend ramp half width

    class Win(val x0: Int, val y0: Int, val coreX: Int, val coreY: Int, val coreW: Int, val coreH: Int, val dstX: Int, val dstY: Int)

    /** Windows whose cores tile the [outW] x [outH] output exactly (no gaps, no overlap). Origins are multiples of 32. */
    fun plan(outW: Int, outH: Int): List<Win> {
        require(outW >= 2 && outH >= 2) { "output too small: ${outW}x$outH" }
        val nx = (outW + CORE - 1) / CORE
        val ny = (outH + CORE - 1) / CORE
        val list = ArrayList<Win>(nx * ny)
        for (ty in 0 until ny) for (tx in 0 until nx) {
            val dx = tx * CORE; val dy = ty * CORE
            list += Win(dx - PAD, dy - PAD, PAD, PAD, min(CORE, outW - dx), min(CORE, outH - dy), dx, dy)
        }
        return list
    }
}

// ------------------------------------------------------------------------------------------------------ contract

/** How a given network wants its two inputs: which mask value means "hole", and whether pixels are [0,1] or [-1,1]. */
data class ModelContract(val holeValue: Float, val minusOneToOne: Boolean) {
    fun describe() = "hole=${if (holeValue == 1f) "1" else "0"} range=${if (minusOneToOne) "[-1,1]" else "[0,1]"}"
}

object ContractCandidates {
    /** The Inpaint screen already runs these models, so the validated contract is tried first: LaMa / AOT-GAN hole=1, MIGAN keep=1, all [0,1]. */
    fun forKey(base: String): List<ModelContract> {
        val keepOne = ModelContract(0f, false)
        val holeOne = ModelContract(1f, false)
        return if (base.startsWith("migan"))
            listOf(keepOne, holeOne, ModelContract(0f, true), ModelContract(1f, true))
        else
            listOf(holeOne, keepOne, ModelContract(1f, true), ModelContract(0f, true))
    }
}

/** One network call in canonical form: image [0,1] RGB interleaved 512x512, hole mask 1 = hole. Contract mapping happens inside. */
interface TileModel {
    /** Returns false when the call failed (the daemon is down, the output file is missing, ...). */
    fun infer(image: FloatArray, hole: FloatArray, out: FloatArray, clock: PhaseClock): Boolean
}

object ContractProbe {
    const val N = 512
    private const val H0 = 176
    private const val H1 = 336          // 160 x 160 hole in the middle

    class Score(val outsideErr: Float, val holeErr: Float, val identityErr: Float, val finite: Boolean) {
        val ok: Boolean get() = finite && outsideErr <= 0.06f && holeErr <= 0.10f && holeErr <= 0.6f * identityErr
        override fun toString() = String.format(Locale.US, "outside %.3f hole %.3f (identity %.3f)%s", outsideErr, holeErr, identityErr, if (finite) "" else " NON-FINITE")
    }

    class Result(val contract: ModelContract?, val lines: List<String>)

    /** Smooth, strongly graded picture: a correct fill is easy to recognise, an identity or blank output is not. */
    fun truth(): FloatArray {
        val t = FloatArray(N * N * 3)
        for (y in 0 until N) for (x in 0 until N) {
            val i = (y * N + x) * 3
            t[i] = 0.15f + 0.70f * x / (N - 1)
            t[i + 1] = 0.15f + 0.70f * y / (N - 1)
            t[i + 2] = 0.50f + 0.30f * kotlin.math.sin(2.0 * Math.PI * (x + y) / N).toFloat()
        }
        return t
    }

    fun holeMask(): FloatArray {
        val m = FloatArray(N * N)
        for (y in H0 until H1) for (x in H0 until H1) m[y * N + x] = 1f
        return m
    }

    fun input(truth: FloatArray): FloatArray {
        val a = truth.copyOf()
        for (y in H0 until H1) for (x in H0 until H1) { val i = (y * N + x) * 3; a[i] = 0.5f; a[i + 1] = 0.5f; a[i + 2] = 0.5f }
        return a
    }

    fun score(truth: FloatArray, input: FloatArray, out: FloatArray): Score {
        var finite = true
        var so = 0.0; var no = 0L; var sh = 0.0; var nh = 0L; var si = 0.0
        for (y in 0 until N) for (x in 0 until N) {
            val inHole = y in H0 until H1 && x in H0 until H1
            val far = y < H0 - 16 || y >= H1 + 16 || x < H0 - 16 || x >= H1 + 16
            if (!inHole && !far) continue
            for (c in 0 until 3) {
                val i = (y * N + x) * 3 + c
                val o = out[i]
                if (!o.isFinite()) { finite = false; continue }
                val e = abs(o - truth[i]).toDouble()
                if (inHole) { sh += e; si += abs(input[i] - truth[i]); nh++ } else { so += e; no++ }
            }
        }
        return Score((so / max(1L, no)).toFloat(), (sh / max(1L, nh)).toFloat(), (si / max(1L, nh)).toFloat(), finite)
    }

    /** Tries the candidates in order and returns the first contract under which the network really fills the hole. */
    fun run(candidates: List<ModelContract>, make: (ModelContract) -> TileModel, clock: PhaseClock = PhaseClock()): Result {
        val truth = truth(); val hole = holeMask(); val input = input(truth)
        val lines = ArrayList<String>()
        val out = FloatArray(N * N * 3)
        for (c in candidates) {
            java.util.Arrays.fill(out, 0f)
            val ok = try { make(c).infer(input, hole, out, clock) } catch (e: Exception) { false }
            if (!ok) { lines += "${c.describe()}: model call failed"; continue }
            val s = score(truth, input, out)
            lines += "${c.describe()}: $s"
            if (s.ok) return Result(c, lines)
        }
        return Result(null, lines)
    }
}

// ------------------------------------------------------------------------------------------------------- gates

/** Index of every entry of the 14 doubles returned by the native compose call. */
object SrStat {
    const val SUM_ABS_D = 0; const val SUM_D2 = 1; const val SUM_NYQ2 = 2; const val SUM_HAD2 = 3; const val SUM_DC2 = 4
    const val SUM_ABS_OU = 5; const val SUM_OU = 6; const val SAMPLES = 7; const val PURE_HOLE = 8; const val IDENTITY_HOLE = 9
    const val CLAMPED = 10; const val NON_FINITE = 11; const val PHASES_USED = 12; const val PHASES_REJECTED = 13
    const val SIZE = 14
}

class SrAggregate {
    val v = DoubleArray(SrStat.SIZE)
    var windows = 0; private set
    fun add(stats: DoubleArray) {
        for (i in 0 until SrStat.SIZE) v[i] += stats[i]
        windows++
    }
    operator fun get(i: Int) = v[i]

    /** Mean |alpha * D| in 8-bit levels. */
    val detailLevels: Double get() = if (v[SrStat.SAMPLES] > 0) v[SrStat.SUM_ABS_D] / v[SrStat.SAMPLES] * 255.0 else 0.0
    val meshShare: Double get() = if (v[SrStat.SUM_HAD2] > 0) v[SrStat.SUM_NYQ2] / v[SrStat.SUM_HAD2] else 0.0
    val identityFraction: Double get() = if (v[SrStat.PURE_HOLE] > 0) v[SrStat.IDENTITY_HOLE] / v[SrStat.PURE_HOLE] else 0.0
    /** RMS change of the detail in the low-resolution domain, in 8-bit levels (what an LR-consistency check would see). */
    val lrChangeLevels: Double get() = if (v[SrStat.SAMPLES] > 0) sqrt(v[SrStat.SUM_DC2] / (v[SrStat.SAMPLES] / 4.0)) * 255.0 else 0.0
    val clampedFraction: Double get() = if (v[SrStat.SAMPLES] > 0) v[SrStat.CLAMPED] / v[SrStat.SAMPLES] else 0.0
    val rejectedPhaseFraction: Double get() {
        val total = v[SrStat.PHASES_USED] + v[SrStat.PHASES_REJECTED]
        return if (total > 0) v[SrStat.PHASES_REJECTED] / total else 1.0
    }
}

enum class SrVerdict { ACCEPT, NO_DETAIL, MESH, UNSTABLE, LR_BROKEN }

class SrGateResult(val verdict: SrVerdict, val reason: String, val metrics: String) {
    val accepted: Boolean get() = verdict == SrVerdict.ACCEPT
}

object SrGates {
    const val MESH_SHARE_MAX = 0.55          // white texture gives ~1/3, a stride-2 lattice gives ~1
    const val IDENTITY_MAX = 0.98
    const val REJECTED_PHASES_MAX = 0.34
    const val LR_CHANGE_MAX_LEVELS = 2.5
    const val DETAIL_MIN_LEVELS = 0.08
    const val CLAMPED_MAX = 0.5
    const val CAP_LEVELS = 6.0               // mean |alpha*D| allowed per window (8-bit levels); stronger windows are re-composed with a smaller alpha
    const val EARLY_WINDOWS = 6

    fun metrics(a: SrAggregate) = String.format(
        Locale.US, "detail %.2f lv | mesh %.2f | identity %.2f | LR change %.2f lv | clamped %.2f | rejected phases %.2f | windows %d",
        a.detailLevels, a.meshShare, a.identityFraction, a.lrChangeLevels, a.clampedFraction, a.rejectedPhaseFraction, a.windows
    )

    /** [early] = true only looks for the failures that are certain after a handful of windows (mesh, unstable model). */
    fun evaluate(a: SrAggregate, early: Boolean = false): SrGateResult {
        val m = metrics(a)
        if (a.windows == 0) return SrGateResult(SrVerdict.UNSTABLE, "no window was processed", m)
        if (a.rejectedPhaseFraction > REJECTED_PHASES_MAX)
            return SrGateResult(SrVerdict.UNSTABLE, "the model output was rejected in ${(a.rejectedPhaseFraction * 100).toInt()}% of the phases (NaN, black or blown-out)", m)
        if (a.meshShare > MESH_SHARE_MAX)
            return SrGateResult(SrVerdict.MESH, "the added detail is a pixel lattice (diagonal share %.2f)".format(Locale.US, a.meshShare), m)
        if (early) return SrGateResult(SrVerdict.ACCEPT, "early check passed", m)
        if (a.lrChangeLevels > LR_CHANGE_MAX_LEVELS)
            return SrGateResult(SrVerdict.LR_BROKEN, "the detail changes the low-resolution picture by %.1f levels rms".format(Locale.US, a.lrChangeLevels), m)
        if (a.clampedFraction > CLAMPED_MAX)
            return SrGateResult(SrVerdict.UNSTABLE, "more than half of the detail hits the soft clamp (the model fights the base)", m)
        if (a.identityFraction >= IDENTITY_MAX)
            return SrGateResult(SrVerdict.NO_DETAIL, "the model returned its input unchanged", m)
        if (a.detailLevels < DETAIL_MIN_LEVELS)
            return SrGateResult(SrVerdict.NO_DETAIL, "the model added no measurable detail", m)
        return SrGateResult(SrVerdict.ACCEPT, "accepted", m)
    }

    /** Alpha to re-compose a window with when its detail is stronger than the cap; null when it is fine. */
    fun cappedAlpha(alpha: Float, stats: DoubleArray): Float? {
        val samples = stats[SrStat.SAMPLES]
        if (samples <= 0) return null
        val levels = stats[SrStat.SUM_ABS_D] / samples * 255.0
        return if (levels > CAP_LEVELS) (alpha * CAP_LEVELS / levels).toFloat() else null
    }
}

// --------------------------------------------------------------------------------------------------- pass runner

class NsdtParams(
    val alpha: Float = 0.6f,
    val tau: Float = 0.08f,
    val driftLimit: Float = 0.30f,
    val cell: Int = SrWindows.CELL,
    val margin: Int = SrWindows.MARGIN
)

/** Native pixel kernels (see sr_core.h). Images are opaque handles so the runner can be tested without android.graphics. */
interface SrKernels {
    fun sampleBase(src: SrImage, x0: Int, y0: Int, win: Int, out: FloatArray): Boolean
    fun holeMask(x0: Int, y0: Int, win: Int, cell: Int, phase: Int, out: FloatArray): Boolean
    fun compose(u: FloatArray, o0: FloatArray?, o1: FloatArray?, win: Int, winX0: Int, winY0: Int, cell: Int, margin: Int,
                alpha: Float, tau: Float, driftLimit: Float, cx0: Int, cy0: Int, cw: Int, ch: Int,
                dst: SrImage, dstX: Int, dstY: Int): DoubleArray?
    fun writeBase(u: FloatArray, win: Int, cx0: Int, cy0: Int, cw: Int, ch: Int, dst: SrImage, dstX: Int, dstY: Int): Boolean
}

interface SrProgress {
    val cancelled: Boolean
    fun report(pass: Int, passes: Int, done: Int, total: Int, note: String)
}

class PassOutcome(val gate: SrGateResult, val aggregate: SrAggregate, val cancelled: Boolean, val earlyAbort: Boolean, val windows: Int, val modelFailures: Int)

class NsdtPass(private val k: SrKernels, private val model: TileModel, private val clock: PhaseClock, private val p: NsdtParams = NsdtParams()) {
    private val n = SrWindows.WIN
    private val u = FloatArray(n * n * 3)
    private val m0 = FloatArray(n * n)
    private val m1 = FloatArray(n * n)
    private val o0 = FloatArray(n * n * 3)
    private val o1 = FloatArray(n * n * 3)

    /** One 2x pass: [dst] must be exactly twice the size of [src]. Every pixel of [dst] is written (cores tile the output). */
    fun run(src: SrImage, dst: SrImage, pass: Int, passes: Int, progress: SrProgress?): PassOutcome {
        require(dst.width == src.width * 2 && dst.height == src.height * 2) { "dst must be 2x src" }
        val wins = SrWindows.plan(dst.width, dst.height)
        val agg = SrAggregate()
        var consecutiveFailures = 0
        var failures = 0
        for ((i, w) in wins.withIndex()) {
            if (progress?.cancelled == true) {
                return PassOutcome(SrGateResult(SrVerdict.UNSTABLE, "cancelled", SrGates.metrics(agg)), agg, true, false, i, failures)
            }
            progress?.report(pass, passes, i, wins.size, "window ${i + 1}/${wins.size}")
            val okBase = clock.prep {
                k.sampleBase(src, w.x0, w.y0, n, u) && k.holeMask(w.x0, w.y0, n, p.cell, 0, m0) && k.holeMask(w.x0, w.y0, n, p.cell, 1, m1)
            }
            check(okBase) { "native kernel refused window $i (${w.x0},${w.y0})" }
            val ok0 = model.infer(u, m0, o0, clock)
            val ok1 = model.infer(u, m1, o1, clock)
            if (!ok0 && !ok1) { consecutiveFailures++; failures++ } else consecutiveFailures = 0
            if (consecutiveFailures >= 3) {
                return PassOutcome(SrGateResult(SrVerdict.UNSTABLE, "the accelerator call failed three windows in a row", SrGates.metrics(agg)), agg, false, true, i + 1, failures)
            }
            val stats = clock.post {
                var s = k.compose(u, if (ok0) o0 else null, if (ok1) o1 else null, n, w.x0, w.y0, p.cell, p.margin, p.alpha, p.tau, p.driftLimit,
                    w.coreX, w.coreY, w.coreW, w.coreH, dst, w.dstX, w.dstY) ?: error("native compose failed for window $i")
                val capped = SrGates.cappedAlpha(p.alpha, s)
                if (capped != null) {
                    s = k.compose(u, if (ok0) o0 else null, if (ok1) o1 else null, n, w.x0, w.y0, p.cell, p.margin, capped, p.tau, p.driftLimit,
                        w.coreX, w.coreY, w.coreW, w.coreH, dst, w.dstX, w.dstY) ?: error("native compose failed for window $i")
                }
                s
            }
            agg.add(stats)
            if (agg.windows == SrGates.EARLY_WINDOWS) {
                val early = SrGates.evaluate(agg, early = true)
                if (!early.accepted) return PassOutcome(early, agg, false, true, i + 1, failures)
            }
        }
        progress?.report(pass, passes, wins.size, wins.size, "checking quality gates")
        return PassOutcome(SrGates.evaluate(agg), agg, false, false, wins.size, failures)
    }

    /** The classical 2x base only (what a rejected pass falls back to): no model call, every pixel written. */
    fun runClassical(src: SrImage, dst: SrImage, progress: SrProgress?, pass: Int, passes: Int) {
        require(dst.width == src.width * 2 && dst.height == src.height * 2) { "dst must be 2x src" }
        val wins = SrWindows.plan(dst.width, dst.height)
        for ((i, w) in wins.withIndex()) {
            if (progress?.cancelled == true) return
            progress?.report(pass, passes, i, wins.size, "classical base ${i + 1}/${wins.size}")
            val ok = clock.post {
                k.sampleBase(src, w.x0, w.y0, n, u) && k.writeBase(u, n, w.coreX, w.coreY, w.coreW, w.coreH, dst, w.dstX, w.dstY)
            }
            check(ok) { "native kernel refused classical window $i" }
        }
    }
}
