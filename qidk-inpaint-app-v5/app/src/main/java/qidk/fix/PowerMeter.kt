package qidk.fix

import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Per-run telemetry shown on the post-run card of BOTH screens (Inpaint and Super-Resolution).
 * Times are wall-clock phases of the app (t_prep: building tensors / files, t_infer: time inside the accelerator
 * call(s), t_post: decoding + compositing). Power fields are INCREMENTAL (what this run added on top of the idle
 * baseline measured just before it), energy is the integral of that power over the run, not "mean power x latency".
 * A null means "not measurable"; [powerNote] says why, the UI prints it instead of inventing a number.
 */
data class RunTelemetry(
    val tPrepMs: Long,
    val tInferMs: Long,
    val tPostMs: Long,
    val tTotalMs: Long,
    val startTempC: Float?,
    val peakTempC: Float?,
    val deltaTempC: Float?,
    val thermalGroup: String,
    val activeWatts: Float?,
    val batteryJoules: Float?,
    val grossWatts: Float?,
    val powerNote: String,
    val core: String
) {
    /** Rows of the post-run card, in display order: (label, value). */
    fun rows(): List<Pair<String, String>> {
        fun ms(v: Long) = if (v >= 10_000) String.format(Locale.US, "%.2f s", v / 1000.0) else "$v ms"
        val temp = if (deltaTempC != null && startTempC != null && peakTempC != null)
            String.format(Locale.US, "%+.1f °C  (%.1f → %.1f, %s)", deltaTempC, startTempC, peakTempC, thermalGroup)
        else "n/a ($thermalGroup unreadable)"
        val w = activeWatts?.let { String.format(Locale.US, "%.2f W", it) } ?: "n/a"
        val j = batteryJoules?.let { String.format(Locale.US, "%.2f J", it) } ?: "n/a"
        return listOf(
            "t_prep" to ms(tPrepMs),
            "t_infer" to ms(tInferMs),
            "t_post" to ms(tPostMs),
            "t_total" to ms(tTotalMs),
            "ΔT silicon" to temp,
            "Active power" to w,
            "Battery drain" to j,
            "Core" to core
        )
    }
}

/** One reading of the battery fuel gauge. current in microamps (sign is device specific), voltage in microvolts. */
class PowerSample(val tNs: Long, val currentUa: Long, val voltageUv: Long)

class PowerSummary(val activeWatts: Float?, val joules: Float?, val grossWatts: Float?, val note: String)

/**
 * Pure maths of the power measurement.
 *
 * The raw battery current mixes everything the phone does (screen, radios, USB charging). Two facts make a trustworthy
 * number possible anyway: (1) a baseline taken right before the run removes the constant part, including the charger;
 * (2) only the SIGN of the load direction is needed, and it is calibrated once with a known CPU burn (current goes
 * the same way for every load). Incremental power = loadSign * (I - I_baseline) * V.
 */
object PowerIntegrator {
    const val MIN_RUN_MS = 1000L
    const val MIN_BASELINE_SAMPLES = 3
    private const val UA_V_TO_W = 1e-12

    /** +1 when a heavier load RAISES current_now, -1 when it lowers it, 0 when the burn was not visible. */
    fun signFromBurn(baselineMeanUa: Double, burnMeanUa: Double, minDeltaUa: Double = 30_000.0): Int {
        val d = burnMeanUa - baselineMeanUa
        return if (abs(d) < minDeltaUa) 0 else if (d > 0) 1 else -1
    }

    fun summarize(baseline: List<PowerSample>, run: List<PowerSample>, tStartNs: Long, tEndNs: Long, loadSign: Int): PowerSummary {
        val durS = (tEndNs - tStartNs) / 1e9
        if (run.isEmpty() || durS <= 0.0) return PowerSummary(null, null, null, "no fuel-gauge samples during the run")
        val gross = run.sumOf { abs(it.currentUa.toDouble()) * it.voltageUv.toDouble() * UA_V_TO_W } / run.size
        val grossF = gross.toFloat()
        if (loadSign == 0) {
            return PowerSummary(null, null, grossF, "load direction not calibrated: the charger cannot be separated from the load")
        }
        if (baseline.size < MIN_BASELINE_SAMPLES) {
            return PowerSummary(null, null, grossF, "no idle baseline yet (keep the app open for 2 s before running)")
        }
        // The fuel gauge refreshes about once a second. A run shorter than that, or a run during which the reading never left
        // its last pre-run value, carries no information about the load.
        val lastBefore = baseline.last().currentUa
        val gaugeMoved = run.any { it.currentUa != lastBefore }
        if (durS * 1000.0 < MIN_RUN_MS) {
            return PowerSummary(null, null, grossF,
                "run shorter than the battery gauge update period (${String.format(Locale.US, "%.2f", durS)} s): energy not resolvable")
        }
        if (!gaugeMoved) {
            return PowerSummary(null, null, grossF, "the battery gauge did not update during the run: energy not resolvable")
        }

        val base = baseline.sumOf { it.currentUa.toDouble() } / baseline.size
        val sd = sqrt(baseline.sumOf { (it.currentUa - base) * (it.currentUa - base) } / baseline.size)
        val vMean = run.sumOf { it.voltageUv.toDouble() } / run.size

        // Piecewise-linear incremental power over [tStart, tEnd]; the first / last value is held to the window edges.
        val ts = DoubleArray(run.size) { (run[it].tNs - tStartNs) / 1e9 }
        val ps = DoubleArray(run.size) { loadSign * (run[it].currentUa - base) * run[it].voltageUv * UA_V_TO_W }
        var energy = ps[0] * max(0.0, ts[0])
        for (i in 0 until run.size - 1) energy += 0.5 * (ps[i] + ps[i + 1]) * (ts[i + 1] - ts[i])
        energy += ps[run.size - 1] * max(0.0, durS - ts[run.size - 1])

        val noiseW = 2.0 * sd * vMean * UA_V_TO_W
        val watts = energy / durS
        val note = String.format(Locale.US, "incremental over a %.0f mA idle baseline, noise ±%.2f W", base / 1000.0, noiseW)
        return if (watts < 0.0)
            PowerSummary(0f, 0f, grossF, "$note; reading below the baseline, reported as 0")
        else PowerSummary(watts.toFloat(), energy.toFloat(), grossF, note)
    }
}

/** Thermal zones of the silicon that did the work, instead of "the hottest of every sensor in the phone". */
object ThermalZones {
    enum class Group(val label: String) {
        NPU("NPU/HVX"), GPU("GPU"), CPU("CPU"), SOC("SoC")
    }

    class Zone(val type: String, val tempFile: File)

    private val npu = Regex("nsp|cdsp|npu|hvx|hmx", RegexOption.IGNORE_CASE)
    private val gpu = Regex("gpu", RegexOption.IGNORE_CASE)
    private val cpu = Regex("^cpu|cpuss", RegexOption.IGNORE_CASE)
    private val soc = Regex("cpu|gpu|nsp|cdsp|npu|ddr|video|aoss|mdmss|camera", RegexOption.IGNORE_CASE)
    private val deny = Regex("battery|skin|xo-therm|quiet|charger|chg|usb|bcl|bms|sdr|pa-|therm-adc|case|back", RegexOption.IGNORE_CASE)

    fun scan(root: File = File("/sys/class/thermal")): List<Zone> {
        val dirs = root.listFiles { f -> f.name.startsWith("thermal_zone") } ?: return emptyList()
        return dirs.sortedBy { it.name }.mapNotNull { d ->
            val type = try { File(d, "type").readText().trim() } catch (_: Exception) { return@mapNotNull null }
            val temp = File(d, "temp")
            if (type.isEmpty() || !temp.exists()) null else Zone(type, temp)
        }
    }

    /** Zones of [group]; an empty group falls back to the SoC set. Never contains battery / skin / charger sensors. */
    fun select(all: List<Zone>, group: Group): List<Zone> {
        val usable = all.filter { !deny.containsMatchIn(it.type) }
        val wanted = when (group) {
            Group.NPU -> usable.filter { npu.containsMatchIn(it.type) }
            Group.GPU -> usable.filter { gpu.containsMatchIn(it.type) }
            Group.CPU -> usable.filter { cpu.containsMatchIn(it.type) }
            Group.SOC -> usable.filter { soc.containsMatchIn(it.type) }
        }
        return if (wanted.isNotEmpty()) wanted else usable.filter { soc.containsMatchIn(it.type) }
    }

    /** Degrees Celsius, or null when unreadable or implausible (sensors report 0 / -274000 when powered down). */
    fun readC(z: Zone): Float? = try {
        val v = z.tempFile.readText().trim().toFloatOrNull()
        val c = if (v == null) null else if (kotlin.math.abs(v) >= 1000f) v / 1000f else v
        if (c != null && c > 5f && c < 130f) c else null
    } catch (_: Exception) { null }

    fun maxC(zones: List<Zone>): Float? = zones.mapNotNull { readC(it) }.maxOrNull()
}

/** Wall-clock phase accumulator (t_prep / t_infer / t_post / t_total). */
class PhaseClock {
    private val t0 = System.nanoTime()
    var prepNs = 0L; private set
    var inferNs = 0L; private set
    var postNs = 0L; private set

    inline fun <T> prep(block: () -> T): T { val s = System.nanoTime(); try { return block() } finally { addPrep(System.nanoTime() - s) } }
    inline fun <T> infer(block: () -> T): T { val s = System.nanoTime(); try { return block() } finally { addInfer(System.nanoTime() - s) } }
    inline fun <T> post(block: () -> T): T { val s = System.nanoTime(); try { return block() } finally { addPost(System.nanoTime() - s) } }

    @PublishedApi internal fun addPrep(ns: Long) { prepNs += ns }
    @PublishedApi internal fun addInfer(ns: Long) { inferNs += ns }
    @PublishedApi internal fun addPost(ns: Long) { postNs += ns }

    /** Adds phases that were timed with plain nanoTime stamps (where wrapping the code in a lambda would be intrusive). */
    fun addAll(prepNs: Long, inferNs: Long, postNs: Long) {
        prepNs.let { this.prepNs += max(0L, it) }
        inferNs.let { this.inferNs += max(0L, it) }
        postNs.let { this.postNs += max(0L, it) }
    }

    /** The accelerator reported its own latency (resident daemon "OK <ms>"): count it as inference time. */
    fun addInferMs(ms: Long) { addInfer(ms * 1_000_000L) }

    fun totalMs(): Long = (System.nanoTime() - t0) / 1_000_000L
    fun prepMs() = prepNs / 1_000_000L
    fun inferMs() = inferNs / 1_000_000L
    fun postMs() = postNs / 1_000_000L
}

/**
 * Runtime sampler. Runs at 5 Hz while the app is in the foreground (start() in onResume, stop() in onPause) so an idle
 * baseline always exists when the user taps Run; a run adds nothing but a begin/end bookmark.
 */
class PowerMonitor(
    private val currentFile: File = File("/sys/class/power_supply/battery/current_now"),
    private val voltageFile: File = File("/sys/class/power_supply/battery/voltage_now"),
    private val thermalRoot: File = File("/sys/class/thermal"),
    private val periodMs: Long = 200L
) {
    @Volatile var loadSign: Int = 0

    private class Tick(val tNs: Long, val currentUa: Long?, val voltageUv: Long?, val temps: Map<ThermalZones.Group, Float?>)

    private val ticks = ArrayList<Tick>()
    private val zones: List<ThermalZones.Zone> by lazy { ThermalZones.scan(thermalRoot) }
    private val groupZones: Map<ThermalZones.Group, List<ThermalZones.Zone>> by lazy {
        ThermalZones.Group.values().associateWith { ThermalZones.select(zones, it) }
    }
    @Volatile private var worker: Thread? = null

    fun start() {
        if (worker != null) return
        val t = Thread({
            while (!Thread.currentThread().isInterrupted) {
                sampleOnce()
                try { Thread.sleep(periodMs) } catch (_: InterruptedException) { break }
            }
        }, "qidk-power").apply { isDaemon = true }
        worker = t
        t.start()
    }

    fun stop() { worker?.interrupt(); worker = null }

    @Synchronized private fun add(t: Tick) {
        ticks.add(t)
        val cutoff = t.tNs - 300_000_000_000L        // five minutes of history: enough for the longest 8x run
        while (ticks.size > 64 && ticks[0].tNs < cutoff) ticks.removeAt(0)
    }

    /** Public for tests and for the single sample taken at begin()/end(). */
    fun sampleOnce() {
        val i = readLong(currentFile)
        val v = readLong(voltageFile)
        val temps = ThermalZones.Group.values().associateWith { ThermalZones.maxC(groupZones[it].orEmpty()) }
        add(Tick(System.nanoTime(), i, v, temps))
    }

    private fun readLong(f: File): Long? = try { f.readText().trim().toLongOrNull() } catch (_: Exception) { null }

    class Token internal constructor(internal val tStartNs: Long, internal val group: ThermalZones.Group, internal val startTempC: Float?)

    fun begin(group: ThermalZones.Group): Token {
        sampleOnce()
        val now = System.nanoTime()
        return Token(now, group, lastTemp(group, now - 2_000_000_000L, now))
    }

    /** Mean of the group temperature over the window, null when there is no reading. */
    @Synchronized private fun lastTemp(g: ThermalZones.Group, fromNs: Long, toNs: Long): Float? {
        val v = ticks.filter { it.tNs in fromNs..toNs }.mapNotNull { it.temps[g] }
        return if (v.isEmpty()) null else (v.sum() / v.size)
    }

    @Synchronized private fun slice(fromNs: Long, toNs: Long) = ticks.filter { it.tNs in fromNs..toNs }

    fun end(token: Token, phases: PhaseClock, core: String): RunTelemetry {
        sampleOnce()
        val tEnd = System.nanoTime()
        val run = slice(token.tStartNs, tEnd)
        val base = slice(token.tStartNs - 2_000_000_000L, token.tStartNs - 1)
        fun samples(l: List<Tick>) = l.mapNotNull { if (it.currentUa != null && it.voltageUv != null) PowerSample(it.tNs, it.currentUa, it.voltageUv) else null }
        val sum = PowerIntegrator.summarize(samples(base), samples(run), token.tStartNs, tEnd, loadSign)
        val peak = run.mapNotNull { it.temps[token.group] }.maxOrNull()
        val delta = if (peak != null && token.startTempC != null) peak - token.startTempC else null
        return RunTelemetry(
            tPrepMs = phases.prepMs(), tInferMs = phases.inferMs(), tPostMs = phases.postMs(), tTotalMs = phases.totalMs(),
            startTempC = token.startTempC, peakTempC = peak, deltaTempC = delta,
            thermalGroup = token.group.label,
            activeWatts = sum.activeWatts, batteryJoules = sum.joules, grossWatts = sum.grossWatts, powerNote = sum.note, core = core
        )
    }

    /** Mean current over the last [ms] milliseconds, null without readings. */
    @Synchronized fun meanCurrentUa(fromNs: Long, toNs: Long): Double? {
        val v = ticks.filter { it.tNs in fromNs..toNs }.mapNotNull { it.currentUa }
        return if (v.isEmpty()) null else v.sum().toDouble() / v.size
    }

    companion object {
        @Volatile private var instance: PowerMonitor? = null
        fun get(): PowerMonitor = instance ?: synchronized(this) { instance ?: PowerMonitor().also { instance = it } }
    }
}

/**
 * Finds out which way current_now moves under load by burning every core for a moment. Called once per installation
 * (the result is a property of the device, not of the run) and stored by the caller.
 */
object PowerCalibration {
    fun measureSign(monitor: PowerMonitor, burnMs: Long = 1600L): Int {
        monitor.start()
        val t0 = System.nanoTime()
        try { Thread.sleep(1200L) } catch (_: InterruptedException) { return 0 }
        val tBurn = System.nanoTime()
        val stop = java.util.concurrent.atomic.AtomicBoolean(false)
        val n = max(2, Runtime.getRuntime().availableProcessors())
        val threads = (0 until n).map {
            Thread {
                var x = 1.000001
                while (!stop.get()) { x = x * 1.0000001 + 1e-9; if (x > 1e6) x = 1.0 }
            }.apply { isDaemon = true; start() }
        }
        try { Thread.sleep(burnMs) } catch (_: InterruptedException) { }
        val tEnd = System.nanoTime()
        stop.set(true)
        threads.forEach { try { it.join(200) } catch (_: InterruptedException) { } }
        val base = monitor.meanCurrentUa(t0, tBurn - 1) ?: return 0
        // skip the first 400 ms of the burn: the gauge lags
        val burn = monitor.meanCurrentUa(tBurn + 400_000_000L, tEnd) ?: return 0
        return PowerIntegrator.signFromBurn(base, burn)
    }
}
