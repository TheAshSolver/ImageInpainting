package qidk.fix

import java.io.File

/** One snapshot of the live signals the router puts into its state vector. Missing sensors read 0 and are named in [notes]. */
class Telemetry(
    val siliconTempC: Float,
    val batteryPct: Float,
    val cpuBusy: Float,      // 0..1, all cores
    val gpuBusy: Float,      // 0..1, Adreno
    val notes: String,
    val takenAtMs: Long
) {
    companion object {
        val EMPTY = Telemetry(0f, 0f, 0f, 0f, "no sample yet", 0L)
    }
}

/**
 * Samples temperature / battery / CPU and GPU load on a background thread every [periodMs]; the router reads [latest]
 * (a volatile reference) so a routing decision never touches sysfs. Every reader fails soft: an unreadable file means
 * "0 + a note", never an exception.
 *
 * Temperature: the same NPU-side zones the benchmark logged (nsphvx-0, nsphmx-0, cdsp, npu), then the hottest of
 * cpuss/gpuss/nsp/skin zones, then thermal_zone0. If SELinux hides /sys/class/thermal from the app on a given build,
 * [Telemetry.notes] says "temp:unreadable" - ask the resident daemon (it runs as shell) instead of loosening SELinux.
 */
class TelemetryPoller(
    private val batteryPct: () -> Float,
    private val periodMs: Long = 500L,
    private val thermalRoot: File = File("/sys/class/thermal"),
    private val procStat: File = File("/proc/stat"),
    private val kgslDir: File = File("/sys/class/kgsl/kgsl-3d0")
) {
    @Volatile var latest: Telemetry = Telemetry.EMPTY
        private set
    @Volatile private var running = false
    private var thread: Thread? = null
    private var prevTotal = 0L
    private var prevIdle = 0L

    fun start() {
        if (running) return
        running = true
        thread = Thread({
            while (running) {
                try { latest = sample() } catch (_: Throwable) { }
                try { Thread.sleep(periodMs) } catch (_: InterruptedException) { break }
            }
        }, "telemetry").apply { isDaemon = true; start() }
    }

    fun stop() { running = false; thread?.interrupt() }

    fun sample(): Telemetry {
        val notes = ArrayList<String>()
        val (temp, src) = readSiliconTemp()
        if (temp == null) notes.add("temp:unreadable") else notes.add("temp:$src")
        val cpu = readCpuBusy()
        if (cpu == null) notes.add("cpu:unreadable")
        val gpu = readGpuBusy()
        if (gpu == null) notes.add("gpu:unreadable")
        return Telemetry(temp ?: 0f, batteryPct().coerceIn(0f, 100f), cpu ?: 0f, gpu ?: 0f, notes.joinToString(" "), System.currentTimeMillis())
    }

    private var preferredTemp: File? = null
    private var preferredName = ""

    private fun readSiliconTemp(): Pair<Float?, String> {
        try {
            preferredTemp?.let { f -> parseMilliC(f.readText())?.let { return it to preferredName } }
            val zones = thermalRoot.listFiles { f -> f.name.startsWith("thermal_zone") }?.sortedBy { it.name } ?: return null to ""
            fun type(z: File) = try { File(z, "type").readText().trim().lowercase() } catch (_: Exception) { "" }
            for (want in listOf("nsphvx-0", "nsphmx-0", "cdsp", "npu")) {
                val z = zones.firstOrNull { type(it) == want } ?: continue
                val t = parseMilliC(File(z, "temp").readText()) ?: continue
                preferredTemp = File(z, "temp"); preferredName = want
                return t to want
            }
            var hottest: Float? = null
            var name = ""
            for (z in zones) {
                val ty = type(z)
                if (!(ty.startsWith("cpu") || ty.startsWith("gpu") || ty.startsWith("nsp") || ty.contains("skin"))) continue
                val t = try { parseMilliC(File(z, "temp").readText()) } catch (_: Exception) { null } ?: continue
                if (hottest == null || t > hottest) { hottest = t; name = ty }
            }
            if (hottest != null) return hottest to "hottest:$name"
            val t0 = parseMilliC(File(zones.first(), "temp").readText())
            return t0 to "zone0"
        } catch (_: Exception) {
            return null to ""
        }
    }

    private fun readCpuBusy(): Float? {
        return try {
            val line = procStat.bufferedReader().use { it.readLine() } ?: return null
            val (total, idle) = parseProcStatLine(line) ?: return null
            val dt = total - prevTotal; val di = idle - prevIdle
            val first = prevTotal == 0L
            prevTotal = total; prevIdle = idle
            if (first || dt <= 0) null else (1f - di.toFloat() / dt.toFloat()).coerceIn(0f, 1f)
        } catch (_: Exception) { null }
    }

    private fun readGpuBusy(): Float? {
        return try {
            val pct = File(kgslDir, "gpu_busy_percentage")
            if (pct.canRead()) parseGpuPercent(pct.readText()) else parseGpuBusyPair(File(kgslDir, "gpubusy").readText())
        } catch (_: Exception) { null }
    }

    companion object {
        /** sysfs temps are milli-degrees C on Qualcomm; plain degrees are accepted when the value is small. */
        fun parseMilliC(text: String): Float? {
            val v = text.trim().toFloatOrNull() ?: return null
            val c = if (v > 1000f || v < -1000f) v / 1000f else v
            return if (c in -40f..150f) c else null
        }

        /** "cpu  user nice system idle iowait irq softirq steal ..." -> (total, idle + iowait). */
        fun parseProcStatLine(line: String): Pair<Long, Long>? {
            val p = line.trim().split(Regex("\\s+"))
            if (p.size < 5 || p[0] != "cpu") return null
            val n = p.drop(1).mapNotNull { it.toLongOrNull() }
            if (n.size < 4) return null
            val idle = n[3] + (n.getOrNull(4) ?: 0L)
            return n.take(8).sum() to idle
        }

        fun parseGpuPercent(text: String): Float? = text.trim().removeSuffix("%").trim().toFloatOrNull()?.let { (it / 100f).coerceIn(0f, 1f) }

        /** kgsl "gpubusy" holds "<busy> <total>" cycle counters of the last interval. */
        fun parseGpuBusyPair(text: String): Float? {
            val p = text.trim().split(Regex("\\s+")).mapNotNull { it.toLongOrNull() }
            return if (p.size >= 2 && p[1] > 0) (p[0].toFloat() / p[1].toFloat()).coerceIn(0f, 1f) else null
        }
    }
}
