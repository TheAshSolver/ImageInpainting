package qidk.fix

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files
import java.util.Random

class PowerMeterTest {
    private val v = 4_000_000L // 4.0 V in microvolts

    private fun series(startNs: Long, n: Int, stepMs: Long, currentUa: (Int) -> Long) =
        List(n) { PowerSample(startNs + it * stepMs * 1_000_000L, currentUa(it), v) }

    @Test fun dischargingPhoneNegativeSignConvention() {
        // current_now is NEGATIVE while discharging; a heavier load makes it more negative: loadSign = -1.
        val base = series(-2_000_000_000L, 10, 200) { -500_000L }
        val run = series(0L, 21, 200) { -1_500_000L }                // 1 A extra at 4 V for 4 s
        val s = PowerIntegrator.summarize(base, run, 0L, 4_000_000_000L, loadSign = -1)
        assertEquals(4.0f, s.activeWatts!!, 0.01f)
        assertEquals(16.0f, s.joules!!, 0.05f)
    }

    @Test fun chargerDoesNotContaminateTheIncrementalReading() {
        // plugged in: the battery CHARGES at +800 mA while idle; the run takes 500 mA of that away: sign -1
        val base = series(-2_000_000_000L, 10, 200) { 800_000L }
        val run = series(0L, 16, 200) { 300_000L }                   // 3 s
        val s = PowerIntegrator.summarize(base, run, 0L, 3_000_000_000L, loadSign = -1)
        assertEquals(2.0f, s.activeWatts!!, 0.01f)                   // 0.5 A * 4 V
        assertEquals(6.0f, s.joules!!, 0.05f)
        // the gross figure is |I|*V = 1.2 W and is only a diagnostic
        assertEquals(1.2f, s.grossWatts!!, 0.01f)
    }

    @Test fun energyIsTheIntegralNotMeanPowerTimesLatency() {
        // a ramp: 0 W -> 4 W over 2 s then flat 4 W for 2 s. Integral = 4 + 8 = 12 J, mean power * latency would match, but a
        // run with a gap in sampling must still integrate over the real timestamps.
        val base = series(-2_000_000_000L, 10, 200) { 0L }
        val ts = listOf(0L, 500L, 1000L, 1500L, 2000L, 3000L, 4000L)   // uneven spacing
        val amps = listOf(0L, 250_000L, 500_000L, 750_000L, 1_000_000L, 1_000_000L, 1_000_000L)
        val run = ts.indices.map { PowerSample(ts[it] * 1_000_000L, amps[it], v) }
        val s = PowerIntegrator.summarize(base, run, 0L, 4_000_000_000L, loadSign = 1)
        assertEquals(12.0f, s.joules!!, 0.05f)
        assertEquals(3.0f, s.activeWatts!!, 0.02f)
    }

    @Test fun shortRunsAreReportedAsNotResolvableInsteadOfInvented() {
        val base = series(-2_000_000_000L, 10, 200) { 100_000L }
        val run = series(0L, 3, 200) { 900_000L }
        val s = PowerIntegrator.summarize(base, run, 0L, 400_000_000L, loadSign = 1)
        assertNull(s.activeWatts); assertNull(s.joules)
        assertNotNull(s.grossWatts)
        assertTrue(s.note, s.note.contains("gauge update period"))
    }

    @Test fun staleGaugeIsNotMeasurable() {
        val base = series(-2_000_000_000L, 10, 200) { 100_000L }
        val run = series(0L, 15, 200) { 100_000L }                   // 3 s but the reading never left its pre-run value
        val s = PowerIntegrator.summarize(base, run, 0L, 3_000_000_000L, loadSign = 1)
        assertNull(s.activeWatts)
        assertTrue(s.note, s.note.contains("did not update"))
    }

    @Test fun uncalibratedSignGivesGrossOnly() {
        val base = series(-2_000_000_000L, 10, 200) { 100_000L }
        val run = series(0L, 21, 200) { 900_000L + it }
        val s = PowerIntegrator.summarize(base, run, 0L, 4_000_000_000L, loadSign = 0)
        assertNull(s.activeWatts)
        assertTrue(s.note.contains("not calibrated"))
        assertTrue(s.grossWatts!! > 3f)
    }

    @Test fun missingBaselineIsSaid() {
        val run = series(0L, 21, 200) { 900_000L + it * 1000 }
        val s = PowerIntegrator.summarize(emptyList(), run, 0L, 4_000_000_000L, loadSign = 1)
        assertNull(s.activeWatts)
        assertTrue(s.note.contains("baseline"))
    }

    @Test fun readingBelowTheBaselineIsClampedToZeroAndSaid() {
        val rnd = Random(1)
        val base = series(-2_000_000_000L, 10, 200) { 500_000L + rnd.nextInt(2000) }
        val run = series(0L, 21, 200) { 490_000L + rnd.nextInt(2000) }
        val s = PowerIntegrator.summarize(base, run, 0L, 4_000_000_000L, loadSign = 1)
        assertEquals(0f, s.activeWatts!!, 0f)
        assertTrue(s.note.contains("below the baseline"))
    }

    @Test fun burnSign() {
        assertEquals(1, PowerIntegrator.signFromBurn(300_000.0, 900_000.0))
        assertEquals(-1, PowerIntegrator.signFromBurn(-300_000.0, -900_000.0))
        assertEquals(-1, PowerIntegrator.signFromBurn(800_000.0, 200_000.0))   // charging: load lowers the net charge
        assertEquals(0, PowerIntegrator.signFromBurn(300_000.0, 310_000.0))
    }

    // ------------------------------------------------------------------------------------------------ thermal

    private fun zone(root: File, idx: Int, type: String, milli: String) {
        val d = File(root, "thermal_zone$idx").apply { mkdirs() }
        File(d, "type").writeText(type + "\n")
        File(d, "temp").writeText(milli + "\n")
    }

    @Test fun thermalGroupsPickTheSiliconThatDidTheWork() {
        val root = Files.createTempDirectory("tz").toFile()
        try {
            zone(root, 0, "cpu-0-0-0", "61000")
            zone(root, 1, "cpuss-0", "58000")
            zone(root, 2, "gpuss-0", "52000")
            zone(root, 3, "nsphvx-0", "47000")
            zone(root, 4, "nsphmx-1", "49500")
            zone(root, 5, "battery", "39000")
            zone(root, 6, "skin-msm-therm", "36000")
            zone(root, 7, "ddr", "50000")
            zone(root, 8, "xo-therm", "35000")
            zone(root, 9, "nsphmx-2", "-274000")          // powered-down sensor
            val all = ThermalZones.scan(root)
            assertEquals(10, all.size)
            val npu = ThermalZones.select(all, ThermalZones.Group.NPU)
            assertEquals(listOf("nsphvx-0", "nsphmx-1", "nsphmx-2"), npu.map { it.type })
            assertEquals(49.5f, ThermalZones.maxC(npu)!!, 0.01f)        // not the 61 degrees CPU, not the battery
            assertEquals(52.0f, ThermalZones.maxC(ThermalZones.select(all, ThermalZones.Group.GPU))!!, 0.01f)
            assertEquals(61.0f, ThermalZones.maxC(ThermalZones.select(all, ThermalZones.Group.CPU))!!, 0.01f)
            val soc = ThermalZones.select(all, ThermalZones.Group.SOC).map { it.type }
            assertFalse(soc.any { it.contains("battery") || it.contains("skin") || it.contains("xo") })
        } finally { root.deleteRecursively() }
    }

    @Test fun emptyGroupFallsBackToTheSocSetAndNeverToTheBattery() {
        val root = Files.createTempDirectory("tz2").toFile()
        try {
            zone(root, 0, "cpu-0-0-0", "55000")
            zone(root, 1, "battery", "41000")
            val all = ThermalZones.scan(root)
            val npu = ThermalZones.select(all, ThermalZones.Group.NPU)
            assertEquals(listOf("cpu-0-0-0"), npu.map { it.type })
        } finally { root.deleteRecursively() }
    }

    // --------------------------------------------------------------------------------------------- run telemetry

    @Test fun monitorProducesAFullRecordFromSysfsLikeFiles() {
        val root = Files.createTempDirectory("pm").toFile()
        try {
            zone(File(root, "thermal"), 0, "nsphvx-0", "44000")
            File(root, "i").writeText("-500000\n"); File(root, "v").writeText("4000000\n")
            val m = PowerMonitor(File(root, "i"), File(root, "v"), File(root, "thermal"), periodMs = 10)
            m.loadSign = -1
            repeat(5) { m.sampleOnce(); Thread.sleep(5) }
            val tok = m.begin(ThermalZones.Group.NPU)
            val clock = PhaseClock()
            clock.prep { Thread.sleep(3) }
            clock.infer { Thread.sleep(6) }
            clock.post { Thread.sleep(2) }
            File(File(root, "thermal"), "thermal_zone0/temp").writeText("47500\n")
            m.sampleOnce()
            val t = m.end(tok, clock, "Hexagon HTP v79")
            assertTrue(t.tPrepMs >= 3 && t.tInferMs >= 6 && t.tPostMs >= 2)
            assertTrue(t.tTotalMs >= t.tPrepMs + t.tInferMs + t.tPostMs - 1)
            assertEquals(3.5f, t.deltaTempC!!, 0.01f)
            assertEquals("NPU/HVX", t.thermalGroup)
            assertNull(t.activeWatts)                                  // far shorter than the gauge period: not invented
            val labels = t.rows().map { it.first }
            assertEquals(listOf("t_prep", "t_infer", "t_post", "t_total", "ΔT silicon", "Active power", "Battery drain", "Core"), labels)
        } finally { root.deleteRecursively() }
    }
}
