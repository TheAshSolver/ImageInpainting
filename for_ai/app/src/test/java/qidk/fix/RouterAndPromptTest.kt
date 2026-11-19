package qidk.fix

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File

class RouterAndPromptTest {

    // ---------------------------------------------------------------- helpers
    private fun rgb(r: Int, g: Int, b: Int) = (0xFF shl 24) or (r shl 16) or (g shl 8) or b

    private fun image(w: Int, h: Int, f: (Int, Int) -> Int) = IntArray(w * h) { f(it % w, it / w) }

    private fun disk(w: Int, h: Int, cx: Int, cy: Int, r: Int) =
        ByteArray(w * h) { val x = it % w - cx; val y = it / w - cy; if (x * x + y * y <= r * r) 1 else 0 }

    // ---------------------------------------------------------------- Smart SD prompt
    @Test fun smartPromptReadsTheRing() {
        val w = 200; val h = 200
        val sky = image(w, h) { x, _ -> rgb(90 + x / 2, 140 + x / 2, 235) }                        // blue, clearly brighter to the right
        val top = SmartPrompt.analyse(sky, disk(w, h, 100, 45, 20), w, h)!!
        assertTrue(top.toString(), top.surface.contains("sky"))
        val low = SmartPrompt.analyse(sky, disk(w, h, 100, 140, 20), w, h)!!
        assertTrue(low.toString(), low.surface.contains("water"))
        assertTrue(low.texture.contains("smooth"))
        assertTrue("brighter right -> light from the right: ${low.light}", low.light.contains("right"))

        val grass = image(w, h) { x, y -> rgb(40 + (x * 7 + y * 13) % 30, 120 + (x * 11 + y * 5) % 40, 50) }
        assertTrue(SmartPrompt.analyse(grass, disk(w, h, 100, 100, 20), w, h)!!.surface.contains("foliage"))
        val night = image(w, h) { _, _ -> rgb(20, 20, 22) }
        assertTrue(SmartPrompt.analyse(night, disk(w, h, 100, 100, 20), w, h)!!.light.contains("dim"))
    }

    @Test fun smartPromptKeepsUserTextFirstAndStaysShort() {
        val w = 160; val h = 160
        val img = image(w, h) { _, _ -> rgb(70, 140, 200) }
        val d = SmartPrompt.analyse(img, disk(w, h, 80, 110, 18), w, h)!!
        val p = SmartPrompt.build("empty waters", d)
        assertTrue(p, p.startsWith("empty waters, "))
        assertTrue(p, p.contains("photograph"))
        assertTrue(p.split(Regex("\\s+")).size <= SmartPrompt.MAX_WORDS)
        assertTrue(SmartPrompt.build("", d).startsWith("a photograph of "))
        assertTrue(SmartPrompt.negative("boat").startsWith("boat, painting"))
        assertTrue(SmartPrompt.negative("").startsWith("painting"))
        assertNull("no hole -> no context", SmartPrompt.analyse(img, ByteArray(w * h), w, h))
    }

    // ---------------------------------------------------------------- SD runner command
    @Test fun runnerScriptNeverContainsUserText() {
        val evil = "\$(reboot) `id` \"; rm -rf /; it's"
        val s = SdRunnerCommand.script("./sd_qidk_runner_inpaint", "/data/local/tmp/sd_runtime",
            listOf("export LD_LIBRARY_PATH=/data/local/tmp/sd_runtime:\$LD_LIBRARY_PATH"), 20, 2.0f, 42)
        assertFalse(s.contains("reboot")); assertFalse(s.contains("rm -rf"))
        assertTrue(s, s.contains("--prompt \"\$SD_PROMPT\"") && s.contains("--neg \"\$SD_NEG\"") && s.contains("--steps 20") && s.contains("--cfg 2.00") && s.contains("--seed 42"))
        assertEquals(evil, SdRunnerCommand.environment(evil, "x")["SD_PROMPT"])
        for (bad in listOf(0, 1, 101, -5)) try { SdRunnerCommand.script("./r", "/d", emptyList(), bad, 2f, 1); fail("steps $bad") } catch (_: IllegalArgumentException) {}
        try { SdRunnerCommand.script("./r; reboot", "/d", emptyList(), 20, 2f, 1); fail() } catch (_: IllegalArgumentException) {}
    }

    // ---------------------------------------------------------------- router features
    @Test fun routerFeaturesOnSyntheticGrids() {
        val flat = ProxyGrid.build(image(256, 256) { _, _ -> rgb(128, 128, 128) }, ByteArray(256 * 256), 256, 256)
        val f = RouterFeatures.registry
        assertEquals(0f, f["laplacian_variance"]!!(flat), 1e-3f)
        assertEquals(0f, f["edge_density"]!!(flat), 1e-6f)
        assertEquals(0f, f["mask_area_ratio"]!!(flat), 0f)
        assertEquals(128f / 255f, f["mean_luma"]!!(flat), 1e-3f)
        assertEquals(0f, f["entropy_luma"]!!(flat), 1e-6f)

        val checker = ProxyGrid.build(image(512, 512) { x, y -> if ((x / 4 + y / 4) % 2 == 0) rgb(0, 0, 0) else rgb(255, 255, 255) }, ByteArray(512 * 512), 512, 512)
        assertTrue("checkerboard is sharp", f["laplacian_variance"]!!(checker) > 500f || f["edge_density"]!!(checker) > 0.2f)

        // two separate 64x64 squares in a 512 image -> area 2*(64*64)/512^2, 2 fragments, no border contact
        val m = ByteArray(512 * 512)
        for (y in 100 until 164) for (x in 100 until 164) m[y * 512 + x] = 1
        for (y in 300 until 364) for (x in 350 until 414) m[y * 512 + x] = 1
        val g = ProxyGrid.build(image(512, 512) { _, _ -> rgb(10, 200, 30) }, m, 512, 512)
        assertEquals(2f * 64 * 64 / (512f * 512f), f["mask_area_ratio"]!!(g), 0.003f)
        assertEquals(2f, f["mask_fragments"]!!(g), 0f)
        assertEquals(0f, f["mask_border_touch"]!!(g), 0f)
        assertTrue(f["mask_bbox_fill"]!!(g) < 0.5f)
        for (y in 0 until 8) for (x in 0 until 8) m[y * 512 + x] = 1
        assertEquals(1f, f["mask_border_touch"]!!(ProxyGrid.build(image(512, 512) { _, _ -> 0 }, m, 512, 512)), 0f)
    }

    // ---------------------------------------------------------------- router policy
    private val routerJson = """{"alpha":0.0,"feature_names":["mean_luma","mask_area_ratio","temp_c","cpu_busy"],
        "arms":[{"name":"max_quality","theta":[1,0,0,0]},{"name":"balanced_pareto","theta":[0,1,0,0]},{"name":"thermal_contention_safe","theta":[0,0,0.02,0]}]}"""

    @Test fun policyFollowsTheTelemetry() {
        val model = LinUcbModel.parse(routerJson)
        var tel = Telemetry(40f, 80f, 0.1f, 0.0f, "", 0L)
        val policy = RouterPolicy(model, { tel })
        assertTrue(policy.problems.toString(), policy.usable)
        assertEquals(listOf(RouterMode.MAX_QUALITY, RouterMode.BALANCED_PARETO, RouterMode.THERMAL_SAFE), policy.modes)
        val bright = ProxyGrid.build(image(256, 256) { _, _ -> rgb(250, 250, 250) }, ByteArray(256 * 256), 256, 256)
        assertEquals("cool + bright -> quality", RouterMode.MAX_QUALITY, policy.decide(bright)!!.mode)
        // hot silicon (80 C * 0.02 = 1.6 beats luma 0.98) -> thermal safe
        tel = Telemetry(80f, 80f, 0.1f, 0.0f, "", 0L)
        assertEquals(RouterMode.THERMAL_SAFE, policy.decide(bright)!!.mode)
        // cool again, big hole -> balanced (area ratio 0.5 beats 20 C * 0.02 = 0.4 and luma 0.0 on a black image)
        tel = Telemetry(20f, 80f, 0.1f, 0.0f, "", 0L)
        val black = ProxyGrid.build(image(256, 256) { _, _ -> rgb(0, 0, 0) }, ByteArray(256 * 256) { if (it / 256 < 128) 1 else 0 }, 256, 256)
        assertEquals(RouterMode.BALANCED_PARETO, policy.decide(black)!!.mode)
        repeat(300) { policy.decide(black) }                                   // JIT warm-up
        val us = (1..51).map { policy.decide(black)!!.micros }.sorted()[25]
        println("router decision (features + telemetry + LinUCB), median after warm-up: $us us")
        assertTrue("decision took $us us", us < 1000)
        val d = policy.decide(black)!!
        assertTrue("feature ${d.featureMicros} us + policy ${d.policyMicros} us must add up to <= total ${d.micros} us", d.featureMicros + d.policyMicros <= d.micros + 1)
        assertTrue("LinUCB scoring alone must stay under 100 us, was ${d.policyMicros}", d.policyMicros < 100)
    }

    @Test fun policyRefusesWhatItCannotComputeInsteadOfGuessing() {
        val model = LinUcbModel.parse("""{"feature_names":["mean_luma","some_unknown_feature"],
            "arms":[{"name":"max_quality","theta":[1,1]},{"name":"balanced_pareto","theta":[0,1]},{"name":"weird","theta":[0,2]}]}""")
        val p = RouterPolicy(model, { Telemetry.EMPTY })
        assertFalse(p.usable)
        assertTrue(p.problems.joinToString(), p.problems.any { "some_unknown_feature" in it })
        assertTrue(p.problems.joinToString(), p.problems.any { "weird" in it })
        assertNull(p.decide(ProxyGrid.build(image(128, 128) { _, _ -> 0 }, ByteArray(128 * 128), 128, 128)))
    }

    // ---------------------------------------------------------------- telemetry parsing + sampling
    @Test fun telemetryParsers() {
        assertEquals(45.23f, TelemetryPoller.parseMilliC("45230\n")!!, 1e-3f)
        assertEquals(36.5f, TelemetryPoller.parseMilliC("36.5")!!, 1e-3f)
        assertNull(TelemetryPoller.parseMilliC("garbage")); assertNull(TelemetryPoller.parseMilliC("999999999"))
        val (total, idle) = TelemetryPoller.parseProcStatLine("cpu  100 0 50 800 40 5 5 0 0 0")!!
        assertEquals(1000L, total); assertEquals(840L, idle)
        assertNull(TelemetryPoller.parseProcStatLine("cpu0 1 2 3 4"))
        assertEquals(0.37f, TelemetryPoller.parseGpuPercent("37 %")!!, 1e-6f)
        assertEquals(0.25f, TelemetryPoller.parseGpuBusyPair("120 480")!!, 1e-6f)
        assertNull(TelemetryPoller.parseGpuBusyPair("0 0"))
    }

    @Test fun telemetrySamplesFromSysfsLikeFiles() {
        val root = java.nio.file.Files.createTempDirectory("qidk-thermal").toFile()
        fun zone(i: Int, type: String, milli: Int) {
            File(root, "thermal_zone$i").apply { mkdirs(); File(this, "type").writeText("$type\n"); File(this, "temp").writeText("$milli\n") }
        }
        zone(0, "cpuss-0", 62000); zone(1, "gpuss-0", 58000); zone(7, "nsphvx-0", 51500)
        val stat = File(root, "stat").apply { writeText("cpu  100 0 50 800 40 5 5 0 0 0\n") }
        val p = TelemetryPoller({ 77f }, thermalRoot = root, procStat = stat, kgslDir = File(root, "none"))
        val s1 = p.sample()
        assertEquals(51.5f, s1.siliconTempC, 1e-3f)          // NPU-side zone preferred over the hotter CPU zone
        assertEquals(77f, s1.batteryPct, 0f)
        assertTrue(s1.notes, s1.notes.contains("temp:nsphvx-0") && s1.notes.contains("gpu:unreadable"))
        stat.writeText("cpu  200 0 100 900 40 5 5 0 0 0\n")   // +150 busy, +100 idle -> 60 % busy
        assertEquals(0.6f, p.sample().cpuBusy, 1e-3f)
        root.deleteRecursively()
    }
}
