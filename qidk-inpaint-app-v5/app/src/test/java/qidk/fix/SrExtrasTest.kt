package qidk.fix

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class SrExtrasTest {

    @Test fun runnerScriptCarriesOnlyPlainExtraSwitches() {
        val s = SdRunnerCommand.script("./sd_qidk_runner_inpaint", "/data/local/tmp/sd_runtime", emptyList(), 12, 4.5f, 7,
            extra = listOf("--strength", "0.30", "--no-composite"))
        assertTrue(s, s.contains("--seed 7 --strength 0.30 --no-composite;"))
        assertTrue(s.contains("--cfg 4.50"))
        for (bad in listOf("; reboot", "$(id)", "a b", "`x`", "")) {
            try { SdRunnerCommand.script("./r", "/d", emptyList(), 12, 2f, 1, extra = listOf(bad)); fail("accepted '$bad'") } catch (_: IllegalArgumentException) {}
        }
        // the user's prompt still never reaches the script text
        val evil = "a'; rm -rf /; echo '"
        assertFalse(SdRunnerCommand.script("./r", "/d", emptyList(), 12, 2f, 1).contains("rm -rf"))
        assertEquals(evil, SdRunnerCommand.environment(evil, "n")["SD_PROMPT"])
    }

    @Test fun mipLevelKeepsTheSourceCloseToOneToOne() {
        assertEquals(0, MipPyramid.levelFor(1.0f, 5))
        assertEquals(0, MipPyramid.levelFor(3.0f, 5))            // magnified: level 0, real pixels
        assertEquals(1, MipPyramid.levelFor(0.5f, 5))
        assertEquals(3, MipPyramid.levelFor(0.09f, 5))           // 4096 px output in a 330 px pane
        assertEquals(2, MipPyramid.levelFor(0.09f, 3))           // never beyond the last level
        assertEquals(0, MipPyramid.levelFor(0.09f, 1))
    }

    @Test fun smartPromptNegativeNoLongerHoldsTheEyeglassWords() {
        val n = SmartPrompt.negative("")
        for (bad in listOf("frame", "border", "duplicate")) assertFalse("negative still contains $bad: $n", n.contains(bad))
        assertTrue(n.startsWith("painting"))
    }
}
