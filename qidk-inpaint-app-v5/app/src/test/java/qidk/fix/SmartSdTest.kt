package qidk.fix

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SmartSdTest {
    private val n = 512
    private val skin = 0xFF000000.toInt() or 0xC8A080

    private fun scene(faces: List<FaceInfo>, rgb: Int = skin) =
        SmartSd.analyseScene(IntArray(n * n) { rgb }, n, n, faces)

    private fun rect(x0: Int, y0: Int, x1: Int, y1: Int): ByteArray {
        val m = ByteArray(n * n)
        for (y in y0..y1) for (x in x0..x1) m[y * n + x] = 1
        return m
    }

    private val face = listOf(FaceInfo(256f, 200f, 100f))

    @Test fun sunglassesOverTheEyesLeadWithTheUsersIntent() {
        val sc = scene(face)
        val tc = SmartSd.analyseTarget(sc, rect(150, 160, 300, 260), n, n, null)
        assertEquals(TargetKind.EYEWEAR, tc.kind)
        val p = SmartSd.plan("sunglasses", "", sc, tc)
        assertEquals(IntentKind.WEARABLE, p.intent)
        assertTrue(p.positive, p.positive.startsWith("sunglasses, a person wearing sunglasses"))
        assertTrue(p.positive, p.positive.contains("worn over the eyes"))
        assertTrue(p.positive, p.positive.contains("natural skin texture"))
        // the words that describe eyeglass frames must not be in the negative prompt any more
        for (bad in listOf("frame", "border", "duplicate")) assertFalse("negative contains '$bad': ${p.negative}", p.negative.contains(bad))
        assertTrue(p.negative, p.negative.contains("no sunglasses"))
        assertTrue(p.negative, p.negative.contains("bare eyes"))
        assertTrue("CFG ${p.cfg}", p.cfg >= 6.5f)
        assertEquals("sunglasses", p.stage3)
        assertTrue(p.stage1, p.stage1.startsWith("Scene: portrait"))
        assertTrue(p.stage2, p.stage2.contains("eyewear"))
        assertTrue(p.stage2, p.stage2.contains("face 1"))
        assertTrue(SmartSd.estimateTokens(p.positive) <= SmartSd.TOKEN_BUDGET)
    }

    @Test fun emptyPromptOnAFaceInfillsSkinAndInventsNoObject() {
        val sc = scene(face)
        val tc = SmartSd.analyseTarget(sc, rect(150, 160, 300, 260), n, n, null)
        val p = SmartSd.plan("", "", sc, tc)
        assertEquals(IntentKind.EMPTY, p.intent)
        assertEquals("[Empty]", p.stage3)
        assertTrue(p.positive, p.positive.contains("skin"))
        assertFalse(p.positive.contains("sunglasses"))
        assertTrue(p.cfg in 3f..4f)
    }

    @Test fun conflictingNegativeWordsAreDroppedButTheRestIsKept() {
        val sc = scene(emptyList(), 0xFF204080.toInt())
        val tc = SmartSd.analyseTarget(sc, rect(200, 200, 260, 260), n, n, null)
        val p = SmartSd.plan("picture frame", "frame, blur", sc, tc)
        val items = p.negative.split(", ")
        assertFalse(items.contains("frame"))
        assertTrue(items.contains("blur"))
        val q = SmartSd.plan("a drawing of a cat", "", sc, tc)
        assertTrue(q.negative.split(", ").contains("painting"))   // not protected: only "drawing" is in the intent
        assertFalse(q.negative.split(", ").contains("drawing"))
    }

    @Test fun longExpertPromptStaysVerbatimAndFirst() {
        val sc = scene(emptyList())
        val tc = SmartSd.analyseTarget(sc, rect(100, 100, 200, 200), n, n, null)
        val prompt = "a weathered wooden fishing boat with peeling blue paint moored at a stone quay in the early morning mist"
        val p = SmartSd.plan(prompt, "", sc, tc)
        assertEquals(IntentKind.DESCRIPTIVE, p.intent)
        assertTrue(p.positive.startsWith(prompt))
        assertEquals(5.0f, p.cfg, 0f)
        assertEquals(prompt, p.stage3)
    }

    @Test fun objectOnTheBackgroundKeepsTheIntentFirstAndMatchesTheLight() {
        val sc = scene(emptyList(), 0xFF6A9BD8.toInt())
        val tc = SmartSd.analyseTarget(sc, rect(60, 40, 140, 120), n, n, null)
        assertEquals(TargetKind.BACKGROUND, tc.kind)
        assertTrue(tc.describe(), tc.describe().contains("upper left"))
        val p = SmartSd.plan("a red balloon", "", sc, tc)
        assertEquals(IntentKind.OBJECT, p.intent)
        assertTrue(p.positive.startsWith("a red balloon, a red balloon clearly visible in the scene"))
        assertTrue(p.cfg >= 6f)
    }

    @Test fun standardSdIsVanilla() {
        val p = SmartSd.standardPlan("  a red car ", "", 12)
        assertFalse(p.smart)
        assertEquals("a red car", p.positive)                    // trimmed, nothing added
        assertEquals(SmartSd.STANDARD_DEFAULT_NEGATIVE, p.negative)
        assertEquals(2.0f, p.cfg, 0f)
        assertEquals("", p.stage1); assertEquals("", p.stage2)
        val q = SmartSd.standardPlan("x", "my negative", 20)
        assertEquals("my negative", q.negative)
        assertEquals(20, q.steps)
    }

    @Test fun userTextIsNeverCutByTheTokenBudget() {
        val sc = scene(face)
        val tc = SmartSd.analyseTarget(sc, rect(150, 160, 300, 260), n, n, null)
        for (prompt in listOf("sunglasses", "round golden sunglasses", "beard", "a tiny hat", "red lipstick")) {
            val p = SmartSd.plan(prompt, "", sc, tc)
            assertTrue(p.positive.startsWith(prompt))
            assertTrue("${p.positive} = ${SmartSd.estimateTokens(p.positive)}", SmartSd.estimateTokens(p.positive) <= SmartSd.TOKEN_BUDGET)
        }
        val long = (1..80).joinToString(" ") { "word$it" }
        assertEquals(long, SmartSd.plan(long, "", sc, tc).positive)   // exceeds the budget alone: nothing is appended, nothing is cut
    }

    @Test fun featureOnAFaceAndWearableWithoutAFace() {
        val sc = scene(face)
        val tc = SmartSd.analyseTarget(sc, rect(180, 330, 330, 380), n, n, null)   // mouth / chin area
        assertEquals(TargetKind.MOUTH, tc.kind)
        val beard = SmartSd.plan("beard", "", sc, tc)
        assertEquals(IntentKind.FEATURE, beard.intent)
        assertTrue(beard.positive.contains("a person with beard"))
        val noFace = scene(emptyList())
        val tc2 = SmartSd.analyseTarget(noFace, rect(100, 100, 180, 160), n, n, null)
        val hat = SmartSd.plan("hat", "", noFace, tc2)
        assertEquals(IntentKind.WEARABLE, hat.intent)
        assertTrue(hat.positive.startsWith("hat, a person wearing hat"))
        assertTrue(hat.negative.contains("bare head"))
    }

    @Test fun sceneStagesDescribeLightingAndPlace() {
        val img = IntArray(n * n) { i -> if (i % n < n / 2) 0xFF202020.toInt() else 0xFFE0E0E0.toInt() }
        val sc = SmartSd.analyseScene(img, n, n, emptyList())
        assertTrue(sc.lighting, sc.lighting.contains("from the right"))
        assertTrue(sc.describe().startsWith("Scene: "))
    }

    @Test fun tokenEstimateCountsPunctuationAndDigits() {
        assertEquals(1, SmartSd.estimateTokens("cat"))
        assertEquals(3, SmartSd.estimateTokens("cat, dog"))       // cat , dog
        assertEquals(5, SmartSd.estimateTokens("f/5.6"))          // f / 5 . 6: CLIP splits digits and punctuation one by one
    }
}
