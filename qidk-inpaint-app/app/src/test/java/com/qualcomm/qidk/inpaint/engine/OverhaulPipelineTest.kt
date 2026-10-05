package com.qualcomm.qidk.inpaint.engine

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

class OverhaulPipelineTest {

    @Test
    fun testDistanceTransformAvoidsCentroidCollapseOnUShape() {
        val w = 100
        val h = 100
        val mask = BooleanArray(w * h)

        // Create a concave U-shaped stroke:
        // Left pillar: x in 20..35, y in 20..80
        // Right pillar: x in 65..80, y in 20..80
        // Connecting bottom bar: x in 20..80, y in 65..80
        // The center void (x in 36..64, y in 20..64) is BACKGROUND!
        var sumX = 0L
        var sumY = 0L
        var count = 0

        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                val inLeft = (x in 20..35) && (y in 20..80)
                val inRight = (x in 65..80) && (y in 20..80)
                val inBottom = (x in 20..80) && (y in 65..80)
                if (inLeft || inRight || inBottom) {
                    mask[row + x] = true
                    sumX += x
                    sumY += y
                    count++
                }
            }
        }

        // Demonstrate legacy failure: Arithmetic centroid falls in empty background void!
        val centroidX = (sumX / count).toInt()
        val centroidY = (sumY / count).toInt()
        val isCentroidInMask = mask[centroidY * w + centroidX]
        assertFalse("Arithmetic centroid must fall in hollow void (demonstrating centroid collapse)", isCentroidInMask)

        // Now test Overhauled Medial Axis Sampler
        val seeds = DistanceTransformSampler.sampleMedialAxisSeeds(mask, w, h, maxSeeds = 3)
        assertTrue("Sampler must return valid medial seeds", seeds.isNotEmpty())

        // Every single seed MUST be inside the foreground
        for (seed in seeds) {
            val inMask = mask[seed.y * w + seed.x]
            assertTrue("Seed (${seed.x}, ${seed.y}) must lie strictly inside foreground mask", inMask)
            assertTrue("Seed depth must be substantial (>2px from boundary)", seed.dist >= 2.0f)
        }

        // Verify that seeds span both pillars (left pillar x < 40, right pillar x > 60)
        val hasLeftSeed = seeds.any { it.x in 20..35 }
        val hasRightSeed = seeds.any { it.x in 65..80 }
        assertTrue("Medial axis seeds must cover left lobe of U-shape", hasLeftSeed)
        assertTrue("Medial axis seeds must cover right lobe of U-shape", hasRightSeed)
    }

    @Test
    fun testFastGuidedFilterSnapsToPhysicalEdges() {
        val w = 60
        val h = 60
        val I = FloatArray(w * h)
        val p = FloatArray(w * h)

        // Guidance image I has a sharp vertical edge at x = 30:
        // Left side (x < 30) is dark (0.1), Right side (x >= 30) is bright (0.9)
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                I[row + x] = if (x < 30) 0.1f else 0.9f
            }
        }

        // Input neural mask p has a 1-pixel misalignment bleeding into background:
        // Neural mask transitions at x = 29, while true image edge is at x = 30.
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                p[row + x] = if (x < 29) 0.05f else 0.95f
            }
        }

        // Execute Guided Filter (r=3, eps=0.01f)
        val q = FastGuidedFilter.filter(I, p, w, h, r = 3, eps = 0.01f)

        // At x = 29: p was falsely 0.95 in the dark background, but guided filter
        // pulls it down below the threshold (< 0.45) matching true edge at x=30!
        val rowCenter = 30 * w
        assertTrue(
            "Guided filter must pull misaligned edge pixel at x=29 below 0.45 (was ${q[rowCenter + 29]})",
            q[rowCenter + 29] < 0.45f
        )

        // At x = 33 (safely inside bright foreground):
        assertTrue(
            "Guided filter must maintain high probability inside true object (was ${q[rowCenter + 33]})",
            q[rowCenter + 33] > 0.70f
        )
    }

    @Test
    fun testCircularSafetyDilation() {
        val w = 30
        val h = 30
        val mask = BooleanArray(w * h)

        // Single isolated foreground point at (15, 15)
        mask[15 * w + 15] = true

        val dilated = FastGuidedFilter.dilateBinary(mask, w, h, radius = 2)

        var count = 0
        for (y in 0 until h) {
            val row = y * w
            for (x in 0 until w) {
                if (dilated[row + x]) {
                    count++
                    val dx = x - 15
                    val dy = y - 15
                    assertTrue("Dilated point ($x, $y) must be within circular radius 2", dx * dx + dy * dy <= 4)
                }
            }
        }

        // Circular disk of radius 2 on discrete grid:
        // dy = 0: dx in -2..2 (5 pts)
        // dy = ±1: dx in -1..1 (3 pts each -> 6 pts)
        // dy = ±2: dx = 0 (1 pt each -> 2 pts)
        // Total = 5 + 6 + 2 = 13 pixels
        assertEquals("Circular dilation with r=2 on single pixel must produce 13 pixels", 13, count)
    }

    @Test
    fun testBoxFilterExecutionSpeed() {
        val w = 512
        val h = 512
        val src = FloatArray(w * h) { 0.5f }
        val dst = FloatArray(w * h)

        val t0 = System.nanoTime()
        FastGuidedFilter.boxFilter(src, w, h, 4, dst)
        val elapsedMs = (System.nanoTime() - t0) / 1_000_000.0

        println("512x512 separable boxFilter took ${elapsedMs}ms on JVM")
        assertTrue("512x512 boxFilter must execute in <20ms on JVM", elapsedMs < 20.0)
        assertEquals(0.5f, dst[256 * w + 256], 0.001f)
    }
}
