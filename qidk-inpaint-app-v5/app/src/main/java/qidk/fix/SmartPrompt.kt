package qidk.fix

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * "Smart SD" stage 1: a deterministic, millisecond-cheap vision pass over the UNMASKED ring around the hole that turns
 * what the photo already shows (surface, texture, light, colour temperature) into a photographic prompt. Stage 2 is the
 * normal SD run with that prompt. "Normal SD" skips this and sends the user's text unchanged.
 */
object SmartPrompt {
    class Descriptor(
        val surface: String, val texture: String, val light: String, val tone: String, val lens: String,
        val ringPx: Int, val meanR: Float, val meanG: Float, val meanB: Float, val luma: Float, val grad: Float, val cyNorm: Float
    ) {
        override fun toString() = "surface='$surface' texture='$texture' light='$light' tone='$tone' lens='$lens' " +
            "ring=$ringPx px rgb=(%.0f,%.0f,%.0f) luma=%.0f grad=%.1f cy=%.2f".format(meanR, meanG, meanB, luma, grad, cyNorm)
    }

    const val RING_PX = 24
    const val MAX_WORDS = 55            // CLIP reads 75 tokens; stay well below it

    /** Distance (chamfer 3-4) from every pixel to the nearest hole pixel, capped at [cap]. */
    private fun distanceToHole(hole: ByteArray, w: Int, h: Int, cap: Int): IntArray {
        val inf = cap * 3 + 3
        val d = IntArray(w * h) { if (hole[it].toInt() != 0) 0 else inf }
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            var v = d[i]
            if (x > 0) v = min(v, d[i - 1] + 3)
            if (y > 0) {
                v = min(v, d[i - w] + 3)
                if (x > 0) v = min(v, d[i - w - 1] + 4)
                if (x < w - 1) v = min(v, d[i - w + 1] + 4)
            }
            d[i] = v
        }
        for (y in h - 1 downTo 0) for (x in w - 1 downTo 0) {
            val i = y * w + x
            var v = d[i]
            if (x < w - 1) v = min(v, d[i + 1] + 3)
            if (y < h - 1) {
                v = min(v, d[i + w] + 3)
                if (x < w - 1) v = min(v, d[i + w + 1] + 4)
                if (x > 0) v = min(v, d[i + w - 1] + 4)
            }
            d[i] = v
        }
        return d
    }

    /** [argb]: w*h Android ARGB ints. [hole]: w*h bytes, non-zero = masked. Returns null when there is no context at all. */
    fun analyse(argb: IntArray, hole: ByteArray, w: Int, h: Int, ringPx: Int = RING_PX): Descriptor? {
        val dist = distanceToHole(hole, w, h, ringPx)
        val limit = ringPx * 3
        var n = 0
        var sr = 0.0; var sg = 0.0; var sb = 0.0; var sl = 0.0; var sl2 = 0.0
        var hx = 0.0; var hy = 0.0; var hc = 0
        for (i in 0 until w * h) if (hole[i].toInt() != 0) { hx += i % w; hy += i / w; hc++ }
        if (hc == 0) return null
        val cx = (hx / hc).toFloat(); val cy = (hy / hc).toFloat()
        var lLeft = 0.0; var nLeft = 0; var lRight = 0.0; var nRight = 0; var lTop = 0.0; var nTop = 0; var lBot = 0.0; var nBot = 0
        var gradSum = 0.0; var gradN = 0
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            if (hole[i].toInt() != 0 || dist[i] > limit) continue
            val c = argb[i]
            val r = ((c shr 16) and 0xFF).toDouble(); val g = ((c shr 8) and 0xFF).toDouble(); val b = (c and 0xFF).toDouble()
            val l = 0.299 * r + 0.587 * g + 0.114 * b
            n++; sr += r; sg += g; sb += b; sl += l; sl2 += l * l
            if (x < cx) { lLeft += l; nLeft++ } else { lRight += l; nRight++ }
            if (y < cy) { lTop += l; nTop++ } else { lBot += l; nBot++ }
            if (hole[i - 1].toInt() == 0 && hole[i + 1].toInt() == 0 && hole[i - w].toInt() == 0 && hole[i + w].toInt() == 0) {
                fun lum(p: Int) = 0.299 * ((argb[p] shr 16) and 0xFF) + 0.587 * ((argb[p] shr 8) and 0xFF) + 0.114 * (argb[p] and 0xFF)
                gradSum += abs(lum(i + 1) - lum(i - 1)) * 0.5 + abs(lum(i + w) - lum(i - w)) * 0.5; gradN++
            }
        }
        if (n < 50) return null
        val mr = (sr / n).toFloat(); val mg = (sg / n).toFloat(); val mb = (sb / n).toFloat()
        val luma = (sl / n).toFloat()
        val grad = if (gradN > 0) (gradSum / gradN).toFloat() else 0f
        val cyN = cy / h

        // HSV of the mean colour
        val mx = max(mr, max(mg, mb)); val mn = min(mr, min(mg, mb))
        val v = mx / 255f
        val s = if (mx <= 0f) 0f else (mx - mn) / mx
        val hue = if (mx == mn) 0f else when (mx) {
            mr -> (60f * ((mg - mb) / (mx - mn)) + 360f) % 360f
            mg -> 60f * ((mb - mr) / (mx - mn)) + 120f
            else -> 60f * ((mr - mg) / (mx - mn)) + 240f
        }

        val surface = when {
            s < 0.12f -> when {
                v > 0.75f -> if (cyN < 0.5f) "bright overcast sky and soft haze" else "pale neutral surface"
                v < 0.25f -> "deep shadow, dark surface"
                else -> "neutral grey surface"
            }
            hue in 170f..260f -> if (cyN < 0.4f) "clear blue sky" else "calm blue water surface with gentle ripples"
            hue in 70f..170f -> "lush green foliage and grass"
            hue in 15f..70f -> if (s < 0.45f) "sandy dry ground" else "warm earthy surface"
            else -> "warm toned surface"
        }
        val texture = when {
            grad < 2.5f -> "smooth, seamless, no objects"
            grad < 8f -> "fine natural texture"
            else -> "rich detailed texture"
        }
        val dirLR = if (nLeft > 0 && nRight > 0) lRight / nRight - lLeft / nLeft else 0.0
        val dirTB = if (nTop > 0 && nBot > 0) lBot / nBot - lTop / nTop else 0.0
        val dir = when {
            abs(dirLR) < 6 && abs(dirTB) < 6 -> "even lighting"
            abs(dirLR) >= abs(dirTB) -> if (dirLR > 0) "light coming from the right" else "light coming from the left"
            else -> if (dirTB > 0) "light coming from below" else "light coming from above"
        }
        val light = when {
            luma > 170 -> "bright natural daylight, $dir"
            luma > 90 -> "soft diffused light, $dir"
            else -> "dim low-key light, $dir"
        }
        val tone = when {
            mr - mb > 15f -> "warm golden tones"
            mb - mr > 15f -> "cool blue tones"
            else -> "neutral colour balance"
        }
        val lens = if (grad < 2.5f) "shot at f/8, deep depth of field" else "35mm lens, f/5.6, crisp detail"
        return Descriptor(surface, texture, light, tone, lens, ringPx, mr, mg, mb, luma, grad, cyN)
    }

    /** Stage-1 output: the final positive prompt. The user's own text always comes first. */
    fun build(userPrompt: String, d: Descriptor): String {
        val user = userPrompt.trim().trimEnd(',', '.', ' ')
        val parts = ArrayList<String>()
        parts.add(if (user.isEmpty()) "a photograph of ${d.surface}" else user)
        if (user.isNotEmpty()) parts.add(d.surface)
        parts.addAll(listOf(d.texture, d.light, d.tone, "photorealistic photograph", d.lens, "seamless continuation of the surrounding scene"))
        var text = parts.joinToString(", ")
        val words = text.split(Regex("\\s+"))
        if (words.size > MAX_WORDS) text = words.take(MAX_WORDS).joinToString(" ").trimEnd(',')
        return text
    }

    private const val BASE_NEGATIVE = "painting, drawing, illustration, cartoon, glitch, deformed, blurry, low quality, " +
        "artifacts, text, watermark, frame, border, duplicate, oversaturated"

    fun negative(userNegative: String): String {
        val u = userNegative.trim().trimEnd(',', ' ')
        return if (u.isEmpty()) BASE_NEGATIVE else "$u, $BASE_NEGATIVE"
    }
}

/**
 * Builds the shell script that starts the SD runner and the environment that carries the user's text. The text is
 * NEVER spliced into the script: it travels in environment variables and the script only mentions "$SD_PROMPT" /
 * "$SD_NEG", which sh expands as single words without re-parsing (no quoting bugs, no command injection).
 */
object SdRunnerCommand {
    const val MIN_STEPS = 2
    const val MAX_STEPS = 100

    fun environment(prompt: String, negative: String) = mapOf("SD_PROMPT" to prompt, "SD_NEG" to negative)

    /** [runnerBin], [sdDir] and [exports] are fixed constants of the app, never user text. */
    fun script(runnerBin: String, sdDir: String, exports: List<String>, steps: Int, cfg: Float, seed: Int): String {
        require(steps in MIN_STEPS..MAX_STEPS) { "steps $steps outside $MIN_STEPS..$MAX_STEPS" }
        require(cfg in 1f..20f) { "cfg $cfg outside 1..20" }
        require(runnerBin.isNotEmpty() && runnerBin.all { it.isLetterOrDigit() || it in "/_.-" }) { "runner path must be a plain token" }
        require(sdDir.isNotEmpty() && sdDir.all { it.isLetterOrDigit() || it in "/_.-" }) { "sd dir must be a plain path" }
        val pre = (listOf("cd $sdDir || exit 97") + exports + listOf("rm -f sd_output.png 2>/dev/null")).joinToString("; ")
        val cfgText = String.format(java.util.Locale.US, "%.2f", cfg)
        return "$pre; $runnerBin --prompt \"\$SD_PROMPT\" --neg \"\$SD_NEG\" --steps $steps --cfg $cfgText --seed $seed; " +
            "RET=\$?; chmod 666 sd_output.png image.raw mask.raw 2>/dev/null; exit \$RET"
    }
}
