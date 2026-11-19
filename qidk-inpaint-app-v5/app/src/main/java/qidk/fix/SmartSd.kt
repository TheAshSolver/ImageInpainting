package qidk.fix

import java.util.Locale
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/** A face found by android.media.FaceDetector, in the pixel units of the analysed (512 x 512) image. */
data class FaceInfo(val midX: Float, val midY: Float, val eyeDist: Float, val rollDeg: Float = 0f)

enum class SceneClass(val label: String) {
    PORTRAIT("portrait"), SKY("open sky / landscape"), WATER("water scene"), FOLIAGE("foliage / nature"), GENERAL("general scene")
}

enum class TargetKind(val label: String) {
    EYEWEAR("eyewear region (both eyes and the nose bridge)"),
    EYES("eye region"),
    FOREHEAD("forehead / hairline"),
    MOUTH("mouth and chin"),
    CHEEK("cheek"),
    FACE_OTHER("face"),
    BACKGROUND("background / scene region"),
}

enum class IntentKind { EMPTY, WEARABLE, FEATURE, OBJECT, DESCRIPTIVE, REPLACEMENT }

/** Stage 1: what the WHOLE picture shows. */
class SceneContext(
    val width: Int, val height: Int,
    val meanR: Float, val meanG: Float, val meanB: Float,
    val luma: Float, val contrast: Float, val warmth: Float, val edgeDensity: Float,
    val faces: List<FaceInfo>, val sceneClass: SceneClass,
    val lighting: String, val tone: String, val environment: String
) {
    fun describe(): String {
        val faceText = when (faces.size) {
            0 -> sceneClass.label
            1 -> "portrait, 1 face (eye distance %.0f px)".format(Locale.US, faces[0].eyeDist)
            else -> "portrait, ${faces.size} faces"
        }
        return "Scene: $faceText | Lighting: $lighting (luma %.0f, contrast %.0f) | Colour: $tone (R-B %+.0f) | Environment: $environment (edge density %.1f%%)"
            .format(Locale.US, luma, contrast, warmth, edgeDensity * 100f)
    }
}

/** Stage 2: what the mask is about to replace. */
class TargetContext(
    val holePx: Int, val holePct: Float, val bbox: IntArray,
    val kind: TargetKind, val faceIndex: Int, val extra: String, val location: String, val boundary: String,
    val contactSurface: String = "matching surface",
    val interiorDesc: String = "",
    val aspectRatio: Float = 1.0f
) {
    val onFace: Boolean get() = kind != TargetKind.BACKGROUND
    fun describe(): String {
        val what = if (onFace) "${kind.label} of face ${faceIndex + 1}$extra" else "${kind.label} at the $location"
        val surf = if (contactSurface.isNotEmpty()) " | Ground Contact: $contactSurface" else ""
        val intText = if (interiorDesc.isNotEmpty()) " | Pre-inpaint Void: $interiorDesc (AR %.2f)".format(Locale.US, aspectRatio) else ""
        return "Void: %,d px (%.1f%% of the image), box x %d-%d y %d-%d | Target: %s | Boundary: %s%s%s"
            .format(Locale.US, holePx, holePct, bbox[0], bbox[2], bbox[1], bbox[3], what, boundary, surf, intText)
    }
}

/**
 * The single, final description of one Stable Diffusion run. The Smart-SD drawer shows exactly these strings and the
 * dispatcher sends exactly these strings (there is no second composition anywhere), so what you read is what the NPU gets.
 */
class SmartSdPlan(
    val smart: Boolean,
    val stage1: String,
    val stage2: String,
    val stage3: String,
    val positive: String,
    val negative: String,
    val cfg: Float,
    val steps: Int,
    val seed: Int,
    val intent: IntentKind
) {
    fun runParams(): String = String.format(Locale.US, "Sampler: DPM++ %d steps | CFG %.1f | seed %d | intent %s", steps, cfg, seed, intent.name.lowercase(Locale.US))
}

/**
 * Smart SD: turns (whole image, masked target, the user's words) into one prompt in which the USER'S INTENT leads and the
 * scene only decorates it. Standard SD never calls into this object.
 *
 * What went wrong before (root cause, see SUPER_STUDIO_REPAIR_GUIDE.md section 3): the negative prompt carried "frame,
 * border, duplicate", which are exactly the words that describe eyeglass frames; the guidance was fixed at 2.0, too low
 * for an object the model has to invent; the prompt said "a photograph OF sunglasses" (a product shot) instead of a
 * person WEARING them; and the "context" was a colour statistic that read skin as "sandy dry ground".
 */
object SmartSd {
    const val DEFAULT_NEGATIVE_BASE = "painting, drawing, illustration, cartoon, glitch, deformed, blurry, low quality, artifacts, text, watermark, oversaturated"
    private const val FACE_NEGATIVE = "cross-eyed, ugly, disfigured, mutated"
    const val STANDARD_DEFAULT_NEGATIVE = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
    const val STANDARD_CFG = 2.0f
    const val TOKEN_BUDGET = 68            // CLIP reads 77 positions including BOS / EOS

    private val glassesWords = setOf("sunglasses", "sunglass", "shades", "glasses", "eyeglasses", "spectacles", "goggles", "monocle", "eyewear")
    private val headWords = setOf("hat", "cap", "beanie", "helmet", "crown", "tiara", "turban", "hood", "headband", "bonnet", "fedora")
    private val coverWords = setOf("mask", "bandana", "scarf", "veil", "respirator")
    private val earWords = setOf("earring", "earrings", "headphones", "earphones", "earbuds")
    private val featureWords = setOf(
        "beard", "moustache", "mustache", "stubble", "goatee", "freckles", "scar", "tattoo", "wrinkles", "makeup", "lipstick",
        "eyeshadow", "eyebrows", "mole", "smile", "teeth", "hair", "eyes", "lips", "nose", "blush"
    )

    // ----------------------------------------------------------------------------------------------- Stage 1

    /** [argb] w*h Android ARGB ints (the 512 x 512 working copy). */
    fun analyseScene(argb: IntArray, w: Int, h: Int, faces: List<FaceInfo>): SceneContext {
        var sr = 0.0; var sg = 0.0; var sb = 0.0; var sl = 0.0; var sl2 = 0.0
        var lLeft = 0.0; var nLeft = 0; var lRight = 0.0; var nRight = 0; var lTop = 0.0; var nTop = 0; var lBot = 0.0; var nBot = 0
        // thirds for the scene guess: mean (r,g,b) of the top and bottom thirds
        val top = DoubleArray(3); val bot = DoubleArray(3); var nT = 0; var nB = 0
        var edges = 0L; var edgeN = 0L
        val lum = FloatArray(w * h)
        for (i in 0 until w * h) {
            val c = argb[i]
            val r = ((c shr 16) and 0xFF).toDouble(); val g = ((c shr 8) and 0xFF).toDouble(); val b = (c and 0xFF).toDouble()
            val l = 0.299 * r + 0.587 * g + 0.114 * b
            lum[i] = l.toFloat()
            sr += r; sg += g; sb += b; sl += l; sl2 += l * l
            val x = i % w; val y = i / w
            if (x < w / 2) { lLeft += l; nLeft++ } else { lRight += l; nRight++ }
            if (y < h / 2) { lTop += l; nTop++ } else { lBot += l; nBot++ }
            if (y < h / 3) { top[0] += r; top[1] += g; top[2] += b; nT++ }
            if (y >= 2 * h / 3) { bot[0] += r; bot[1] += g; bot[2] += b; nB++ }
        }
        for (y in 1 until h - 1) for (x in 1 until w - 1) {
            val i = y * w + x
            val gx = abs(lum[i + 1] - lum[i - 1]); val gy = abs(lum[i + w] - lum[i - w])
            if (gx + gy > 40f) edges++
            edgeN++
        }
        val n = (w * h).toDouble()
        val mr = (sr / n).toFloat(); val mg = (sg / n).toFloat(); val mb = (sb / n).toFloat()
        val luma = (sl / n).toFloat()
        val contrast = kotlin.math.sqrt(max(0.0, sl2 / n - (sl / n) * (sl / n))).toFloat()
        val warmth = mr - mb
        val edgeDensity = if (edgeN > 0) edges.toFloat() / edgeN else 0f

        val dirLR = if (nLeft > 0 && nRight > 0) lRight / nRight - lLeft / nLeft else 0.0
        val dirTB = if (nTop > 0 && nBot > 0) lBot / nBot - lTop / nTop else 0.0
        val dir = when {
            abs(dirLR) < 8 && abs(dirTB) < 8 -> "even"
            abs(dirLR) >= abs(dirTB) -> if (dirLR > 0) "from the right" else "from the left"
            else -> if (dirTB > 0) "from below" else "from above"
        }
        val lighting = when {
            luma > 170 -> if (dir == "even") "bright even daylight" else "bright daylight $dir"
            luma > 90 -> if (dir == "even") "soft diffused light" else "soft diffused light $dir"
            else -> if (dir == "even") "dim low-key light" else "dim low-key light $dir"
        }
        val tone = when {
            warmth > 15f -> "warm tones"
            warmth < -15f -> "cool blue tones"
            else -> "neutral colour balance"
        }

        val tr = if (nT > 0) top[0] / nT else 0.0; val tg = if (nT > 0) top[1] / nT else 0.0; val tb = if (nT > 0) top[2] / nT else 0.0
        val br = if (nB > 0) bot[0] / nB else 0.0; val bg = if (nB > 0) bot[1] / nB else 0.0; val bb = if (nB > 0) bot[2] / nB else 0.0
        val sceneClass = when {
            faces.isNotEmpty() -> SceneClass.PORTRAIT
            tb > tr + 15 && tb > tg && luma > 90 && br < tb -> SceneClass.SKY
            bb > br + 15 && bb > bg -> SceneClass.WATER
            mg > mr + 8 && mg > mb + 8 -> SceneClass.FOLIAGE
            else -> SceneClass.GENERAL
        }
        val environment = when (sceneClass) {
            SceneClass.PORTRAIT -> if (edgeDensity < 0.06f) "plain backdrop" else "detailed surroundings"
            SceneClass.SKY -> "open air"
            SceneClass.WATER -> "water and shoreline"
            SceneClass.FOLIAGE -> "outdoor vegetation"
            SceneClass.GENERAL -> if (edgeDensity < 0.05f) "smooth, low-detail surroundings" else "detailed surroundings"
        }
        return SceneContext(w, h, mr, mg, mb, luma, contrast, warmth, edgeDensity, faces, sceneClass, lighting, tone, environment)
    }

    // ----------------------------------------------------------------------------------------------- Stage 2

    /** [hole] w*h bytes, non-zero = masked. [ring] is the optional boundary descriptor of the older ring analysis. */
    fun analyseTarget(scene: SceneContext, hole: ByteArray, w: Int, h: Int, ring: SmartPrompt.Descriptor?, argb: IntArray? = null): TargetContext {
        var n = 0; var x0 = w; var y0 = h; var x1 = -1; var y1 = -1
        for (i in 0 until w * h) if (hole[i].toInt() != 0) {
            n++
            val x = i % w; val y = i / w
            if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y
        }
        val boundary = when {
            ring != null -> "ring of ${ring.ringPx} px, luma %.0f, %s".format(Locale.US, ring.luma, ring.tone)
            else -> "no usable ring"
        }
        if (n == 0) {
            return TargetContext(0, 0f, intArrayOf(0, 0, 0, 0), TargetKind.BACKGROUND, -1, "", "image", boundary)
        }
        val pct = n * 100f / (w * h)
        val cx = (x0 + x1) / 2f / w; val cy = (y0 + y1) / 2f / h
        val location = (if (cy < 0.34f) "upper " else if (cy > 0.66f) "lower " else "") +
            (if (cx < 0.34f) "left" else if (cx > 0.66f) "right" else "centre")
        val bbox = intArrayOf(x0, y0, x1, y1)

        // Dual-Context Extraction: (1) Pre-inpaint mask interior & (2) Ground contact horizon
        val ar = if (y1 >= y0) (x1 - x0 + 1).toFloat() / max(1, y1 - y0 + 1).toFloat() else 1.0f
        var interiorDesc = ""
        var contactSurface = if (ring != null) ring.surface else "table or floor surface"

        if (argb != null && argb.size == w * h) {
            var inR = 0.0; var inG = 0.0; var inB = 0.0; var inL = 0.0; var inCount = 0
            for (y in y0..y1) {
                for (x in x0..x1) {
                    val idx = y * w + x
                    if (idx in 0 until (w * h) && hole[idx].toInt() != 0) {
                        val c = argb[idx]
                        val r = (c shr 16) and 0xFF
                        val g = (c shr 8) and 0xFF
                        val b = c and 0xFF
                        inR += r; inG += g; inB += b
                        inL += 0.299 * r + 0.587 * g + 0.114 * b
                        inCount++
                    }
                }
            }
            if (inCount > 0) {
                val avgL = (inL / inCount).toFloat()
                interiorDesc = when {
                    ar < 0.65f -> if (avgL < 90) "dark upright object" else "slender upright object"
                    ar > 1.45f -> if (avgL < 90) "dark horizontal object" else "wide horizontal object"
                    else -> if (avgL < 90) "dark compact object" else "compact object"
                }
            }

            var botR = 0.0; var botG = 0.0; var botB = 0.0; var botL = 0.0; var botCount = 0
            val scanY0 = min(h - 1, y1 + 1)
            val scanY1 = min(h - 1, y1 + 12)
            for (sy in scanY0..scanY1) {
                for (sx in x0..x1) {
                    val idx = sy * w + sx
                    if (idx in 0 until (w * h) && hole[idx].toInt() == 0) {
                        val c = argb[idx]
                        val r = (c shr 16) and 0xFF
                        val g = (c shr 8) and 0xFF
                        val b = c and 0xFF
                        botR += r; botG += g; botB += b
                        botL += 0.299 * r + 0.587 * g + 0.114 * b
                        botCount++
                    }
                }
            }
            if (botCount > 10) {
                val mR = (botR / botCount).toFloat()
                val mG = (botG / botCount).toFloat()
                val mB = (botB / botCount).toFloat()
                val mL = (botL / botCount).toFloat()
                val warmth = mR - mB
                contactSurface = when {
                    mG > mR + 10 && mG > mB + 10 -> "natural lawn or foliage ground"
                    warmth > 18f && mL in 40f..180f -> "wooden table or tabletop surface"
                    mL > 190f -> "bright polished tabletop"
                    mL < 45f -> "dark table surface"
                    abs(warmth) < 12f -> "neutral stone tabletop"
                    else -> "tabletop surface"
                }
            }
        }

        // Face zones in face-aligned units (u, v) = rotate(p - mid, -roll) / eyeDist.
        var bestFace = -1; var bestCount = 0
        val faceZones = scene.faces.map { zoneCounts(it, hole, w, h) }
        for (i in faceZones.indices) if (faceZones[i].total > bestCount) { bestCount = faceZones[i].total; bestFace = i }
        if (bestFace < 0 || bestCount < n * 0.25) {
            return TargetContext(n, pct, bbox, TargetKind.BACKGROUND, -1, "", location, boundary, contactSurface, interiorDesc, ar)
        }
        val z = faceZones[bestFace]
        val eyeShare = z.eye.toFloat() / bestCount
        val eyeBandArea = z.eyeArea.coerceAtLeast(1)
        val kind: TargetKind
        var extra = ""
        when {
            eyeShare >= 0.2f && z.eye.toFloat() / eyeBandArea >= 0.3f -> {
                kind = TargetKind.EYEWEAR
                if (z.cheek > bestCount * 0.1f) extra = " and the adjacent cheek"
            }
            eyeShare >= 0.2f -> kind = TargetKind.EYES
            z.forehead >= max(z.mouth, z.cheek) && z.forehead > bestCount * 0.3f -> kind = TargetKind.FOREHEAD
            z.mouth >= z.cheek && z.mouth > bestCount * 0.3f -> kind = TargetKind.MOUTH
            z.cheek > bestCount * 0.3f -> kind = TargetKind.CHEEK
            else -> kind = TargetKind.FACE_OTHER
        }
        return TargetContext(n, pct, bbox, kind, bestFace, extra, location, boundary, contactSurface, interiorDesc, ar)
    }

    private class Zones(var total: Int = 0, var eye: Int = 0, var eyeArea: Int = 0, var forehead: Int = 0, var mouth: Int = 0, var cheek: Int = 0)

    private fun zoneCounts(f: FaceInfo, hole: ByteArray, w: Int, h: Int): Zones {
        val z = Zones()
        val ed = max(f.eyeDist, 1f)
        val a = Math.toRadians(-f.rollDeg.toDouble()); val ca = cos(a).toFloat(); val sa = sin(a).toFloat()
        val reach = (ed * 3f).toInt() + 2
        val xa = max(0, (f.midX - reach).toInt()); val xb = min(w - 1, (f.midX + reach).toInt())
        val ya = max(0, (f.midY - reach).toInt()); val yb = min(h - 1, (f.midY + reach).toInt())
        for (y in ya..yb) for (x in xa..xb) {
            val dx = x - f.midX; val dy = y - f.midY
            val u = (dx * ca - dy * sa) / ed; val v = (dx * sa + dy * ca) / ed
            val inEye = abs(v) <= 0.55f && abs(u) <= 1.15f
            if (inEye) z.eyeArea++
            if (hole[y * w + x].toInt() == 0) continue
            if (abs(u) > 1.8f || v < -2.0f || v > 2.6f) continue
            z.total++
            when {
                inEye -> z.eye++
                v < -0.55f -> z.forehead++
                v in 0.8f..1.7f && abs(u) <= 0.9f -> z.mouth++
                v > 0.55f && abs(u) >= 0.6f -> z.cheek++
            }
        }
        return z
    }

    // ----------------------------------------------------------------------------------------------- intent

    private fun words(text: String): List<String> =
        text.lowercase(Locale.US).split(Regex("[^a-z0-9']+")).filter { it.isNotEmpty() }

    fun classify(userPrompt: String): IntentKind {
        val ws = words(userPrompt)
        if (ws.isEmpty() || userPrompt.equals("remove", ignoreCase = true) || userPrompt.equals("clean", ignoreCase = true) || userPrompt.equals("erase", ignoreCase = true)) return IntentKind.EMPTY
        val lower = userPrompt.lowercase(Locale.US)
        if (lower.contains("replace") || lower.contains("substitute") || lower.contains("swap") || lower.contains("change to") || lower.contains("put")) return IntentKind.REPLACEMENT
        if (ws.size > 8) return IntentKind.DESCRIPTIVE
        if (ws.any { it in glassesWords || it in headWords || it in coverWords || it in earWords }) return IntentKind.WEARABLE
        if (ws.any { it in featureWords }) return IntentKind.FEATURE
        return IntentKind.OBJECT
    }

    fun extractReplacementTarget(userPrompt: String): String {
        val lower = userPrompt.trim()
        val replaceRegex = Regex("(?i)replace\\s+(?:the\\s+|a\\s+|an\\s+)?(?:.+?)\\s+with\\s+(?:an?\\s+)?(.+)", RegexOption.IGNORE_CASE)
        val matchReplace = replaceRegex.find(lower)
        if (matchReplace != null && matchReplace.groupValues.size > 1) {
            return matchReplace.groupValues[1].trim()
        }
        val subRegex = Regex("(?i)substitute\\s+(?:the\\s+|a\\s+|an\\s+)?(?:.+?)\\s+for\\s+(?:an?\\s+)?(.+)", RegexOption.IGNORE_CASE)
        val matchSub = subRegex.find(lower)
        if (matchSub != null && matchSub.groupValues.size > 1) {
            return matchSub.groupValues[1].trim()
        }
        val putRegex = Regex("(?i)put\\s+(?:an?\\s+)?(.+?)\\s+(?:on|in|onto)\\s+(?:the|a)?\\s*(.+)", RegexOption.IGNORE_CASE)
        val matchPut = putRegex.find(lower)
        if (matchPut != null && matchPut.groupValues.size > 1) {
            return matchPut.groupValues[1].trim()
        }
        return userPrompt.trim()
    }

    private fun wornPhrase(user: String): String {
        val ws = words(user)
        return when {
            ws.any { it in glassesWords } -> "worn over the eyes"
            ws.any { it in headWords } -> "worn on the head"
            ws.any { it in coverWords } -> "covering the lower face"
            ws.any { it in earWords } -> "worn on the ears"
            else -> "worn naturally"
        }
    }

    private fun absencePhrase(user: String): String {
        val ws = words(user)
        return when {
            ws.any { it in glassesWords } -> "bare eyes, no eyewear"
            ws.any { it in headWords } -> "bare head, no headwear"
            ws.any { it in coverWords } -> "uncovered face"
            ws.any { it in earWords } -> "bare ears"
            else -> ""
        }
    }

    /** Words that must not appear in the negative prompt because they describe what the user asked for. */
    private fun protectedWords(user: String): Set<String> {
        val target = extractReplacementTarget(user)
        val ws = (words(user) + words(target)).toMutableSet()
        if (ws.any { it in glassesWords }) ws += setOf("glasses", "sunglasses", "eyewear", "lens", "lenses", "frame", "frames", "shades", "reflection")
        if (ws.any { it in headWords }) ws += setOf("hat", "cap", "headwear", "brim")
        if (ws.any { it in coverWords }) ws += setOf("mask", "cover", "covered")
        return ws
    }

    // ----------------------------------------------------------------------------------------------- budget

    /** Conservative CLIP token estimate: words, single digits and punctuation each count, long words split. */
    fun estimateTokens(text: String): Int {
        var t = 0
        for (m in Regex("[a-z]+|[0-9]|[^\\sa-z0-9]").findAll(text.lowercase(Locale.US))) {
            val s = m.value
            t += if (s[0] in 'a'..'z') 1 + (s.length - 1) / 6 else 1
        }
        return t
    }

    private fun joinWithinBudget(parts: List<String>, budget: Int): String {
        val out = ArrayList<String>()
        for ((i, p) in parts.withIndex()) {
            if (p.isBlank()) continue
            val candidate = (out + p).joinToString(", ")
            if (i == 0 || estimateTokens(candidate) <= budget) out.add(p)
        }
        return out.joinToString(", ")
    }

    // ----------------------------------------------------------------------------------------------- plans

    /** Standard SD: nothing is analysed, nothing is added. */
    fun standardPlan(userPrompt: String, userNegative: String, steps: Int, seed: Int = 42): SmartSdPlan {
        val pos = userPrompt.trim()
        val neg = userNegative.trim().ifEmpty { STANDARD_DEFAULT_NEGATIVE }
        return SmartSdPlan(false, "", "", pos.ifEmpty { "[Empty]" }, pos, neg, STANDARD_CFG, steps, seed, classify(pos))
    }

    fun plan(userPrompt: String, userNegative: String, scene: SceneContext, target: TargetContext, steps: Int = 12, seed: Int = 42): SmartSdPlan {
        val user = userPrompt.trim().trimEnd(',', '.', ' ')
        val intent = classify(user)
        val onFace = target.onFace
        val skin = if (onFace) "natural skin texture" else ""
        val light = scene.lighting
        val parts = ArrayList<String>()
        val cfg: Float
        when (intent) {
            IntentKind.EMPTY -> {
                if (onFace) {
                    parts += "natural human skin and facial features"
                    parts += "photorealistic portrait"
                } else {
                    parts += "seamless photographic continuation of ${target.contactSurface}"
                    parts += "photorealistic photograph"
                }
                parts += light; parts += scene.tone; parts += "seamless continuation of the surrounding scene, no artifacts"
                cfg = 3.5f
            }
            IntentKind.REPLACEMENT -> {
                val targetObj = extractReplacementTarget(user)
                parts += targetObj
                parts += "a high quality photo of a $targetObj"
                parts += "photorealistic $targetObj"
                parts += "sharp focus"
                parts += "natural lighting"
                parts += "seamless continuation"
                cfg = 7.5f
            }
            IntentKind.OBJECT -> {
                parts += user
                parts += "$user clearly visible in the scene"
                parts += "a photo of a $user"
                parts += "photorealistic $user"
                parts += "sharp focus"
                parts += "natural lighting"
                parts += "seamless blend with the surrounding photo"
                cfg = 7.5f
            }
            IntentKind.WEARABLE -> {
                parts += user
                parts += "a person wearing $user"
                parts += "$user ${wornPhrase(user)}"
                parts += light; parts += scene.tone
                parts += "photorealistic"; parts += "sharp focus"; if (skin.isNotEmpty()) parts += skin
                parts += "seamless blend with the surrounding photo"
                cfg = 7.0f
            }
            IntentKind.FEATURE -> {
                parts += user
                parts += if (onFace) "a person with $user" else "$user clearly visible"
                parts += light; parts += scene.tone
                parts += "photorealistic"; parts += "sharp focus"; if (skin.isNotEmpty()) parts += skin
                parts += "seamless blend with the surrounding photo"
                cfg = 7.0f
            }
            IntentKind.DESCRIPTIVE -> {
                parts += user                                     // an expert prompt: verbatim, only a light touch after it
                parts += light
                parts += "seamless blend with the surrounding photo"
                cfg = 5.0f
            }
        }
        val positive = joinWithinBudget(parts, TOKEN_BUDGET)

        // negative: user's own words first, then the generic list, then the absence of what was asked for
        val prot = protectedWords(user)
        val generic = ArrayList<String>()
        generic += DEFAULT_NEGATIVE_BASE.split(", ")
        if (onFace) generic += FACE_NEGATIVE.split(", ")
        if (intent == IntentKind.WEARABLE) {
            generic += "no $user"; generic += "without $user"
            val absent = absencePhrase(user)
            if (absent.isNotEmpty()) generic += absent.split(", ")
        }
        fun conflicts(item: String): Boolean = intent != IntentKind.EMPTY &&
            words(item).any { it in prot } && !(item.startsWith("no ") || item.startsWith("without "))
        val userNeg = userNegative.trim().trimEnd(',', ' ')
        val kept = ArrayList<String>()
        if (userNeg.isNotEmpty()) kept += userNeg.split(",").map { it.trim() }.filter { it.isNotEmpty() && !conflicts(it) }
        for (g in generic) if (g !in kept && !conflicts(g)) kept += g
        val negative = kept.joinToString(", ")

        return SmartSdPlan(
            smart = true,
            stage1 = scene.describe(),
            stage2 = target.describe(),
            stage3 = if (user.isEmpty()) "[Empty]" else userPrompt.trim(),
            positive = positive,
            negative = negative,
            cfg = cfg,
            steps = steps,
            seed = seed,
            intent = intent
        )
    }
}
