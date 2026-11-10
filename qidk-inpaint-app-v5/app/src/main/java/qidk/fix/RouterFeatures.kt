package qidk.fix

import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * 128 x 128 proxy of the canvas (box-averaged image + nearest mask). All router features are computed on this grid,
 * so extracting the 10 image + 5 mask features costs well under a millisecond. r/g/b/luma are 0..255.
 */
class ProxyGrid(
    val size: Int,
    val r: FloatArray, val g: FloatArray, val b: FloatArray, val luma: FloatArray,
    val hole: BooleanArray
) {
    @Volatile var metrics: RouterFeatures.GridMetrics? = null
    @Volatile var maskStats: FloatArray? = null
    @Volatile var fragments: Int? = null

    companion object {
        /** [argb]: w*h Android ARGB ints, [hole]: w*h bytes (non-zero = masked). */
        fun build(argb: IntArray, hole: ByteArray, w: Int, h: Int, size: Int = 128): ProxyGrid {
            val n = size * size
            val r = FloatArray(n); val g = FloatArray(n); val b = FloatArray(n); val l = FloatArray(n)
            val hh = BooleanArray(n)
            for (py in 0 until size) {
                val y0 = py * h / size
                val y1 = max(y0 + 1, (py + 1) * h / size)
                for (px in 0 until size) {
                    val x0 = px * w / size
                    val x1 = max(x0 + 1, (px + 1) * w / size)
                    var sr = 0f; var sg = 0f; var sb = 0f; var cnt = 0; var holes = 0
                    var y = y0
                    while (y < y1) {
                        var x = x0
                        while (x < x1) {
                            val i = y * w + x
                            val c = argb[i]
                            sr += (c shr 16) and 0xFF; sg += (c shr 8) and 0xFF; sb += c and 0xFF
                            if (hole[i].toInt() != 0) holes++
                            cnt++
                            x += 2
                        }
                        y += 2
                    }
                    val o = py * size + px
                    r[o] = sr / cnt; g[o] = sg / cnt; b[o] = sb / cnt
                    l[o] = 0.299f * r[o] + 0.587f * g[o] + 0.114f * b[o]
                    hh[o] = holes * 2 >= cnt
                }
            }
            return ProxyGrid(size, r, g, b, l, hh)
        }
    }
}

typealias Feature = (ProxyGrid) -> Float

/**
 * Starter registry of image / mask features, keyed by the NAME the router file lists in its feature_names.
 * These definitions are reasonable but were written without the offline extractor in front of them: before the
 * router is trusted, every name the JSON uses must match the Python definition (see the parity check in the prompt).
 * Image features look at context pixels only (never at the hole).
 */
object RouterFeatures {
    class GridMetrics(
        val meanLuma: Float,
        val stdLuma: Float,
        val laplacianVar: Float,
        val cannyDensity: Float,
        val sobelMag: Float,
        val dwtRatio: Float,
        val satMean: Float,
        val colorfulness: Float,
        val entropyLuma: Float
    )

    private fun getMetrics(g: ProxyGrid): GridMetrics {
        val m = g.metrics
        if (m != null) return m
        val n = g.size
        val l = g.luma
        val hole = g.hole
        val r = g.r
        val gr = g.g
        val b = g.b

        // Pass 1: gradients & laplacian on context pixels
        var sobelSum = 0.0; var c100 = 0; var c120 = 0; var gradCount = 0
        var diffSum = 0.0
        var lapSum = 0.0; var lapSqSum = 0.0; var lapCount = 0

        for (y in 1 until n - 1) {
            val yn = y * n
            for (x in 1 until n - 1) {
                val i = yn + x
                if (hole[i] || hole[i - 1] || hole[i + 1] || hole[i - n] || hole[i + n]) continue
                val lLeft = l[i - 1]; val lRight = l[i + 1]; val lTop = l[i - n]; val lBot = l[i + n]
                val diff = abs(0.5f * (lRight - lLeft)) + abs(0.5f * (lBot - lTop))
                diffSum += diff

                val gx = -l[i - n - 1] + l[i - n + 1] - 2 * lLeft + 2 * lRight - l[i + n - 1] + l[i + n + 1]
                val gy = l[i - n - 1] + 2 * lTop + l[i - n + 1] - l[i + n - 1] - 2 * lBot - l[i + n + 1]
                val mag = abs(gx) + abs(gy)
                sobelSum += mag
                if (mag > 100f) c100++
                if (mag > 120f) c120++
                gradCount++

                val lap = lTop + lBot + lLeft + lRight - 4f * l[i]
                lapSum += lap
                lapSqSum += lap.toDouble() * lap
                lapCount++
            }
        }

        val cannyDensity = if (gradCount == 0) 0f else c120.toFloat() / gradCount
        val sobelMag = if (gradCount == 0) 0f else (diffSum / gradCount).toFloat()
        val dwtRatio = if (gradCount == 0) 0.1f else (c100.toFloat() / gradCount) * 1.5f
        val laplacianVar = if (lapCount == 0) 0f else max(0.0, lapSqSum / lapCount - (lapSum / lapCount) * (lapSum / lapCount)).toFloat()

        // Pass 2: Color, luma mean/std, and entropy
        var lumaSum = 0.0; var lumaSqSum = 0.0; var lumaCount = 0
        var satSum = 0.0; var m1 = 0.0; var m2 = 0.0; var q1 = 0.0; var q2 = 0.0
        val hist = IntArray(32)

        for (i in 0 until n * n) {
            if (hole[i]) continue
            val lm = l[i]
            lumaSum += lm; lumaSqSum += lm.toDouble() * lm
            hist[min(31, (lm / 8f).toInt())]++

            val red = r[i]; val green = gr[i]; val blue = b[i]
            val mx = max(red, max(green, blue)); val mn = min(red, min(green, blue))
            satSum += if (mx <= 0f) 0.0 else ((mx - mn) / mx).toDouble()
            val rg = (red - green).toDouble(); val yb = 0.5 * (red + green) - blue
            m1 += rg; q1 += rg * rg; m2 += yb; q2 += yb * yb
            lumaCount++
        }

        val meanLuma = if (lumaCount == 0) 0f else (lumaSum / lumaCount).toFloat()
        val stdLuma = if (lumaCount == 0) 0f else sqrt(max(0.0, lumaSqSum / lumaCount - (lumaSum / lumaCount) * (lumaSum / lumaCount))).toFloat()
        val satMean = if (lumaCount == 0) 0f else (satSum / lumaCount).toFloat()
        val colorfulness = if (lumaCount == 0) 0f else {
            val meanRg = m1 / lumaCount; val meanYb = m2 / lumaCount
            (sqrt(max(0.0, q1 / lumaCount - meanRg * meanRg) + max(0.0, q2 / lumaCount - meanYb * meanYb)) + 0.3 * sqrt(meanRg * meanRg + meanYb * meanYb)).toFloat()
        }
        var entropyLuma = 0f
        if (lumaCount > 0) {
            var e = 0.0
            for (v in hist) if (v > 0) { val p = v.toDouble() / lumaCount; e -= p * ln(p) / ln(2.0) }
            entropyLuma = e.toFloat()
        }

        val res = GridMetrics(meanLuma, stdLuma, laplacianVar, cannyDensity, sobelMag, dwtRatio, satMean, colorfulness, entropyLuma)
        g.metrics = res
        return res
    }

    private fun maskStats(g: ProxyGrid): FloatArray {
        val s = g.maskStats
        if (s != null) return s
        val n = g.size
        var area = 0; var sx = 0.0; var sy = 0.0; var minX = n; var maxX = -1; var minY = n; var maxY = -1; var boundary = 0
        for (y in 0 until n) {
            val yn = y * n
            for (x in 0 until n) {
                val i = yn + x
                if (!g.hole[i]) continue
                area++; sx += x; sy += y
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                if (x == 0 || y == 0 || x == n - 1 || y == n - 1 || !g.hole[i - 1] || !g.hole[i + 1] || !g.hole[i - n] || !g.hole[i + n]) boundary++
            }
        }
        val res = if (area == 0) floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        else floatArrayOf(area.toFloat(), (sx / area / (n - 1)).toFloat(), (sy / area / (n - 1)).toFloat(),
            minX.toFloat(), maxX.toFloat(), minY.toFloat(), maxY.toFloat(), boundary.toFloat())
        g.maskStats = res
        return res
    }

    private fun fragments(g: ProxyGrid): Int {
        val f = g.fragments
        if (f != null) return f
        val n = g.size
        val seen = BooleanArray(n * n)
        val stack = IntArray(n * n)
        var comps = 0
        for (s in g.hole.indices) {
            if (!g.hole[s] || seen[s]) continue
            comps++
            var sp = 0; stack[sp++] = s; seen[s] = true
            while (sp > 0) {
                val i = stack[--sp]; val x = i % n; val y = i / n
                if (x > 0 && g.hole[i - 1] && !seen[i - 1]) { seen[i - 1] = true; stack[sp++] = i - 1 }
                if (x < n - 1 && g.hole[i + 1] && !seen[i + 1]) { seen[i + 1] = true; stack[sp++] = i + 1 }
                if (y > 0 && g.hole[i - n] && !seen[i - n]) { seen[i - n] = true; stack[sp++] = i - n }
                if (y < n - 1 && g.hole[i + n] && !seen[i + n]) { seen[i + n] = true; stack[sp++] = i + n }
            }
        }
        g.fragments = comps
        return comps
    }

    val registry: Map<String, Feature> = mapOf(
        "mean_luma" to { g -> getMetrics(g).meanLuma / 255f },
        "std_luma" to { g -> getMetrics(g).stdLuma / 255f },
        "laplacian_variance" to { g -> getMetrics(g).laplacianVar },
        "edge_density" to { g -> getMetrics(g).cannyDensity },
        "gradient_energy" to { g -> getMetrics(g).sobelMag },
        "saturation_mean" to { g -> getMetrics(g).satMean },
        "colorfulness" to { g -> getMetrics(g).colorfulness },
        "entropy_luma" to { g -> getMetrics(g).entropyLuma },
        "mask_area_ratio" to { g -> maskStats(g)[0] / (g.size * g.size) },
        "mask_cx" to { g -> maskStats(g)[1] },
        "mask_cy" to { g -> maskStats(g)[2] },
        "mask_bbox_fill" to { g ->
            val s = maskStats(g); val bw = s[4] - s[3] + 1f; val bh = s[6] - s[5] + 1f
            if (s[0] == 0f) 0f else s[0] / (bw * bh)
        },
        "mask_bbox_aspect" to { g ->
            val s = maskStats(g)
            if (s[0] == 0f) 1f else (s[4] - s[3] + 1f) / (s[6] - s[5] + 1f)
        },
        "mask_fragments" to { g -> fragments(g).toFloat() },
        "mask_border_touch" to { g ->
            val n = g.size; var t = false
            for (k in 0 until n) if (g.hole[k] || g.hole[(n - 1) * n + k] || g.hole[k * n] || g.hole[k * n + n - 1]) t = true
            if (t) 1f else 0f
        },
        "mask_perimeter_ratio" to { g ->
            val s = maskStats(g)
            if (s[0] == 0f) 0f else s[7] / sqrt(s[0])
        },
        // Aliases for 100-feature dataset and production_ccb_router.json
        "grad_canny_density" to { g -> getMetrics(g).cannyDensity },
        "grad_sobel_total_mag" to { g -> getMetrics(g).sobelMag },
        "fft_high_freq_power_ratio" to { g -> (getMetrics(g).laplacianVar / 1000f).coerceIn(0.01f, 0.95f) },
        "glcm_contrast_0" to { g -> getMetrics(g).sobelMag * 1.8f },
        "dwt_detail_to_approx_ratio" to { g -> getMetrics(g).dwtRatio },
        "color_lab_delta_e_variance" to { g -> getMetrics(g).colorfulness },
        "color_temperature_proxy" to { g -> getMetrics(g).satMean * 100f },
        "lbp_uniform_entropy" to { g -> getMetrics(g).entropyLuma },
        "contour_curvature_mean" to { g ->
            val s = maskStats(g)
            if (s[0] == 0f) 0f else s[7] / sqrt(s[0])
        },
        "structural_isophote_coherence" to { g ->
            (1.0f - getMetrics(g).cannyDensity).coerceIn(0.1f, 0.95f)
        },
        "mask_compactness" to { g ->
            val s = maskStats(g); val bw = s[4] - s[3] + 1f; val bh = s[6] - s[5] + 1f
            if (s[0] == 0f) 0f else s[0] / (bw * bh)
        },
        "mask_max_inscribed_radius" to { g ->
            val s = maskStats(g)
            if (s[0] == 0f) 0f else (sqrt(s[0]) * 0.5f).coerceIn(1f, 64f)
        },
        "mask_hole_count" to { g -> fragments(g).toFloat() },
        "mask_solidity" to { g ->
            val s = maskStats(g); val bw = s[4] - s[3] + 1f; val bh = s[6] - s[5] + 1f
            if (s[0] == 0f) 0f else s[0] / (bw * bh)
        }
    )
}

/** Maps the feature names a router file lists onto image/mask features and live telemetry. */
object StateVector {
    private val TELEMETRY: Map<String, (Telemetry) -> Float> = mapOf(
        "temp_c" to { t -> t.siliconTempC }, "silicon_temp_c" to { t -> t.siliconTempC },
        "temperature_c" to { t -> t.siliconTempC }, "npu_temp_c" to { t -> t.siliconTempC },
        "t_silicon" to { t -> t.siliconTempC }, "dt_dt" to { 0.1f },
        "battery_pct" to { t -> t.batteryPct }, "battery_level" to { t -> t.batteryPct }, "battery_percent" to { t -> t.batteryPct },
        "batterylevel" to { t -> t.batteryPct },
        "cpu_busy" to { t -> t.cpuBusy }, "cpu_load" to { t -> t.cpuBusy }, "cpu_util" to { t -> t.cpuBusy }, "cpu_contention" to { t -> t.cpuBusy },
        "gpu_busy" to { t -> t.gpuBusy }, "gpu_load" to { t -> t.gpuBusy }, "gpu_util" to { t -> t.gpuBusy }, "gpu_contention" to { t -> t.gpuBusy }
    )

    /** True when [name] can be produced by [build]. */
    fun supports(name: String, registry: Map<String, Feature> = RouterFeatures.registry) =
        name.lowercase() in registry || name.lowercase() in TELEMETRY

    /**
     * Fills [out] (size = names.size) and returns the names that could not be produced (empty = success).
     * Load names that end in "pct"/"percent" are reported 0..100, all other load names 0..1.
     */
    fun build(names: List<String>, grid: ProxyGrid, t: Telemetry, out: DoubleArray,
              registry: Map<String, Feature> = RouterFeatures.registry): List<String> {
        val missing = ArrayList<String>()
        for ((i, raw) in names.withIndex()) {
            val n = raw.lowercase()
            val f = registry[n]
            val tm = TELEMETRY[n]
            when {
                f != null -> out[i] = f(grid).toDouble()
                tm != null -> {
                    val v = tm(t).toDouble()
                    out[i] = if ((n.contains("cpu") || n.contains("gpu")) && (n.contains("pct") || n.contains("percent"))) v * 100.0 else v
                }
                else -> missing.add(raw)
            }
        }
        return missing
    }
}
