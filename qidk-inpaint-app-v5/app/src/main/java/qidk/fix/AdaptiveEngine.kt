package qidk.fix

import java.util.ArrayDeque
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

enum class OperationalMode(val label: String, val wQ: Float, val wL: Float, val wP: Float) {
    QUALITY("Quality Mode (High-Fidelity)", 0.70f, 0.15f, 0.15f),
    BALANCED("Balanced Mode (Pareto Optimal)", 0.40f, 0.35f, 0.25f),
    POWER_EFFICIENCY("Power Efficiency (Battery/Thermal Priority)", 0.20f, 0.30f, 0.50f)
}

enum class RouterAlgorithm(val label: String) {
    HEURISTIC("Heuristic (Rule Ladder)"),
    LINUCB("LinUCB (Contextual Bandit)"),
    RL_POLICY("RL Policy (Actor-Critic)"),
    DEEP_MLP("Deep Learning (Multi-Head MLP)")
}

data class PredictionEstimate(
    val modelName: String,
    val expectedLatencyMs: Float,
    val expectedPowerW: Float,
    val expectedSsim: Float,
    val expectedLpips: Float,
    val driftMultiplier: Float
) {
    fun summary(): String = String.format(
        Locale.US,
        "🔮 Predicted: %.0f ms | %.2f W | SSIM %.3f | Drift: %.2f×%s",
        expectedLatencyMs,
        expectedPowerW,
        expectedSsim,
        driftMultiplier,
        if (driftMultiplier > 1.25f) " [CONTENDED -> REROUTING]" else " [OPTIMAL]"
    )
}

/**
 * Real-time "Alive" adaptation engine tracking residual latency drift over a sliding FIFO window of W=25.
 */
object AdaptiveDriftTracker {
    private const val WINDOW_SIZE = 25
    private val latencyResiduals = ArrayDeque<Double>()

    @Volatile var currentDriftMultiplier: Float = 1.0f; private set

    @Synchronized
    fun recordInference(predictedMs: Float, actualMs: Float) {
        if (predictedMs <= 0f || actualMs <= 0f) return
        val ratio = (actualMs / predictedMs).toDouble()
        if (latencyResiduals.size >= WINDOW_SIZE) latencyResiduals.removeFirst()
        latencyResiduals.addLast(ratio)
        val avg = latencyResiduals.average()
        currentDriftMultiplier = avg.toFloat().coerceIn(0.6f, 3.5f)
    }

    fun isThrottled(): Boolean = currentDriftMultiplier > 1.25f

    @Synchronized
    fun reset() {
        latencyResiduals.clear()
        currentDriftMultiplier = 1.0f
    }
}

/**
 * Predicts latency, power, and quality prior to dispatch based on mask area, texture, and silicon thermals.
 */
object PredictionEngine {
    fun predict(modelName: String, maskAreaRatio: Float, edgeDensity: Float, tempC: Float): PredictionEstimate {
        val drift = AdaptiveDriftTracker.currentDriftMultiplier
        val thermalMult = 1.0f + max(0.0f, (tempC - 35.0f) * 0.015f)

        val (baseLat, basePwr, baseSsim, baseLpips) = when (modelName.uppercase()) {
            "SD_DUAL", "STABLE DIFFUSION DUAL-DLC" -> Tuple4(1529.24f, 6.85f, 0.982f, 0.042f)
            "LAMA", "LAMA DILATED" -> Tuple4(129.2f, 3.45f, 0.958f, 0.072f)
            "AOTGAN", "AOT-GAN" -> Tuple4(179.9f, 4.10f, 0.964f, 0.065f)
            "MIGAN" -> Tuple4(31.1f, 2.10f, 0.942f, 0.088f)
            else -> Tuple4(18.5f, 1.45f, 0.845f, 0.280f) // Algorithmic / Telea
        }

        val lat = baseLat * thermalMult * drift
        val pwr = basePwr + max(0.0f, (tempC - 32.0f) * 0.05f)

        var ssim = baseSsim
        var lpips = baseLpips
        if (modelName.contains("LAMA") && maskAreaRatio > 0.15f) {
            ssim += 0.008f
            lpips -= 0.012f
        } else if (modelName.contains("MIGAN") && maskAreaRatio > 0.20f) {
            ssim -= (maskAreaRatio - 0.20f) * 0.35f
            lpips += (maskAreaRatio - 0.20f) * 0.40f
        }

        return PredictionEstimate(
            modelName = modelName,
            expectedLatencyMs = lat,
            expectedPowerW = pwr,
            expectedSsim = ssim.coerceIn(0.50f, 0.999f),
            expectedLpips = lpips.coerceIn(0.01f, 0.50f),
            driftMultiplier = drift
        )
    }

    private data class Tuple4<A, B, C, D>(val a: A, val b: B, val c: C, val d: D)
}

/**
 * Online Reinforcement Learning / Bandit state updater driven by implicit user Save vs Discard actions.
 * Implements exact Sherman-Morrison Woodbury rank-1 covariance updates:
 *   A^{-1} <- A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
 *   b <- b + r * x
 *   theta <- A^{-1} b
 * Latency overhead: ~2-5 microseconds on Oryon CPU.
 */
object OnlineLearningEngine {
    private var lastArm: String? = null
    private var lastState: DoubleArray? = null

    private const val DEFAULT_DIM = 20
    private val armAInv = HashMap<String, Array<DoubleArray>>()
    private val armB = HashMap<String, DoubleArray>()
    private val armTheta = HashMap<String, DoubleArray>()

    init {
        val defaultArms = listOf("MIGAN", "LAMA", "AOTGAN", "SD_DUAL", "TELEA", "Algorithmic")
        for (arm in defaultArms) {
            initArm(arm, DEFAULT_DIM)
        }
    }

    private fun initArm(arm: String, d: Int) {
        val inv = Array(d) { i -> DoubleArray(d) { j -> if (i == j) 1.0 else 0.0 } }
        armAInv[arm] = inv
        armB[arm] = DoubleArray(d)
        armTheta[arm] = DoubleArray(d)
    }

    @Synchronized
    fun recordDecision(arm: String, state: DoubleArray) {
        lastArm = arm
        lastState = state.copyOf()
    }

    @Synchronized
    fun onUserFeedback(saved: Boolean): String? {
        val arm = lastArm ?: return null
        val x = lastState ?: return null
        val d = x.size
        val r = if (saved) 1.0 else -1.0

        if (!armAInv.containsKey(arm) || armAInv[arm]?.size != d) {
            initArm(arm, d)
        }

        val aInv = armAInv[arm]!!
        val b = armB[arm]!!

        // 1. Compute v = A^{-1} x
        val v = DoubleArray(d)
        for (i in 0 until d) {
            var sum = 0.0
            val row = aInv[i]
            for (j in 0 until d) {
                sum += row[j] * x[j]
            }
            v[i] = sum
        }

        // 2. Compute denominator = 1.0 + x^T v
        var denom = 1.0
        for (i in 0 until d) {
            denom += x[i] * v[i]
        }

        // 3. Update A^{-1} = A^{-1} - (v v^T) / denom
        for (i in 0 until d) {
            val vi = v[i]
            val row = aInv[i]
            for (j in 0 until d) {
                row[j] -= (vi * v[j]) / denom
            }
        }

        // 4. Update b = b + r * x
        for (i in 0 until d) {
            b[i] += r * x[i]
        }

        // 5. Update theta = A^{-1} b
        val theta = armTheta[arm]!!
        var thetaNormSq = 0.0
        for (i in 0 until d) {
            var sum = 0.0
            val row = aInv[i]
            for (j in 0 until d) {
                sum += row[j] * b[j]
            }
            theta[i] = sum
            thetaNormSq += sum * sum
        }
        val thetaNorm = kotlin.math.sqrt(thetaNormSq)

        val msg = if (saved) "🎯 Online Bandit Reward: +1.0 (Saved)" else "⚠️ Online Bandit Penalty: -1.0 (Discarded)"
        lastArm = null
        lastState = null
        return "$msg for $arm | ||θ||=%.3f".format(Locale.US, thetaNorm)
    }
}
