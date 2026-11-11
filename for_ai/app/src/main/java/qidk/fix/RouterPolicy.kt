package qidk.fix

enum class RouterMode(val label: String) {
    MAX_QUALITY("Max Quality"),
    BALANCED_PARETO("Balanced Pareto"),
    THERMAL_SAFE("Thermal Contention Safe");

    companion object {
        /** Maps an arm name from the router file to a mode; null when the name is not recognisable. */
        fun fromArmName(name: String): RouterMode? {
            val t = name.lowercase()
            return when {
                "thermal" in t || "safe" in t || "contention" in t || "mode_3" in t -> THERMAL_SAFE
                "quality" in t || t == "max" || "mode_1" in t -> MAX_QUALITY
                "balanc" in t || "pareto" in t || "mode_2" in t -> BALANCED_PARETO
                "migan" in t -> THERMAL_SAFE
                "lama" in t || "aot" in t || "sd" in t || "diffusion" in t || "stable" in t -> MAX_QUALITY
                "telea" in t || "navier" in t || "criminisi" in t || "biharmonic" in t -> BALANCED_PARETO
                else -> null
            }
        }
    }
}

class RouterDecision(
    val mode: RouterMode,
    val armName: String,
    val scores: DoubleArray,
    /** Whole decision: telemetry snapshot + features + LinUCB. */
    val micros: Long,
    /** Feature extraction on the proxy grid only (target < 1000 us). */
    val featureMicros: Long,
    /** Standardisation + LinUCB scoring only (target < 100 us). */
    val policyMicros: Long,
    val telemetry: Telemetry
)

/**
 * Online LinUCB routing: proxy-grid features + live telemetry -> state vector -> arm. Returns null (never throws, never
 * guesses) when the model cannot be used, and [problems] says why; the caller then keeps its heuristic rule ladder.
 *
 * [alphaOverride]: null = use the alpha stored in the file (UCB); 0.0 = greedy exploitation.
 */
class RouterPolicy(
    val model: LinUcbModel,
    private val telemetry: () -> Telemetry,
    private val registry: Map<String, Feature> = RouterFeatures.registry,
    private val alphaOverride: Double? = null
) {
    val modes: List<RouterMode?> = model.armNames.map { RouterMode.fromArmName(it) }
    val problems: List<String> = buildList {
        if (model.featureNames.size != model.dim) add("file lists ${model.featureNames.size} feature names for a ${model.dim}-d state")
        val missing = model.featureNames.filterNot { StateVector.supports(it, registry) }
        if (missing.isNotEmpty()) add("no implementation for features: $missing")
        val unknownArms = model.armNames.filterIndexed { i, _ -> modes[i] == null }
        if (unknownArms.isNotEmpty()) add("arm names not mapped to a mode: $unknownArms")
    }
    val usable: Boolean get() = problems.isEmpty()

    private val raw = DoubleArray(model.dim)
    private val z = DoubleArray(model.dim)
    private val sc = DoubleArray(model.arms)

    @Synchronized
    fun decide(grid: ProxyGrid): RouterDecision? {
        if (!usable) return null
        val t0 = System.nanoTime()
        val tel = telemetry()
        val missing = StateVector.build(model.featureNames, grid, tel, raw, registry)
        if (missing.isNotEmpty()) return null
        val t1 = System.nanoTime()
        model.standardise(raw, z)
        model.scores(z, alphaOverride, sc)
        var best = 0
        for (k in 1 until sc.size) if (sc[k] > sc[best]) best = k
        val t2 = System.nanoTime()
        return RouterDecision(modes[best]!!, model.armNames[best], sc.copyOf(), (t2 - t0) / 1000, (t1 - t0) / 1000, (t2 - t1) / 1000, tel)
    }
}

/**
 * Turns a router mode into a model choice. DEFAULTS, written from the benchmark table in router.py (MIGAN 216 ms / 0.62 J /
 * +9.6 C, LaMa 321 ms / 0.99 J / +24.6 C, AOT-GAN 390 ms / 1.30 J / +25.4 C); the owner may change them - this is the only place.
 * Model names are the strings RoutingDecision.recommendedModel already uses: "MIGAN", "LAMA", "AOTGAN".
 */
object ModeDispatch {
    fun model(mode: RouterMode, ladderModel: String, maskAreaRatio: Float): String = when (mode) {
        RouterMode.MAX_QUALITY -> ladderModel                                           // whatever the rule ladder judges best
        RouterMode.BALANCED_PARETO -> if (ladderModel == "LAMA" || maskAreaRatio > 0.25f) "LAMA" else "MIGAN"
        RouterMode.THERMAL_SAFE -> "MIGAN"                                              // lowest energy and temperature rise
    }
}
