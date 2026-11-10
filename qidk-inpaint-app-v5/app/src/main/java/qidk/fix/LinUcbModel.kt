package qidk.fix

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Disjoint LinUCB policy loaded from the offline-trained router file (production_ccb_router.json).
 *
 *   score_a(x) = theta_a . x + alpha * sqrt(x^T A_a^-1 x),     theta_a = A_a^-1 b_a
 *
 * The file's exact schema was not available when this class was written, so the loader accepts the usual layouts
 * (arms as list or object; per-arm A / A_inv / b / theta; stacked K x d arrays; optional scaler and feature names)
 * and records what it understood in [schemaNotes]. If it cannot find theta for every arm it throws with the list of
 * top-level keys - then adapt [parse] to the real file, do NOT guess values.
 * Pure Kotlin (no android.*), evaluated in a few microseconds for d = 20, K = 3.
 */
class LinUcbModel private constructor(
    val dim: Int,
    val armNames: List<String>,
    private val theta: Array<DoubleArray>,
    private val aInv: Array<Array<DoubleArray>?>,
    val alpha: Double,
    val featureNames: List<String>,
    private val mean: DoubleArray?,
    private val scale: DoubleArray?,
    val schemaNotes: String
) {
    val arms: Int get() = armNames.size

    /** Applies the training-time standardisation (identity if the file carries none). Result has length [dim]. */
    fun standardise(raw: DoubleArray, out: DoubleArray = DoubleArray(dim)): DoubleArray {
        require(raw.size == dim) { "state has ${raw.size} values, model expects $dim" }
        for (i in 0 until dim) {
            val m = mean?.get(i) ?: 0.0
            val s = scale?.get(i)?.takeIf { it != 0.0 && !it.isNaN() } ?: 1.0
            out[i] = (raw[i] - m) / s
        }
        return out
    }

    /** UCB score per arm for a standardised state. [alphaOverride] replaces the file's alpha (0.0 = pure exploitation). */
    fun scores(x: DoubleArray, alphaOverride: Double? = null, out: DoubleArray = DoubleArray(arms)): DoubleArray {
        require(x.size == dim) { "state has ${x.size} values, model expects $dim" }
        val a = alphaOverride ?: alpha
        for (k in 0 until arms) {
            var s = 0.0
            val th = theta[k]
            for (i in 0 until dim) s += th[i] * x[i]
            val inv = aInv[k]
            if (inv != null && a != 0.0) {
                var q = 0.0
                for (i in 0 until dim) {
                    var row = 0.0
                    val r = inv[i]
                    for (j in 0 until dim) row += r[j] * x[j]
                    q += x[i] * row
                }
                s += a * sqrt(max(0.0, q))
            }
            out[k] = s
        }
        return out
    }

    fun select(x: DoubleArray, alphaOverride: Double? = null): Int {
        val sc = scores(x, alphaOverride)
        var best = 0
        for (k in 1 until sc.size) if (sc[k] > sc[best]) best = k
        return best
    }

    companion object {
        private val ARM_NAME_KEYS = listOf("name", "arm", "id", "label", "mode", "action")
        private val THETA_KEYS = listOf("theta", "weights", "theta_weights", "w", "coef", "coefficients")
        private val AINV_KEYS = listOf("a_inv", "ainv", "a_inverse", "inv_a")
        private val MATRIX_KEYS = (THETA_KEYS + AINV_KEYS + listOf("a", "b")).toSet()

        fun parse(json: String): LinUcbModel {
            val root = MiniJson.parse(json)
            require(root is Map<*, *>) { "router file is not a JSON object" }
            val notes = StringBuilder()

            val named = findArms(root) ?: stackedArms(root, notes)
                ?: throw IllegalArgumentException("no LinUCB arms found; top-level keys = ${root.keys}")
            val names = ArrayList<String>()
            val thetas = ArrayList<DoubleArray>()
            val invs = ArrayList<Array<DoubleArray>?>()
            for ((i, pair) in named.withIndex()) {
                val (nm, arm) = pair
                val m = lowerKeys(arm)
                val invM = matrix(first(m, AINV_KEYS)) ?: matrix(m["a"])?.let { invert(it) }
                val b = numbers(m["b"])
                var th = numbers(first(m, THETA_KEYS))
                if (th == null && invM != null && b != null) th = matVec(invM, b)
                requireNotNull(th) { "arm '$nm' has neither theta nor (A, b); keys = ${arm.keys}" }
                names.add(nm.ifEmpty { "arm$i" }); thetas.add(th); invs.add(invM)
            }
            val d = thetas[0].size
            require(thetas.all { it.size == d }) { "arms disagree on dimension: ${thetas.map { it.size }}" }
            require(invs.all { it == null || (it.size == d && it.all { r -> r.size == d }) }) { "A / A_inv shape is not ${d}x$d" }
            notes.append("arms=").append(names).append(" dim=").append(d)
                .append(" ucbBonus=").append(invs.all { it != null }).append(';')

            val alpha = number(findKey(root, listOf("serve_alpha", "alpha_serve", "alpha", "exploration_alpha", "ucb_alpha"))) ?: 0.0
            val fn = findKey(root, listOf("feature_names", "state_features", "state_names", "features"))
                .let { f -> (f as? List<*>)?.mapNotNull { it as? String } ?: emptyList() }
            var mean = numbers(findKey(root, listOf("scaler_mean", "feature_mean", "mean", "mu")))
            var scale = numbers(findKey(root, listOf("scaler_scale", "feature_std", "std", "scale", "sigma")))
            if (mean != null && mean.size != d) mean = null
            if (scale != null && scale.size != d) scale = null
            notes.append(" alpha=").append(alpha).append(" featureNames=").append(fn.size)
                .append(" scaler=").append(mean != null && scale != null)
            return LinUcbModel(d, names, thetas.toTypedArray(), invs.toTypedArray(), alpha, fn, mean, scale, notes.toString())
        }

        // ---- schema discovery ---------------------------------------------------------------------------------------
        private fun lowerKeys(m: Map<*, *>): Map<String, Any?> = m.entries.associate { it.key.toString().lowercase() to it.value }
        private fun first(m: Map<String, Any?>, keys: List<String>): Any? = keys.firstNotNullOfOrNull { m[it] }
        private fun looksLikeArm(v: Any?) = v is Map<*, *> && v.keys.any { it.toString().lowercase() in MATRIX_KEYS }
        private fun armName(m: Map<*, *>): String? {
            val lm = lowerKeys(m)
            return ARM_NAME_KEYS.firstNotNullOfOrNull { lm[it]?.toString() }
        }

        private fun findArms(node: Any?, depth: Int = 0): List<Pair<String, Map<*, *>>>? {
            if (depth > 4) return null
            when (node) {
                is List<*> -> {
                    if (node.isNotEmpty() && node.all { looksLikeArm(it) })
                        return node.mapIndexed { i, m -> (armName(m as Map<*, *>) ?: "arm$i") to m }
                    for (c in node) findArms(c, depth + 1)?.let { return it }
                }
                is Map<*, *> -> {
                    if (node.isNotEmpty() && node.values.all { looksLikeArm(it) })
                        return node.entries.map { (k, v) -> (armName(v as Map<*, *>) ?: k.toString()) to v }
                    for (c in node.values) findArms(c, depth + 1)?.let { return it }
                }
            }
            return null
        }

        /** Layout with K x d theta / b and K x d x d A or A_inv at the top level. */
        private fun stackedArms(root: Map<*, *>, notes: StringBuilder): List<Pair<String, Map<*, *>>>? {
            val m = lowerKeys(root)
            val th = first(m, THETA_KEYS) as? List<*>
            val b = m["b"] as? List<*>
            val k = (th ?: b)?.size ?: return null
            if (k == 0 || ((th ?: b)!![0] !is List<*>)) return null
            val ainv = first(m, AINV_KEYS) as? List<*>
            val a = m["a"] as? List<*>
            val names = ((findKey(root, listOf("arm_names", "arms", "actions", "modes", "names")) as? List<*>)
                ?.mapNotNull { it as? String }) ?: emptyList()
            notes.append("stacked;")
            return (0 until k).map { i ->
                val arm = LinkedHashMap<String, Any?>()
                th?.let { arm["theta"] = it[i] }; b?.let { arm["b"] = it[i] }
                ainv?.let { arm["a_inv"] = it[i] }; a?.let { arm["a"] = it[i] }
                (names.getOrNull(i) ?: "arm$i") to arm
            }
        }

        private fun findKey(node: Any?, keys: List<String>, depth: Int = 0): Any? {
            if (depth > 4 || node !is Map<*, *>) return null
            val lm = lowerKeys(node)
            for (k in keys) lm[k]?.let { return it }
            for (c in node.values) findKey(c, keys, depth + 1)?.let { return it }
            return null
        }

        // ---- numeric helpers ----------------------------------------------------------------------------------------
        private fun number(v: Any?): Double? = (v as? Number)?.toDouble()
        private fun numbers(v: Any?): DoubleArray? {
            val l = v as? List<*> ?: return null
            if (l.isEmpty() || l.any { it !is Number }) return null
            return DoubleArray(l.size) { (l[it] as Number).toDouble() }
        }
        private fun matrix(v: Any?): Array<DoubleArray>? {
            val l = v as? List<*> ?: return null
            if (l.isEmpty()) return null
            val rows = l.map { numbers(it) ?: return null }
            return rows.toTypedArray()
        }
        private fun matVec(a: Array<DoubleArray>, x: DoubleArray) = DoubleArray(a.size) { i -> var s = 0.0; for (j in x.indices) s += a[i][j] * x[j]; s }

        /** Gauss-Jordan inverse with partial pivoting (d is ~20). */
        fun invert(a: Array<DoubleArray>): Array<DoubleArray> {
            val n = a.size
            require(a.all { it.size == n }) { "matrix is not square" }
            val m = Array(n) { i -> DoubleArray(2 * n) { j -> if (j < n) a[i][j] else if (j - n == i) 1.0 else 0.0 } }
            for (c in 0 until n) {
                var p = c
                for (r in c + 1 until n) if (abs(m[r][c]) > abs(m[p][c])) p = r
                require(abs(m[p][c]) > 1e-12) { "singular matrix" }
                val t = m[c]; m[c] = m[p]; m[p] = t
                val d = m[c][c]
                for (j in 0 until 2 * n) m[c][j] /= d
                for (r in 0 until n) if (r != c) {
                    val f = m[r][c]
                    if (f != 0.0) for (j in 0 until 2 * n) m[r][j] -= f * m[c][j]
                }
            }
            return Array(n) { i -> DoubleArray(n) { j -> m[i][n + j] } }
        }
    }
}
