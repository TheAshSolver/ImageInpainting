package qidk.fix

enum class Backend(val label: String) {
    NPU("Hexagon NPU (HTP v79)"),
    GPU("Adreno 830 GPU (OpenCL)"),
    CPU("Oryon CPU (ARM64)")
}

enum class ModelFamily {
    SD15, MIGAN, LAMA, AOT_GAN, CLASSICAL,
    SR_BICUBIC, SR_TARSH, SR_MIGAN, SR_LAMA,
    SR_AOT_GAN, SR_SD
}

/**
 * The only place that knows which backend can run which model. Every Spinner, every router decision and the driver
 * itself must ask [allowed] / [require]; nothing may offer or dispatch a pair that is not listed here.
 */
object BackendMatrix {
    private val GAN = listOf(Backend.NPU, Backend.GPU, Backend.CPU)

    fun allowed(model: ModelFamily): List<Backend> = when (model) {
        ModelFamily.SD15, ModelFamily.SR_SD -> listOf(Backend.NPU)                    // UNet/VAE/text-encoder are HTP context binaries
        ModelFamily.MIGAN, ModelFamily.LAMA, ModelFamily.AOT_GAN -> GAN
        ModelFamily.CLASSICAL -> listOf(Backend.CPU)
        ModelFamily.SR_BICUBIC, ModelFamily.SR_TARSH -> listOf(Backend.CPU)
        ModelFamily.SR_MIGAN, ModelFamily.SR_LAMA, ModelFamily.SR_AOT_GAN -> GAN
    }

    fun isAllowed(model: ModelFamily, backend: Backend) = backend in allowed(model)

    /** Keeps a previous choice when it is still valid, otherwise falls back to the first (fastest) allowed backend. */
    fun coerce(model: ModelFamily, previous: Backend?): Backend {
        val list = allowed(model)
        return if (previous != null && previous in list) previous else list.first()
    }

    fun require(model: ModelFamily, backend: Backend) {
        require(isAllowed(model, backend)) { "$model cannot run on $backend; allowed: ${allowed(model)}" }
    }

    /**
     * Maps the model name shown in the existing model Spinner to a family. Matching is deliberately loose
     * (lower-cased substring); extend it when a new Spinner entry is added.
     */
    fun familyOf(spinnerText: String): ModelFamily? {
        val t = spinnerText.lowercase()
        return when {
            "stable" in t || "diffusion" in t || Regex("\\bsd\\b").containsMatchIn(t) -> ModelFamily.SD15
            "migan" in t || "mi-gan" in t -> ModelFamily.MIGAN
            "lama" in t -> ModelFamily.LAMA
            "aot" in t -> ModelFamily.AOT_GAN
            "criminisi" in t || "biharmonic" in t || "telea" in t || "fast marching" in t ||
                "navier" in t || "fluid" in t || "streamline" in t || "poisson" in t || "exemplar" in t -> ModelFamily.CLASSICAL
            else -> null
        }
    }
}
