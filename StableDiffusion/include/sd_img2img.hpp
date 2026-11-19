// sd_img2img.hpp - img2img (SDEdit) start for sd_qidk_runner_inpaint, used by the "Stable Diffusion detail transfer" super-resolution.
//
// Header only, no dependencies. Host-tested by tools/native/test_sd_img2img.cpp (in the app project).
//
// WHY: the app's old "SD latent SR" sent a 0.35-gray mask (not a hole) and let the runner start from pure noise, so the model
// re-imagined every tile from scratch: warped, hallucinated tiles. A proper "SD upscale" starts the sampler part-way down the
// schedule from the NOISED encoding of the tile and denoises only the remaining steps (SDEdit, Meng et al. 2021).
//
// The runner's schedule (sd_runner_inpaint.cpp, "schedule / fusedStep"): N Karras sigmas sigma[0] > ... > sigma[N-1] > 0,
//     alpha_t[i] = 1 / sqrt(sigma[i]^2 + 1),   sigma_t[i] = sigma[i] * alpha_t[i],   init_noise_sigma = 1.0
// (a variance-preserving parameterisation: alpha_t^2 + sigma_t^2 = 1). Pure text-to-image / inpainting starts at i = 0 from
// x = noise. img2img starts at i = k = N - round(N * strength) from
//     x_k = alpha_t[k] * z0 + sigma_t[k] * noise          (z0 = VAE-encoded tile, scaled by 0.18215 like every other latent)
// and runs the loop for i = k .. N-1 (the first executed step keeps the runner's existing "first-order step" special case).
//
// THREE EDIT POINTS in sd_runner_inpaint.cpp (command line + two places in main), all guarded by the new switches so that
// the old behaviour is bit-identical when they are absent:
//   1. argument parsing:   --strength <0..1>   (default 1.0)   and   --no-composite   (default off)
//        float strength = 1.0f; bool noComposite = false;   // set from argv next to --cfg / --seed
//   2. before the denoising loop, after the schedule is built and the 16-channel input latents exist:
//        const int k = sd_img2img::startIndex(numSteps, strength);
//        if (k > 0) sd_img2img::noiseInit(x, z0, noise, alpha_t[k], sigma_t[k], latentCount);   // x = latent being denoised
//        for (int i = k; i < numSteps; ++i) { ...existing step... }                              // was: for (int i = 0; ...)
//      With strength == 1.0 the start index is 0 and x stays the pure noise the runner already creates.
//   3. after VAE decode: if (noComposite) write the decoded image as is; else keep the existing "blend with the original
//      outside the mask". With an all-zero mask the old composite returned the INPUT image unchanged, which is exactly the
//      "output identical to input" symptom, so --no-composite is required for super-resolution tiles.
// Then: touch /data/local/tmp/sd_runtime/sd_img2img_v1 (the app's SdSuperResEngine checks for this marker).
#pragma once

#include <cmath>
#include <cstddef>

namespace sd_img2img {

// First step that is executed. strength 1 -> 0 (all steps, the existing behaviour); strength 0.3 of 12 steps -> 8 (4 steps run).
inline int startIndex(int numSteps, float strength) {
    if (numSteps <= 0) return 0;
    if (!(strength > 0.0f)) return numSteps - 1;           // nothing to denoise: run only the last step
    if (strength >= 1.0f) return 0;
    int run = static_cast<int>(std::lround(static_cast<double>(numSteps) * strength));
    if (run < 1) run = 1;
    if (run > numSteps) run = numSteps;
    return numSteps - run;
}

// x[i] = alphaT * z0[i] + sigmaT * noise[i]   (variance-preserving forward process evaluated at schedule index k)
inline void noiseInit(float* x, const float* z0, const float* noise, float alphaT, float sigmaT, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = alphaT * z0[i] + sigmaT * noise[i];
}

// The runner's Karras-derived pair for a given sigma.
inline void vpPair(float sigma, float* alphaT, float* sigmaT) {
    const float a = 1.0f / std::sqrt(sigma * sigma + 1.0f);
    *alphaT = a;
    *sigmaT = sigma * a;
}

}  // namespace sd_img2img
