// classical.h - self-contained classical inpainting kernels (no OpenCV, no Android headers).
#pragma once
#include <cstdint>

namespace qidk {

// Criminisi, Perez & Toyama (2004) exemplar-based inpainting.
//   rgba   : w*h*4 bytes, row-major RGBA8. Hole pixels are overwritten, alpha is set to 255.
//   hole   : w*h bytes, non-zero = pixel to synthesise. On return it holds the pixels that are STILL unfilled.
//   patch_radius : half patch size, clamped to [2,12] (4 => 9x9 patch).
//   search_radius: source patches are searched in a (2R+1)^2 window around the target patch; if that window holds
//                  no fully known patch the whole image is searched.
//   threads      : worker threads for the exemplar search, clamped to [1,8]. Result does not depend on it.
//   budget_ms    : stop after this many milliseconds (0 = no limit); the caller finishes the rest.
// Returns the number of hole pixels left unfilled (0 = done), -1 invalid input, -2 no fully known source patch.
int criminisi_inpaint(uint8_t* rgba, uint8_t* hole, int w, int h,
                      int patch_radius, int search_radius, int threads, int budget_ms);

// Biharmonic inpainting: solves the discrete biharmonic equation (graph-Laplacian squared, i.e. the 13-point stencil
// 1 / 2 -8 2 / 1 -8 20 -8 1 / 2 -8 2 / 1 in the interior) over the hole with the known pixels as boundary data.
// Banded Cholesky, exact up to rounding. Cost ~ n * band^2 (n = hole pixels, band = 2 * shorter side of the hole box).
//   hole : w*h bytes, non-zero = pixel to synthesise (not modified).
//   max_cells: refuse (-2) when n*(band+1) exceeds this many doubles, or when n*band^2 exceeds ~1.2e10 flops.
// Returns 0 on success, -1 invalid input, -2 hole too large, -3 out of memory / numerically not positive definite.
int biharmonic_inpaint(uint8_t* rgba, const uint8_t* hole, int w, int h, int64_t max_cells);

}  // namespace qidk
