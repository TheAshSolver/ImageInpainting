// sr_core.h - Null-Space Detail Transfer (NSDT) kernels for the neural super-resolution path.
//
// Pure C++17, no Android and no JNI headers, so the same code is unit-tested on the host (tools/native/test_sr_core.cpp)
// and linked into libqidk_sr.so on the device.
//
// Geometry (all coordinates are in the OUTPUT grid of one 2x pass, origin top-left, pixel centre = +0.5):
//   * U  = Keys cubic (a = -0.5) 2x up-sample of the source, reflect-101 borders. It is the classical base.
//   * A  = 2x2 box average (the "camera"); P(Z) = Z - UP(A(Z)) removes everything the low-resolution grid can already
//          represent, leaving only the null-space (detail) component. A(P(Z)) is ~0, so detail cannot move geometry.
//   * An inpainting model is asked to fill checkerboard cells of K x K output pixels (phase 0 = even cells, phase 1 =
//     odd cells). Its output O_p differs from U only inside the holes. Detail D = sum_p w_p * P(O_p - U), where
//     w_p = clamp((s_p * e + m) / (2m), 0, 1), e = distance of the pixel centre to the nearest cell edge and
//     s_p = +1 inside a hole of phase p, -1 outside. For two complementary phases w_0 + w_1 == 1 everywhere, so the
//     detail amplitude is spatially uniform (no periodic modulation = no "screen door"). Only the phase whose hole
//     contains a pixel carries real detail there, so a wide ramp dips the amplitude at the cell edges (measured by
//     tools/native/test_sr_core.cpp: max/min 1.11 at m=0, 1.40 at m=1, 1.77 at m>=3). The default is m=1.
//   * X = clamp01(U + alpha * tau * tanh(D / tau)).
#pragma once

#include <cstdint>
#include <cstddef>

namespace qsr {

// Reflect-101 index into [0, n).
int reflect101(int i, int n);

// Floor division and modulo for possibly negative numerators (windows at the image border start at negative origins).
int floorDiv(int a, int b);

// Writes U for the window [x0, x0 + win) x [y0, y0 + win) of the 2x grid into out (win*win*3 floats, RGB, [0,1]).
// rgba points at the source (R,G,B,A byte order, as Android's ARGB_8888 Bitmap stores it), strideBytes = row pitch.
void sampleBase2x(const uint8_t* rgba, int sw, int sh, int strideBytes, int x0, int y0, int win, float* outRGB);

// mask[y*win+x] = 1 for hole pixels of the given phase (0 or 1), else 0. cell <= 0 selects the uniform mode used by
// img2img tiles: phase 0 is a hole everywhere (weight 1), phase 1 nowhere. Cell grid is aligned to ABSOLUTE output
// coordinates, so neighbouring windows agree on which cells are holes.
void holeMask(int x0, int y0, int win, int cell, int phase, float* mask);

// Weight of phase `phase` at output pixel (X, Y): clamp((s*e + m) / (2m), 0, 1). m <= 0 gives a hard 0/1 weight.
float phaseWeight(int X, int Y, int cell, int margin, int phase);

// out = Z - UP2(A(Z)) for a win x win x 3 image (win even). Border handling is reflect-101.
void nullSpaceProject(const float* Z, int win, float* out);

struct ComposeStats {
    double sumAbsD = 0;        // sum |alpha * soft(D)| over written samples
    double sumD2 = 0;          // sum D^2 over written samples (after clamp, before alpha)
    double sumNyq2 = 0;        // sum of squared diagonal-Hadamard coefficients of D (2x2 blocks)
    double sumHad2 = 0;        // sum of squared horizontal+vertical+diagonal Hadamard coefficients
    double sumDc2 = 0;         // sum of squared 2x2 block means of alpha*D: the change the detail makes in the LOW-resolution domain
    double sumAbsOU = 0;       // sum |O - U| over pure-hole samples (both phases)
    double sumOU = 0;          // sum (O - U) over pure-hole samples (signed drift)
    long samples = 0;          // written samples (core pixels * 3)
    long pureHole = 0;         // pure-hole samples (w == 1 for the phase whose output is read)
    long identityHole = 0;     // pure-hole samples whose |O - U| < 0.5/255
    long clamped = 0;          // samples where the soft clamp changed D by more than 10 percent
    long nonFinite = 0;        // NaN / Inf found in O
    int phasesUsed = 0;        // 0, 1 or 2 phase outputs that passed the per-window sanity check
    int phasesRejected = 0;    // phases dropped (non finite, or mean drift > driftLimit)
};

// Combines the base U (win*win*3) with the model outputs of the two phases (o1 may be nullptr) and writes the core
// rectangle [cx0, cx0+cw) x [cy0, cy0+ch) (window coordinates) into dst (RGBA8888, alpha forced to 255) at
// (dstX, dstY). alpha scales the detail, tau is the soft clamp (in [0,1] units), driftLimit rejects a phase whose pure-hole
// mean |O - U| exceeds it (a black or blown-out output). Returns the statistics for the gates.
ComposeStats composeCore(const float* U, const float* o0, const float* o1, int win, int winX0, int winY0, int cell,
                         int margin, float alpha, float tau, float driftLimit, int cx0, int cy0, int cw, int ch,
                         uint8_t* dst, int dstW, int dstH, int dstStrideBytes, int dstX, int dstY);

// Writes the classical base (alpha = 0 path) of the core rectangle: identical to composeCore with no phase output.
void writeBaseCore(const float* U, int win, int cx0, int cy0, int cw, int ch, uint8_t* dst, int dstW, int dstH,
                   int dstStrideBytes, int dstX, int dstY);

// PSNR (dB, 8-bit RGB) between src and the scale x scale box average of dst. Returns +inf style 99.0 when identical.
double lrPsnr(const uint8_t* srcRgba, int sw, int sh, int srcStride, const uint8_t* dstRgba, int dw, int dh,
              int dstStride, int scale);

}  // namespace qsr
