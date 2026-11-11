// sr_core.cpp - see sr_core.h for the maths.
#include "sr_core.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qsr {

namespace {

// Keys cubic convolution, a = -0.5, evaluated for the 2x up-sample: output pixel 2k reads source taps k-2..k+1 and
// output pixel 2k+1 reads taps k-1..k+2 (centre-aligned mapping sx = (X + 0.5) / 2 - 0.5).
constexpr float kEven[4] = {-0.0234375f, 0.2265625f, 0.8671875f, -0.0703125f};
constexpr float kOdd[4] = {-0.0703125f, 0.8671875f, 0.2265625f, -0.0234375f};

// NaN maps to 0 (the comparison is false), so a corrupt value can never reach the byte conversion.
inline float clamp01(float v) { return v > 0.f ? (v < 1.f ? v : 1.f) : 0.f; }

// Separable 2x up-sample of a window of an image given by an accessor acc(row, col, channel) -> float.
// The window is [x0, x0+ww) x [y0, y0+wh) of the 2x grid; sw x sh is the source size (reflect-101 borders).
template <typename Acc>
void up2Window(const Acc& acc, int sw, int sh, int x0, int y0, int ww, int wh, float* out) {
    // Source row ids touched by the window (unreflected): k-2 .. k+2 for k in [floor(y0/2), floor((y0+wh-1)/2)].
    const int kyMin = floorDiv(y0, 2) - 2;
    const int kyMax = floorDiv(y0 + wh - 1, 2) + 2;
    const int rows = kyMax - kyMin + 1;

    // Per output column: tap indices (already reflected) and the 4 weights.
    std::vector<int> colIdx(static_cast<size_t>(ww) * 4);
    std::vector<const float*> colW(static_cast<size_t>(ww));
    for (int i = 0; i < ww; ++i) {
        const int X = x0 + i;
        const int k = floorDiv(X, 2);
        const bool odd = (X - 2 * k) == 1;
        const int first = odd ? k - 1 : k - 2;
        for (int t = 0; t < 4; ++t) colIdx[static_cast<size_t>(i) * 4 + t] = reflect101(first + t, sw);
        colW[static_cast<size_t>(i)] = odd ? kOdd : kEven;
    }

    // Horizontal pass for every touched source row.
    std::vector<float> H(static_cast<size_t>(rows) * ww * 3);
    for (int r = 0; r < rows; ++r) {
        const int sy = reflect101(kyMin + r, sh);
        float* hr = &H[static_cast<size_t>(r) * ww * 3];
        for (int i = 0; i < ww; ++i) {
            const int* ci = &colIdx[static_cast<size_t>(i) * 4];
            const float* w = colW[static_cast<size_t>(i)];
            for (int c = 0; c < 3; ++c) {
                hr[i * 3 + c] = w[0] * acc(sy, ci[0], c) + w[1] * acc(sy, ci[1], c) + w[2] * acc(sy, ci[2], c) +
                                w[3] * acc(sy, ci[3], c);
            }
        }
    }

    // Vertical pass.
    for (int j = 0; j < wh; ++j) {
        const int Y = y0 + j;
        const int k = floorDiv(Y, 2);
        const bool odd = (Y - 2 * k) == 1;
        const int first = odd ? k - 1 : k - 2;
        const float* w = odd ? kOdd : kEven;
        const float* r0 = &H[static_cast<size_t>(first + 0 - kyMin) * ww * 3];
        const float* r1 = &H[static_cast<size_t>(first + 1 - kyMin) * ww * 3];
        const float* r2 = &H[static_cast<size_t>(first + 2 - kyMin) * ww * 3];
        const float* r3 = &H[static_cast<size_t>(first + 3 - kyMin) * ww * 3];
        float* o = out + static_cast<size_t>(j) * ww * 3;
        for (int i = 0; i < ww * 3; ++i) o[i] = w[0] * r0[i] + w[1] * r1[i] + w[2] * r2[i] + w[3] * r3[i];
    }
}

}  // namespace

int reflect101(int i, int n) {
    if (n <= 1) return 0;
    const int period = 2 * (n - 1);
    int r = i % period;
    if (r < 0) r += period;
    return r < n ? r : period - r;
}

int floorDiv(int a, int b) {
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

void sampleBase2x(const uint8_t* rgba, int sw, int sh, int strideBytes, int x0, int y0, int win, float* outRGB) {
    auto acc = [&](int y, int x, int c) -> float {
        return static_cast<float>(rgba[static_cast<size_t>(y) * strideBytes + static_cast<size_t>(x) * 4 + c]) *
               (1.0f / 255.0f);
    };
    up2Window(acc, sw, sh, x0, y0, win, win, outRGB);
    const size_t n = static_cast<size_t>(win) * win * 3;
    for (size_t i = 0; i < n; ++i) outRGB[i] = clamp01(outRGB[i]);
}

void holeMask(int x0, int y0, int win, int cell, int phase, float* mask) {
    if (cell <= 0) {  // uniform mode (img2img tiles): phase 0 is a hole everywhere, phase 1 nowhere
        std::fill(mask, mask + static_cast<size_t>(win) * win, phase == 0 ? 1.0f : 0.0f);
        return;
    }
    for (int y = 0; y < win; ++y) {
        const int cy = floorDiv(y0 + y, cell);
        for (int x = 0; x < win; ++x) {
            const int cx = floorDiv(x0 + x, cell);
            mask[static_cast<size_t>(y) * win + x] = (((cx + cy) & 1) == phase) ? 1.0f : 0.0f;
        }
    }
}

float phaseWeight(int X, int Y, int cell, int margin, int phase) {
    if (cell <= 0) return phase == 0 ? 1.0f : 0.0f;
    const int cx = floorDiv(X, cell);
    const int cy = floorDiv(Y, cell);
    const bool hole = (((cx + cy) & 1) == phase);
    if (margin <= 0) return hole ? 1.0f : 0.0f;
    const float u = static_cast<float>(X - cx * cell) + 0.5f;
    const float v = static_cast<float>(Y - cy * cell) + 0.5f;
    const float e = std::min(std::min(u, static_cast<float>(cell) - u), std::min(v, static_cast<float>(cell) - v));
    const float s = hole ? 1.0f : -1.0f;
    const float w = (s * e + static_cast<float>(margin)) / (2.0f * static_cast<float>(margin));
    return clamp01(w);
}

void nullSpaceProject(const float* Z, int win, float* out) {
    const int h = win / 2;
    std::vector<float> down(static_cast<size_t>(h) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < h; ++x) {
            const float* a = Z + (static_cast<size_t>(2 * y) * win + 2 * x) * 3;
            const float* b = a + 3;
            const float* c = a + static_cast<size_t>(win) * 3;
            const float* d = c + 3;
            for (int k = 0; k < 3; ++k) down[(static_cast<size_t>(y) * h + x) * 3 + k] = 0.25f * (a[k] + b[k] + c[k] + d[k]);
        }
    }
    auto acc = [&](int y, int x, int c) -> float { return down[(static_cast<size_t>(y) * h + x) * 3 + c]; };
    up2Window(acc, h, h, 0, 0, win, win, out);
    const size_t n = static_cast<size_t>(win) * win * 3;
    for (size_t i = 0; i < n; ++i) out[i] = Z[i] - out[i];
}

ComposeStats composeCore(const float* U, const float* o0, const float* o1, int win, int winX0, int winY0, int cell,
                         int margin, float alpha, float tau, float driftLimit, int cx0, int cy0, int cw, int ch,
                         uint8_t* dst, int dstW, int dstH, int dstStrideBytes, int dstX, int dstY) {
    ComposeStats st;
    const size_t n = static_cast<size_t>(win) * win * 3;
    const float* outs[2] = {o0, o1};
    std::vector<float> P[2];
    bool use[2] = {false, false};

    for (int p = 0; p < 2; ++p) {
        if (!outs[p]) continue;
        const float* o = outs[p];
        // Per-window sanity. 1) every value of the window must be finite (the projection spreads a NaN over its
        // neighbours). 2) no black / blown-out output: mean |O - U| over the pure hole must stay below driftLimit.
        long nonFinite = 0, cnt = 0, ident = 0;
        double sumAbs = 0, sumSigned = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(o[i])) ++nonFinite;
        }
        if (nonFinite == 0) {
            for (int y = cy0; y < cy0 + ch; ++y) {
                for (int x = cx0; x < cx0 + cw; ++x) {
                    if (phaseWeight(winX0 + x, winY0 + y, cell, margin, p) < 0.999f) continue;
                    const size_t base = (static_cast<size_t>(y) * win + x) * 3;
                    for (int c = 0; c < 3; ++c) {
                        const float d = o[base + c] - U[base + c];
                        sumAbs += std::fabs(d);
                        sumSigned += d;
                        if (std::fabs(d) < 0.5f / 255.0f) ++ident;
                        ++cnt;
                    }
                }
            }
        }
        st.nonFinite += nonFinite;
        if (nonFinite > 0 || (cnt > 0 && sumAbs / static_cast<double>(cnt) > static_cast<double>(driftLimit))) {
            ++st.phasesRejected;
            continue;
        }
        st.pureHole += cnt;
        st.identityHole += ident;
        st.sumAbsOU += sumAbs;
        st.sumOU += sumSigned;

        std::vector<float> Z(n);
        for (size_t i = 0; i < n; ++i) Z[i] = o[i] - U[i];
        P[p].assign(n, 0.0f);
        nullSpaceProject(Z.data(), win, P[p].data());
        use[p] = true;
        ++st.phasesUsed;
    }

    // Soft-clamped detail for the core rectangle.
    std::vector<float> Dc(static_cast<size_t>(cw) * ch * 3, 0.0f);
    const float invTau = tau > 0.f ? 1.0f / tau : 0.f;
    for (int y = 0; y < ch; ++y) {
        for (int x = 0; x < cw; ++x) {
            const int wx = cx0 + x, wy = cy0 + y;
            float w0 = 0.f, w1 = 0.f;
            if (use[0]) w0 = phaseWeight(winX0 + wx, winY0 + wy, cell, margin, 0);
            if (use[1]) w1 = phaseWeight(winX0 + wx, winY0 + wy, cell, margin, 1);
            if (w0 == 0.f && w1 == 0.f) continue;
            const size_t src = (static_cast<size_t>(wy) * win + wx) * 3;
            float* d = &Dc[(static_cast<size_t>(y) * cw + x) * 3];
            for (int c = 0; c < 3; ++c) {
                float v = 0.f;
                if (w0 != 0.f) v += w0 * P[0][src + c];
                if (w1 != 0.f) v += w1 * P[1][src + c];
                float sv = v;
                if (tau > 0.f) sv = tau * std::tanh(v * invTau);
                if (std::fabs(v) > 1e-6f && std::fabs(sv - v) > 0.1f * std::fabs(v)) ++st.clamped;
                d[c] = sv;
            }
        }
    }

    // Write X = clamp01(U + alpha * D) and gather the detail statistics.
    for (int y = 0; y < ch; ++y) {
        const int oy = dstY + y;
        for (int x = 0; x < cw; ++x) {
            const int ox = dstX + x;
            const size_t src = (static_cast<size_t>(cy0 + y) * win + (cx0 + x)) * 3;
            const float* d = &Dc[(static_cast<size_t>(y) * cw + x) * 3];
            float rgb[3];
            for (int c = 0; c < 3; ++c) {
                const float a = alpha * d[c];
                rgb[c] = clamp01(U[src + c] + a);
                st.sumAbsD += std::fabs(a);
                st.sumD2 += static_cast<double>(d[c]) * d[c];
                ++st.samples;
            }
            if (ox < 0 || oy < 0 || ox >= dstW || oy >= dstH) continue;
            uint8_t* px = dst + static_cast<size_t>(oy) * dstStrideBytes + static_cast<size_t>(ox) * 4;
            px[0] = static_cast<uint8_t>(std::lround(rgb[0] * 255.0f));
            px[1] = static_cast<uint8_t>(std::lround(rgb[1] * 255.0f));
            px[2] = static_cast<uint8_t>(std::lround(rgb[2] * 255.0f));
            px[3] = 255;
        }
    }

    // Hadamard split of the detail over 2x2 blocks: the diagonal coefficient is the "checker / screen door" component.
    for (int by = 0; by + 1 < ch; by += 2) {
        for (int bx = 0; bx + 1 < cw; bx += 2) {
            const float* a = &Dc[(static_cast<size_t>(by) * cw + bx) * 3];
            const float* b = a + 3;
            const float* c2 = a + static_cast<size_t>(cw) * 3;
            const float* d2 = c2 + 3;
            for (int k = 0; k < 3; ++k) {
                const double dc = 0.25 * (a[k] + b[k] + c2[k] + d2[k]) * static_cast<double>(alpha);
                st.sumDc2 += dc * dc;
                const double hx = 0.25 * (a[k] - b[k] + c2[k] - d2[k]);
                const double hy = 0.25 * (a[k] + b[k] - c2[k] - d2[k]);
                const double hxy = 0.25 * (a[k] - b[k] - c2[k] + d2[k]);
                st.sumNyq2 += hxy * hxy;
                st.sumHad2 += hx * hx + hy * hy + hxy * hxy;
            }
        }
    }
    return st;
}

void writeBaseCore(const float* U, int win, int cx0, int cy0, int cw, int ch, uint8_t* dst, int dstW, int dstH,
                   int dstStrideBytes, int dstX, int dstY) {
    for (int y = 0; y < ch; ++y) {
        const int oy = dstY + y;
        if (oy < 0 || oy >= dstH) continue;
        for (int x = 0; x < cw; ++x) {
            const int ox = dstX + x;
            if (ox < 0 || ox >= dstW) continue;
            const float* s = U + (static_cast<size_t>(cy0 + y) * win + (cx0 + x)) * 3;
            uint8_t* px = dst + static_cast<size_t>(oy) * dstStrideBytes + static_cast<size_t>(ox) * 4;
            px[0] = static_cast<uint8_t>(std::lround(clamp01(s[0]) * 255.0f));
            px[1] = static_cast<uint8_t>(std::lround(clamp01(s[1]) * 255.0f));
            px[2] = static_cast<uint8_t>(std::lround(clamp01(s[2]) * 255.0f));
            px[3] = 255;
        }
    }
}

double lrPsnr(const uint8_t* srcRgba, int sw, int sh, int srcStride, const uint8_t* dstRgba, int dw, int dh,
              int dstStride, int scale) {
    if (scale < 1 || dw < sw * scale || dh < sh * scale) return -1.0;
    double sse = 0;
    const double inv = 1.0 / (static_cast<double>(scale) * scale);
    for (int y = 0; y < sh; ++y) {
        for (int x = 0; x < sw; ++x) {
            const uint8_t* s = srcRgba + static_cast<size_t>(y) * srcStride + static_cast<size_t>(x) * 4;
            double acc[3] = {0, 0, 0};
            for (int j = 0; j < scale; ++j) {
                const uint8_t* row = dstRgba + static_cast<size_t>(y * scale + j) * dstStride + static_cast<size_t>(x * scale) * 4;
                for (int i = 0; i < scale; ++i) {
                    acc[0] += row[i * 4 + 0];
                    acc[1] += row[i * 4 + 1];
                    acc[2] += row[i * 4 + 2];
                }
            }
            for (int c = 0; c < 3; ++c) {
                const double d = acc[c] * inv - static_cast<double>(s[c]);
                sse += d * d;
            }
        }
    }
    const double mse = sse / (static_cast<double>(sw) * sh * 3.0);
    if (mse < 1e-12) return 99.0;
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

}  // namespace qsr
