// Host self-test for the NSDT kernels: g++ -O2 -std=c++17 -I app/src/main/cpp/sr tools/native/test_sr_core.cpp
//                                      app/src/main/cpp/sr/sr_core.cpp -o test_sr_core && ./test_sr_core
// Exit code 0 = every check passed.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "sr_core.h"

static int g_fail = 0;
static int g_checks = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        ++g_checks;                                       \
        if (!(cond)) {                                    \
            ++g_fail;                                     \
            std::printf("FAIL line %d: %s | ", __LINE__, #cond); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

// Independent reference: Keys cubic a = -0.5 evaluated directly from its piecewise definition.
static double keys(double d) {
    d = std::fabs(d);
    if (d <= 1.0) return 1.5 * d * d * d - 2.5 * d * d + 1.0;
    if (d < 2.0) return -0.5 * d * d * d + 2.5 * d * d - 4.0 * d + 2.0;
    return 0.0;
}

static double refUp2(const std::vector<uint8_t>& rgba, int sw, int sh, int X, int Y, int c) {
    const double sx = (X + 0.5) / 2.0 - 0.5, sy = (Y + 0.5) / 2.0 - 0.5;
    const int ix = (int)std::floor(sx), iy = (int)std::floor(sy);
    double acc = 0;
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            const int xx = qsr::reflect101(ix + i, sw), yy = qsr::reflect101(iy + j, sh);
            acc += keys(sx - (ix + i)) * keys(sy - (iy + j)) * rgba[((size_t)yy * sw + xx) * 4 + c] / 255.0;
        }
    }
    return std::fmin(1.0, std::fmax(0.0, acc));
}

static std::vector<uint8_t> randomRgba(int w, int h, unsigned seed) {
    std::mt19937 g(seed);
    std::vector<uint8_t> v((size_t)w * h * 4);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (i % 4 == 3) ? 255 : (uint8_t)(g() & 255);
    return v;
}

static void testIndexing() {
    CHECK(qsr::reflect101(-1, 10) == 1, "reflect -1");
    CHECK(qsr::reflect101(-2, 10) == 2, "reflect -2");
    CHECK(qsr::reflect101(10, 10) == 8, "reflect 10");
    CHECK(qsr::reflect101(11, 10) == 7, "reflect 11");
    CHECK(qsr::reflect101(27, 10) == 9 - ((27 % 18) - 9), "reflect far");
    CHECK(qsr::reflect101(5, 1) == 0, "reflect n=1");
    CHECK(qsr::floorDiv(-1, 32) == -1 && qsr::floorDiv(-32, 32) == -1 && qsr::floorDiv(-33, 32) == -2, "floorDiv");
    CHECK(qsr::floorDiv(31, 32) == 0 && qsr::floorDiv(32, 32) == 1, "floorDiv pos");
}

static void testSampleBase() {
    const int sw = 37, sh = 29;  // odd sizes on purpose
    auto src = randomRgba(sw, sh, 7);
    const int win = 64;
    double worst = 0;
    const int origins[][2] = {{0, 0}, {-32, -32}, {10, 20}, {2 * sw - 40, 2 * sh - 30}, {-64, 8}};
    for (auto& o : origins) {
        std::vector<float> out((size_t)win * win * 3);
        qsr::sampleBase2x(src.data(), sw, sh, sw * 4, o[0], o[1], win, out.data());
        for (int y = 0; y < win; ++y)
            for (int x = 0; x < win; ++x)
                for (int c = 0; c < 3; ++c) {
                    const double r = refUp2(src, sw, sh, o[0] + x, o[1] + y, c);
                    worst = std::fmax(worst, std::fabs(r - out[((size_t)y * win + x) * 3 + c]));
                }
    }
    CHECK(worst < 2e-6, "sampleBase2x vs brute-force reference, worst abs error %.3g", worst);

    // A constant image stays constant and a linear ramp is reproduced exactly away from the border.
    std::vector<uint8_t> flat((size_t)sw * sh * 4, 200);
    std::vector<float> o2((size_t)win * win * 3);
    qsr::sampleBase2x(flat.data(), sw, sh, sw * 4, 0, 0, win, o2.data());
    double dev = 0;
    for (float v : o2) dev = std::fmax(dev, std::fabs(v - 200.0 / 255.0));
    CHECK(dev < 1e-6, "constant image deviation %.3g", dev);

    std::vector<uint8_t> ramp((size_t)sw * sh * 4, 255);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x)
            for (int c = 0; c < 3; ++c) ramp[((size_t)y * sw + x) * 4 + c] = (uint8_t)(x * 5);
    qsr::sampleBase2x(ramp.data(), sw, sh, sw * 4, 0, 0, win, o2.data());
    double rdev = 0;
    for (int x = 8; x < 2 * sw - 8 && x < win; ++x) {
        const double expect = (((x + 0.5) / 2.0 - 0.5) * 5.0) / 255.0;
        rdev = std::fmax(rdev, std::fabs(o2[((size_t)20 * win + x) * 3] - expect));
    }
    CHECK(rdev < 1e-5, "linear ramp reproduction %.3g", rdev);
}

static void testWeights() {
    const int cell = 32, margin = 8;
    double worstSum = 0;
    int deepIn = 0, deepOut = 0;
    for (int Y = -70; Y < 140; ++Y)
        for (int X = -70; X < 140; ++X) {
            const float w0 = qsr::phaseWeight(X, Y, cell, margin, 0);
            const float w1 = qsr::phaseWeight(X, Y, cell, margin, 1);
            worstSum = std::fmax(worstSum, std::fabs(w0 + w1 - 1.0));
            CHECK(w0 >= 0 && w0 <= 1 && w1 >= 0 && w1 <= 1, "weight range");
            const int cx = qsr::floorDiv(X, cell), cy = qsr::floorDiv(Y, cell);
            const int u = X - cx * cell, v = Y - cy * cell;
            if (u >= margin && u < cell - margin && v >= margin && v < cell - margin) {
                const int phaseHole = (cx + cy) & 1;
                if (qsr::phaseWeight(X, Y, cell, margin, phaseHole) == 1.0f) ++deepIn;
                if (qsr::phaseWeight(X, Y, cell, margin, 1 - phaseHole) == 0.0f) ++deepOut;
            }
        }
    CHECK(worstSum < 1e-6, "w0 + w1 must be 1, worst deviation %.3g", worstSum);
    CHECK(deepIn > 1000 && deepIn == deepOut, "deep interior weights are 0/1 (%d %d)", deepIn, deepOut);

    std::vector<float> m0(64 * 64), m1(64 * 64);
    qsr::holeMask(-32, -16, 64, cell, 0, m0.data());
    qsr::holeMask(-32, -16, 64, cell, 1, m1.data());
    bool comp = true;
    for (size_t i = 0; i < m0.size(); ++i) comp = comp && (m0[i] + m1[i] == 1.0f);
    CHECK(comp, "hole masks of the two phases are complementary");
    // 25% hole fraction would need a sparse grid; the checkerboard is 50%.
    double frac = 0;
    for (float v : m0) frac += v;
    frac /= m0.size();
    CHECK(std::fabs(frac - 0.5) < 0.05, "checkerboard hole fraction %.3f", frac);
    // alignment: neighbouring windows agree on the same absolute pixel.
    std::vector<float> a(64 * 64), b(64 * 64);
    qsr::holeMask(0, 0, 64, cell, 0, a.data());
    qsr::holeMask(32, 32, 64, cell, 0, b.data());
    CHECK(a[40 * 64 + 40] == b[8 * 64 + 8], "absolute grid alignment");
}

static void testNullSpace() {
    const int win = 64;
    std::vector<float> Z((size_t)win * win * 3), P(Z.size());
    // constant -> 0
    std::fill(Z.begin(), Z.end(), 0.37f);
    qsr::nullSpaceProject(Z.data(), win, P.data());
    double mx = 0;
    for (float v : P) mx = std::fmax(mx, std::fabs(v));
    CHECK(mx < 1e-6, "null-space of a constant is zero, got %.3g", mx);
    // linear ramp -> 0 away from the border
    for (int y = 0; y < win; ++y)
        for (int x = 0; x < win; ++x)
            for (int c = 0; c < 3; ++c) Z[((size_t)y * win + x) * 3 + c] = 0.004f * x + 0.002f * y;
    qsr::nullSpaceProject(Z.data(), win, P.data());
    mx = 0;
    for (int y = 8; y < win - 8; ++y)
        for (int x = 8; x < win - 8; ++x) mx = std::fmax(mx, std::fabs(P[((size_t)y * win + x) * 3]));
    CHECK(mx < 2e-6, "null-space of a ramp is zero in the interior, got %.3g", mx);
    // pure Nyquist checker passes through untouched (it is invisible to the 2x2 box camera)
    for (int y = 0; y < win; ++y)
        for (int x = 0; x < win; ++x)
            for (int c = 0; c < 3; ++c) Z[((size_t)y * win + x) * 3 + c] = ((x + y) & 1) ? 0.1f : -0.1f;
    qsr::nullSpaceProject(Z.data(), win, P.data());
    mx = 0;
    for (size_t i = 0; i < Z.size(); ++i) mx = std::fmax(mx, std::fabs(P[i] - Z[i]));
    CHECK(mx < 1e-6, "checker survives the projection unchanged, deviation %.3g", mx);
    // random field: energy that the low-resolution grid can see is strongly reduced
    std::mt19937 g(5);
    std::normal_distribution<float> nd(0.f, 0.1f);
    for (auto& v : Z) v = nd(g);
    qsr::nullSpaceProject(Z.data(), win, P.data());
    auto lrEnergy = [&](const std::vector<float>& a) {
        double e = 0;
        for (int y = 0; y < win / 2; ++y)
            for (int x = 0; x < win / 2; ++x) {
                const float* p = &a[((size_t)(2 * y) * win + 2 * x) * 3];
                const double m = 0.25 * (p[0] + p[3] + p[win * 3] + p[win * 3 + 3]);
                e += m * m;
            }
        return e;
    };
    const double before = lrEnergy(Z), after = lrEnergy(P);
    CHECK(after < 0.25 * before, "LR energy after projection %.4g vs before %.4g", after, before);
}

struct Fx {
    int win = 128, cell = 32;
    std::vector<uint8_t> src;
    std::vector<float> U;
    int sw = 96, sh = 96;
    Fx() {
        src = randomRgba(sw, sh, 3);
        // smooth the source a bit so that U is a plausible blurry base
        for (int y = 1; y + 1 < sh; ++y)
            for (int x = 1; x + 1 < sw; ++x)
                for (int c = 0; c < 3; ++c) {
                    int s = 0;
                    for (int j = -1; j <= 1; ++j)
                        for (int i = -1; i <= 1; ++i) s += src[((size_t)(y + j) * sw + (x + i)) * 4 + c];
                    src[((size_t)y * sw + x) * 4 + c] = (uint8_t)(s / 9);
                }
        U.resize((size_t)win * win * 3);
        qsr::sampleBase2x(src.data(), sw, sh, sw * 4, 0, 0, win, U.data());
    }
};

static void testComposeIdentityAndGates() {
    Fx fx;
    const int win = fx.win;
    std::vector<uint8_t> dst((size_t)win * win * 4, 0);

    // 1. a network that returns its input -> output is the base, bit exact, and the stats say "identity".
    auto st = qsr::composeCore(fx.U.data(), fx.U.data(), fx.U.data(), win, 0, 0, fx.cell, 1, 1.0f, 0.08f, 0.3f,
                               16, 16, 96, 96, dst.data(), win, win, win * 4, 16, 16);
    bool exact = true;
    for (int y = 16; y < 112 && exact; ++y)
        for (int x = 16; x < 112; ++x)
            for (int c = 0; c < 3; ++c) {
                const int e = (int)std::lround(fx.U[((size_t)y * win + x) * 3 + c] * 255.0f);
                if (dst[((size_t)y * win + x) * 4 + c] != e) exact = false;
            }
    CHECK(exact, "identity network gives the classical base");
    CHECK(st.phasesUsed == 2 && st.pureHole > 0 && st.identityHole == st.pureHole, "identity stats: used=%d pure=%ld ident=%ld",
          st.phasesUsed, st.pureHole, st.identityHole);
    CHECK(st.sumAbsD == 0.0, "no detail added by an identity network");

    // 2. a network that hallucinates stationary pixel-level texture inside its holes (independent per phase).
    std::mt19937 g(11);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> T0(fx.U.size()), T1(fx.U.size());
    for (auto& v : T0) v = nd(g);
    for (auto& v : T1) v = nd(g);
    std::vector<float> o0(fx.U), o1(fx.U);
    std::vector<float> m0((size_t)win * win), m1((size_t)win * win);
    qsr::holeMask(0, 0, win, fx.cell, 0, m0.data());
    qsr::holeMask(0, 0, win, fx.cell, 1, m1.data());
    for (int y = 0; y < win; ++y)
        for (int x = 0; x < win; ++x)
            for (int c = 0; c < 3; ++c) {
                const size_t i = ((size_t)y * win + x) * 3 + c;
                o0[i] += m0[(size_t)y * win + x] * T0[i];
                o1[i] += m1[(size_t)y * win + x] * T1[i];
            }
    double bestRatio = 1e9;
    for (int mg : {0, 1, 2, 3, 4, 8}) {
        qsr::composeCore(fx.U.data(), o0.data(), o1.data(), win, 0, 0, fx.cell, mg, 1.0f, 0.5f, 0.3f, 16, 16, 96, 96,
                         dst.data(), win, win, win * 4, 16, 16);
        double prof[32] = {0};
        long cnt[32] = {0};
        for (int y = 16; y < 112; ++y)
            for (int x = 16; x < 112; ++x) {
                double d = 0;
                for (int c = 0; c < 3; ++c)
                    d += std::fabs(dst[((size_t)y * win + x) * 4 + c] / 255.0 - fx.U[((size_t)y * win + x) * 3 + c]);
                prof[x % 32] += d;
                ++cnt[x % 32];
            }
        double lo = 1e9, hi = 0;
        for (int i = 0; i < 32; ++i) {
            const double v = prof[i] / cnt[i];
            lo = std::fmin(lo, v);
            hi = std::fmax(hi, v);
        }
        std::printf("  margin %d: detail amplitude max/min across the cell = %.3f\n", mg, hi / lo);
        if (mg == 1) bestRatio = hi / lo;
    }
    CHECK(bestRatio < 1.5, "detail amplitude with the default margin 1 is not modulated by the cell grid: max/min = %.3f", bestRatio);
    st = qsr::composeCore(fx.U.data(), o0.data(), o1.data(), win, 0, 0, fx.cell, 1, 1.0f, 0.5f, 0.3f, 16, 16, 96, 96,
                          dst.data(), win, win, win * 4, 16, 16);
    CHECK(st.phasesUsed == 2 && st.phasesRejected == 0, "both phases accepted");
    CHECK(st.sumAbsD > 0, "detail was added");
    const double had = st.sumHad2 > 0 ? st.sumNyq2 / st.sumHad2 : 0;
    CHECK(had > 0.2 && had < 0.5, "white texture has ~1/3 of its energy on the diagonal, got %.3f", had);

    // 3. mesh: a network that writes a 2x2 checker into its holes -> Nyquist share close to 1.
    std::vector<float> c0(fx.U), c1(fx.U);
    for (int y = 0; y < win; ++y)
        for (int x = 0; x < win; ++x)
            for (int c = 0; c < 3; ++c) {
                const size_t i = ((size_t)y * win + x) * 3 + c;
                const float chk = ((x + y) & 1) ? 0.1f : -0.1f;
                c0[i] += m0[(size_t)y * win + x] * chk;
                c1[i] += m1[(size_t)y * win + x] * chk;
            }
    st = qsr::composeCore(fx.U.data(), c0.data(), c1.data(), win, 0, 0, fx.cell, 1, 1.0f, 0.5f, 0.3f, 16, 16, 96,
                          96, dst.data(), win, win, win * 4, 16, 16);
    const double meshShare = st.sumNyq2 / st.sumHad2;
    CHECK(meshShare > 0.85, "checker texture is flagged, Nyquist share %.3f", meshShare);

    // 4. NaN in one phase -> that phase is rejected, the other still works, output stays finite (bytes always valid).
    std::vector<float> bad(o0);
    bad[(size_t)(win * 64 + 64) * 3] = NAN;
    st = qsr::composeCore(fx.U.data(), bad.data(), o1.data(), win, 0, 0, fx.cell, 1, 1.0f, 0.5f, 0.3f, 16, 16, 96,
                          96, dst.data(), win, win, win * 4, 16, 16);
    CHECK(st.phasesRejected == 1 && st.phasesUsed == 1 && st.nonFinite >= 1, "NaN phase rejected (used=%d rej=%d)", st.phasesUsed,
          st.phasesRejected);

    // 5. black output (wrong range / polarity) -> both phases rejected by the drift gate, result equals the base.
    std::vector<float> black(fx.U.size(), 0.0f);
    st = qsr::composeCore(fx.U.data(), black.data(), black.data(), win, 0, 0, fx.cell, 1, 1.0f, 0.5f, 0.2f, 16, 16,
                          96, 96, dst.data(), win, win, win * 4, 16, 16);
    CHECK(st.phasesRejected == 2 && st.sumAbsD == 0.0, "black output rejected (rejected=%d)", st.phasesRejected);

    // 6. LR consistency of the detail result against the base: downscale(X) stays close to downscale(U).
    double sse = 0;
    long n = 0;
    st = qsr::composeCore(fx.U.data(), o0.data(), o1.data(), win, 0, 0, fx.cell, 1, 1.0f, 0.5f, 0.3f, 16, 16, 96,
                          96, dst.data(), win, win, win * 4, 16, 16);
    for (int y = 8; y < 56; ++y)
        for (int x = 8; x < 56; ++x)
            for (int c = 0; c < 3; ++c) {
                double a = 0, b = 0;
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i) {
                        a += dst[((size_t)(2 * y + j) * win + (2 * x + i)) * 4 + c] / 255.0;
                        b += fx.U[((size_t)(2 * y + j) * win + (2 * x + i)) * 3 + c];
                    }
                const double d = (a - b) * 0.25;
                sse += d * d;
                ++n;
            }
    const double rmse = std::sqrt(sse / n) * 255.0;
    // the added texture has sigma 0.05 = 12.75 levels per pixel; its 2x2 average would be ~6.4 levels if it were not projected out.
    CHECK(rmse < 2.5, "LR-domain change after detail transfer: %.2f levels rms (an unprojected texture would give ~6)", rmse);
    const double dcLevels = std::sqrt(st.sumDc2 / (st.samples / 4.0)) * 255.0;
    CHECK(dcLevels < 2.5, "reported LR-domain change %.2f levels rms", dcLevels);
}

static void testUniformMode() {
    Fx fx;
    const int win = fx.win;
    std::vector<float> m(win * win);
    qsr::holeMask(-64, 32, win, 0, 0, m.data());
    bool all1 = true;
    for (float v : m) all1 = all1 && v == 1.0f;
    qsr::holeMask(0, 0, win, 0, 1, m.data());
    bool all0 = true;
    for (float v : m) all0 = all0 && v == 0.0f;
    CHECK(all1 && all0, "uniform mode masks");
    CHECK(qsr::phaseWeight(-5, 7, 0, 0, 0) == 1.0f && qsr::phaseWeight(-5, 7, 0, 0, 1) == 0.0f, "uniform weights");
    // X - U equals the projected difference everywhere in the core when tau <= 0 (no clamp) and alpha = 1
    std::mt19937 g(21);
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> o(fx.U.size());
    for (size_t i = 0; i < o.size(); ++i) o[i] = fx.U[i] + nd(g);
    std::vector<uint8_t> dst((size_t)win * win * 4, 0);
    auto st = qsr::composeCore(fx.U.data(), o.data(), nullptr, win, 0, 0, 0, 0, 1.0f, 0.0f, 0.3f, 16, 16, 96, 96, dst.data(), win,
                               win, win * 4, 16, 16);
    CHECK(st.phasesUsed == 1 && st.phasesRejected == 0, "uniform compose uses phase 0");
    std::vector<float> Z(o.size()), P(o.size());
    for (size_t i = 0; i < o.size(); ++i) Z[i] = o[i] - fx.U[i];
    qsr::nullSpaceProject(Z.data(), win, P.data());
    double worst = 0;
    for (int y = 16; y < 112; ++y)
        for (int x = 16; x < 112; ++x)
            for (int c = 0; c < 3; ++c) {
                const float expect = std::fmin(1.f, std::fmax(0.f, fx.U[((size_t)y * win + x) * 3 + c] + P[((size_t)y * win + x) * 3 + c]));
                worst = std::fmax(worst, std::fabs(expect - dst[((size_t)y * win + x) * 4 + c] / 255.0));
            }
    CHECK(worst < 1.0 / 255.0, "uniform compose equals U + P(O-U) up to 8-bit rounding, worst %.4f", worst);
    // both phases missing: the classical base, nothing else
    st = qsr::composeCore(fx.U.data(), nullptr, nullptr, win, 0, 0, 32, 1, 1.0f, 0.08f, 0.3f, 16, 16, 96, 96, dst.data(), win,
                          win, win * 4, 16, 16);
    CHECK(st.phasesUsed == 0 && st.sumAbsD == 0.0, "no phase output gives the base");
}

static void testLrPsnr() {
    const int sw = 20, sh = 12, s = 2;
    auto src = randomRgba(sw, sh, 9);
    std::vector<uint8_t> dst((size_t)sw * s * sh * s * 4, 255);
    for (int y = 0; y < sh * s; ++y)
        for (int x = 0; x < sw * s; ++x)
            for (int c = 0; c < 3; ++c) dst[((size_t)y * sw * s + x) * 4 + c] = src[((size_t)(y / s) * sw + x / s) * 4 + c];
    CHECK(qsr::lrPsnr(src.data(), sw, sh, sw * 4, dst.data(), sw * s, sh * s, sw * s * 4, s) == 99.0, "replicated image is lr-identical");
    for (int y = 0; y < sh * s; ++y)
        for (int x = 0; x < sw * s; ++x)
            for (int c = 0; c < 3; ++c) {
                int v = dst[((size_t)y * sw * s + x) * 4 + c] + (((x + y) & 1) ? 4 : -4);
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                if (dst[((size_t)y * sw * s + x) * 4 + c] > 10 && dst[((size_t)y * sw * s + x) * 4 + c] < 245) dst[((size_t)y * sw * s + x) * 4 + c] = (uint8_t)v;
            }
    // +-4 checker averages to 0 over 2x2 blocks -> still consistent
    const double p = qsr::lrPsnr(src.data(), sw, sh, sw * 4, dst.data(), sw * s, sh * s, sw * s * 4, s);
    CHECK(p > 60.0, "checker noise is invisible to the LR operator, psnr %.1f", p);
}

int main() {
    testIndexing();
    testSampleBase();
    testWeights();
    testNullSpace();
    testComposeIdentityAndGates();
    testUniformMode();
    testLrPsnr();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
