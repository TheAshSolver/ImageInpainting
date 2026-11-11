// g++ -O2 -std=c++17 -I REPO/StableDiffusion/include tools/native/test_sd_img2img.cpp -o t && ./t
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "sd_img2img.hpp"

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %d: %s | ", __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
    using namespace sd_img2img;
    CHECK(startIndex(12, 1.0f) == 0, "strength 1 keeps the old behaviour");
    CHECK(startIndex(12, 1.5f) == 0, "clamped above");
    CHECK(startIndex(12, 0.30f) == 8, "0.3 of 12 steps = round(3.6) = 4 steps run -> start 8, got %d", startIndex(12, 0.30f));
    CHECK(startIndex(20, 0.25f) == 15, "0.25 of 20 = 5 steps -> start 15");
    CHECK(startIndex(12, 0.01f) == 11, "at least one step runs");
    CHECK(startIndex(12, 0.0f) == 11, "strength 0 runs the last step only");
    CHECK(startIndex(1, 0.5f) == 0, "single step");
    for (int n : {2, 12, 20, 50}) {
        int prev = -1;
        for (float s = 0.05f; s <= 1.0f; s += 0.05f) {
            const int k = startIndex(n, s);
            CHECK(k >= 0 && k < n, "index in range n=%d s=%.2f k=%d", n, s, k);
            CHECK(prev < 0 || k <= prev, "more strength never starts later (n=%d s=%.2f)", n, s);
            prev = k;
        }
    }
    // variance preserving: alpha^2 + sigma_t^2 == 1 for every Karras sigma
    for (float sg : {0.03f, 0.3f, 1.0f, 5.0f, 14.6f}) {
        float a, s;
        vpPair(sg, &a, &s);
        CHECK(std::fabs(a * a + s * s - 1.0f) < 1e-6f, "VP identity at sigma %.2f", sg);
    }
    // statistics: unit-variance z0 and noise keep unit variance; the mean of z0 is preserved with weight alpha
    std::mt19937 g(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t n = 200000;
    std::vector<float> z0(n), eps(n), x(n);
    for (size_t i = 0; i < n; ++i) { z0[i] = 0.7f + nd(g); eps[i] = nd(g); }
    float a, s;
    vpPair(1.2f, &a, &s);
    noiseInit(x.data(), z0.data(), eps.data(), a, s, n);
    double mean = 0, var = 0;
    for (float v : x) mean += v;
    mean /= n;
    for (float v : x) var += (v - mean) * (v - mean);
    var /= n;
    CHECK(std::fabs(mean - 0.7 * a) < 0.01, "mean %.4f expected %.4f", mean, 0.7 * a);
    CHECK(std::fabs(var - 1.0) < 0.02, "variance %.4f", var);
    // a low sigma start is dominated by the encoded tile (that is what keeps the geometry)
    vpPair(0.1f, &a, &s);
    CHECK(a > 0.99f && s < 0.1f, "late start keeps the content: alpha %.3f sigma_t %.3f", a, s);
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
