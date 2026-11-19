// selftest.cpp - host-side checks for classical.h. Build: g++ -O2 -std=c++17 selftest.cpp criminisi.cpp biharmonic.cpp -o selftest
//   ./selftest                         -> built-in self tests (exit code 0 = all pass)
//   ./selftest crim  in.rgba mask.u8 W H out.rgba [patchR searchR threads budgetMs]
//   ./selftest bih   in.rgba mask.u8 W H out.rgba
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "classical.h"

using Clock = std::chrono::steady_clock;
static double msSince(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++fails; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } else { std::printf("ok   "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::vector<uint8_t> readAll(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void errStats(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const std::vector<uint8_t>& holeMask, double& mean, int& mx) {
    double sum = 0; int cnt = 0; mx = 0;
    for (size_t i = 0; i < holeMask.size(); ++i) {
        if (!holeMask[i]) continue;
        for (int c = 0; c < 3; ++c) {
            const int d = std::abs(int(a[4 * i + c]) - int(b[4 * i + c]));
            sum += d; ++cnt; if (d > mx) mx = d;
        }
    }
    mean = cnt ? sum / cnt : 0;
}

static void selfTests() {
    // --- 1. biharmonic reproduces a cubic surface (the discrete biharmonic operator annihilates cubics) -------------
    {
        const int w = 96, h = 80;
        std::vector<uint8_t> truth(4 * w * h), img, hole(w * h, 0);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const double fx = x / 96.0, fy = y / 80.0;
                const double v0 = 90 + 80 * fx + 40 * fy + 30 * fx * fx - 50 * fx * fy + 20 * fy * fy + 25 * fx * fx * fx - 30 * fx * fx * fy + 40 * fy * fy * fy;
                const double v1 = 120 + 30 * fx - 60 * fy + 40 * fx * fy;
                const double v2 = 60 + 100 * fy * fy - 20 * fx;
                const double vs[3] = {v0, v1, v2};
                for (int c = 0; c < 3; ++c) truth[4 * (y * w + x) + c] = (uint8_t)std::lround(std::min(255.0, std::max(0.0, vs[c])));
                truth[4 * (y * w + x) + 3] = 255;
            }
        for (int y = 20; y < 52; ++y) for (int x = 30; x < 70; ++x) hole[y * w + x] = 1;   // 40 x 32 interior hole
        img = truth;
        for (int i = 0; i < w * h; ++i) if (hole[i]) { img[4 * i] = 0; img[4 * i + 1] = 255; img[4 * i + 2] = 0; }
        auto t0 = Clock::now();
        const int rc = qidk::biharmonic_inpaint(img.data(), hole.data(), w, h, 30000000);
        double mean; int mx; errStats(img, truth, hole, mean, mx);
        CHECK(rc == 0, "biharmonic returns 0 (rc=%d, %.1f ms)", rc, msSince(t0));
        CHECK(mean < 0.7 && mx <= 3, "biharmonic cubic reproduction: mean abs err %.3f, max %d (8-bit boundary data, expect <0.7 / <=3)", mean, mx);
    }
    // --- 1b. hole touching the image border and a wide hole (exercises the column-major ordering) -------------------
    {
        const int w = 100, h = 60;
        std::vector<uint8_t> truth(4 * w * h), img, hole(w * h, 0);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const double v = 80 + 0.9 * x + 0.5 * y;
            for (int c = 0; c < 3; ++c) truth[4 * (y * w + x) + c] = (uint8_t)std::lround(v + 10 * c);
            truth[4 * (y * w + x) + 3] = 255;
        }
        for (int y = 0; y < 14; ++y) for (int x = 10; x < 90; ++x) hole[y * w + x] = 1;   // 80 x 14 at the top border
        img = truth;
        for (int i = 0; i < w * h; ++i) if (hole[i]) img[4 * i] = img[4 * i + 1] = img[4 * i + 2] = 0;
        const int rc = qidk::biharmonic_inpaint(img.data(), hole.data(), w, h, 30000000);
        double mean; int mx; errStats(img, truth, hole, mean, mx);
        CHECK(rc == 0 && mean < 2.0 && mx <= 6, "biharmonic border-touching wide hole (natural boundary condition at the image edge): rc=%d mean err %.3f max %d", rc, mean, mx);
    }
    // --- 1c. refusal on an oversize problem ---------------------------------------------------------------------------
    {
        const int w = 300, h = 300;
        std::vector<uint8_t> img(4 * w * h, 100), hole(w * h, 0);
        for (int y = 10; y < 290; ++y) for (int x = 10; x < 290; ++x) hole[y * w + x] = 1;
        CHECK(qidk::biharmonic_inpaint(img.data(), hole.data(), w, h, 30000000) == -2, "biharmonic refuses a 280x280 hole with -2");
    }
    // --- 2. Criminisi continues a periodic structure through the hole --------------------------------------------------
    {
        const int w = 128, h = 128;
        std::vector<uint8_t> truth(4 * w * h), hole(w * h, 0);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const double s = std::sin((x * 0.7 + y * 0.35) * 0.9);
            const int v = (int)std::lround(128 + 100 * s);
            truth[4 * (y * w + x) + 0] = (uint8_t)v;
            truth[4 * (y * w + x) + 1] = (uint8_t)(255 - v);
            truth[4 * (y * w + x) + 2] = (uint8_t)(60 + (x % 32));
            truth[4 * (y * w + x) + 3] = 255;
        }
        for (int y = 50; y < 80; ++y) for (int x = 48; x < 84; ++x) hole[y * w + x] = 1;
        std::vector<uint8_t> a = truth, b = truth, h1 = hole, h4 = hole;
        for (int i = 0; i < w * h; ++i) if (hole[i]) { a[4 * i] = a[4 * i + 1] = a[4 * i + 2] = 0; b[4 * i] = b[4 * i + 1] = b[4 * i + 2] = 0; }
        auto t0 = Clock::now();
        const int r1 = qidk::criminisi_inpaint(a.data(), h1.data(), w, h, 4, 80, 1, 0);
        const double ms1 = msSince(t0);
        t0 = Clock::now();
        const int r4 = qidk::criminisi_inpaint(b.data(), h4.data(), w, h, 4, 80, 4, 0);
        const double ms4 = msSince(t0);
        double mean; int mx; errStats(a, truth, hole, mean, mx);
        CHECK(r1 == 0 && r4 == 0, "criminisi fills everything (1 thread %.1f ms, 4 threads %.1f ms)", ms1, ms4);
        CHECK(mean < 6.0, "criminisi periodic structure: mean abs err %.2f max %d (expect < 6)", mean, mx);
        CHECK(a == b, "criminisi result identical for 1 and 4 threads");
        bool untouched = true;
        for (int i = 0; i < w * h; ++i) if (!hole[i] && std::memcmp(&a[4 * i], &truth[4 * i], 4) != 0) untouched = false;
        CHECK(untouched, "criminisi leaves known pixels untouched");
    }
    // --- 3. budget stop returns the remaining count and keeps it in the hole mask --------------------------------------
    {
        const int w = 160, h = 160;
        std::vector<uint8_t> img(4 * w * h), hole(w * h, 0);
        for (int i = 0; i < w * h; ++i) { img[4 * i] = (uint8_t)(i * 7); img[4 * i + 1] = (uint8_t)(i * 13); img[4 * i + 2] = (uint8_t)(i * 29); img[4 * i + 3] = 255; }
        for (int y = 30; y < 130; ++y) for (int x = 30; x < 130; ++x) hole[y * w + x] = 1;
        const int rc = qidk::criminisi_inpaint(img.data(), hole.data(), w, h, 4, 100, 2, 1);   // 1 ms budget
        int left = 0; for (int i = 0; i < w * h; ++i) left += hole[i] ? 1 : 0;
        CHECK(rc > 0 && rc == left, "criminisi budget stop: rc=%d, mask holds %d", rc, left);
    }
    // --- 4. invalid input ----------------------------------------------------------------------------------------------
    {
        std::vector<uint8_t> img(4 * 16 * 16, 0), hole(16 * 16, 1);
        CHECK(qidk::criminisi_inpaint(img.data(), hole.data(), 16, 16, 4, 20, 1, 0) == -1, "criminisi: all-hole image -> -1");
        CHECK(qidk::biharmonic_inpaint(img.data(), hole.data(), 16, 16, 1000000) == -1, "biharmonic: all-hole image -> -1");
    }
}

int main(int argc, char** argv) {
    if (argc >= 7 && (std::string(argv[1]) == "crim" || std::string(argv[1]) == "bih")) {
        auto img = readAll(argv[2]);
        auto mask = readAll(argv[3]);
        const int w = std::atoi(argv[4]), h = std::atoi(argv[5]);
        if ((int)img.size() != w * h * 4 || (int)mask.size() != w * h) { std::printf("bad sizes\n"); return 2; }
        auto t0 = Clock::now();
        int rc;
        if (std::string(argv[1]) == "crim") {
            const int pr = argc > 7 ? std::atoi(argv[7]) : 4, sr = argc > 8 ? std::atoi(argv[8]) : 96;
            const int th = argc > 9 ? std::atoi(argv[9]) : 4, bud = argc > 10 ? std::atoi(argv[10]) : 0;
            rc = qidk::criminisi_inpaint(img.data(), mask.data(), w, h, pr, sr, th, bud);
        } else {
            rc = qidk::biharmonic_inpaint(img.data(), mask.data(), w, h, 30000000);
        }
        std::printf("%s rc=%d %.1f ms\n", argv[1], rc, msSince(t0));
        std::ofstream(argv[6], std::ios::binary).write(reinterpret_cast<const char*>(img.data()), (std::streamsize)img.size());
        return rc < 0 ? 1 : 0;
    }
    selfTests();
    std::printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nALL CHECKS PASSED\n", fails);
    return fails ? 1 : 0;
}
