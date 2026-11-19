// criminisi.cpp - exemplar-based inpainting, priority P(p) = C(p) * D(p), SSD patch matching.
#include "classical.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

namespace qidk {
namespace {

struct Candidate {
    int64_t ssd = INT64_MAX;
    int x = -1, y = -1;
};

// Strict total order: lower SSD wins, then lower y, then lower x. Makes the result independent of thread count.
inline bool better(const Candidate& a, const Candidate& b) {
    if (a.ssd != b.ssd) return a.ssd < b.ssd;
    if (a.y != b.y) return a.y < b.y;
    return a.x < b.x;
}

}  // namespace

int criminisi_inpaint(uint8_t* rgba, uint8_t* holeIO, int w, int h,
                      int patch_radius, int search_radius, int threads, int budget_ms) {
    if (!rgba || !holeIO || w < 8 || h < 8) return -1;
    const int r = std::clamp(patch_radius, 2, 12);
    const int S = std::clamp(search_radius, 8, std::max(w, h));
    const int T = std::clamp(threads, 1, 8);
    const int N = w * h;

    std::vector<uint8_t> rgb(static_cast<size_t>(N) * 3);
    std::vector<uint8_t> hole(N);
    std::vector<float> gray(N), conf(N);
    std::vector<int> rowCount(h, 0);
    int remaining = 0, minX = w, maxX = -1;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            const uint8_t* p = rgba + 4 * static_cast<size_t>(i);
            rgb[3 * i] = p[0]; rgb[3 * i + 1] = p[1]; rgb[3 * i + 2] = p[2];
            gray[i] = (0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2]) / 255.0f;
            hole[i] = holeIO[i] ? 1 : 0;
            conf[i] = hole[i] ? 0.0f : 1.0f;
            if (hole[i]) {
                ++remaining; ++rowCount[y];
                minX = std::min(minX, x); maxX = std::max(maxX, x);
            }
        }
    }
    if (remaining == 0) return 0;
    if (remaining == N) return -1;
    const int initialHole = remaining;
    int y0 = 0, y1 = h - 1;
    const int x0 = minX, x1 = maxX;

    // Source patches must be fully inside the image and contain no ORIGINAL hole pixel.
    std::vector<int32_t> integ(static_cast<size_t>(w + 1) * (h + 1), 0);
    for (int y = 0; y < h; ++y) {
        int32_t run = 0;
        for (int x = 0; x < w; ++x) {
            run += hole[y * w + x];
            integ[static_cast<size_t>(y + 1) * (w + 1) + (x + 1)] = integ[static_cast<size_t>(y) * (w + 1) + (x + 1)] + run;
        }
    }
    std::vector<uint8_t> validCenter(N, 0);
    int validCount = 0;
    for (int y = r; y < h - r; ++y) {
        for (int x = r; x < w - r; ++x) {
            const int32_t s = integ[static_cast<size_t>(y + r + 1) * (w + 1) + (x + r + 1)]
                            - integ[static_cast<size_t>(y - r) * (w + 1) + (x + r + 1)]
                            - integ[static_cast<size_t>(y + r + 1) * (w + 1) + (x - r)]
                            + integ[static_cast<size_t>(y - r) * (w + 1) + (x - r)];
            if (s == 0) { validCenter[y * w + x] = 1; ++validCount; }
        }
    }
    if (validCount == 0) return -2;

    const int stride = initialHole > 20000 ? 2 : 1;
    const auto tStart = std::chrono::steady_clock::now();
    std::vector<int> offs;
    std::vector<uint8_t> tgt;

    auto M = [&](int xx, int yy) {  // known map with clamped coordinates
        xx = std::clamp(xx, 0, w - 1); yy = std::clamp(yy, 0, h - 1);
        return hole[yy * w + xx] ? 0.0f : 1.0f;
    };

    while (remaining > 0) {
        if (budget_ms > 0 &&
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tStart).count() > budget_ms)
            break;
        while (y0 <= y1 && rowCount[y0] == 0) ++y0;
        while (y1 >= y0 && rowCount[y1] == 0) --y1;

        // 1. Priorities along the fill front.
        float bestP = -1.0f, bestC = 0.0f;
        int bx = -1, by = -1;
        for (int y = y0; y <= y1; ++y) {
            if (rowCount[y] == 0) continue;
            for (int x = x0; x <= x1; ++x) {
                const int i = y * w + x;
                if (!hole[i]) continue;
                const bool front = (x > 0 && !hole[i - 1]) || (x < w - 1 && !hole[i + 1]) ||
                                   (y > 0 && !hole[i - w]) || (y < h - 1 && !hole[i + w]);
                if (!front) continue;

                float csum = 0.0f; int cnt = 0;
                float bestMag = 0.0f, gxBest = 0.0f, gyBest = 0.0f;
                for (int dy = -r; dy <= r; ++dy) {
                    const int py = y + dy;
                    if (py < 0 || py >= h) continue;
                    for (int dx = -r; dx <= r; ++dx) {
                        const int px = x + dx;
                        if (px < 0 || px >= w) continue;
                        const int q = py * w + px;
                        csum += conf[q]; ++cnt;
                        if (hole[q] || px < 1 || px >= w - 1 || py < 1 || py >= h - 1) continue;
                        if (hole[q - 1] || hole[q + 1] || hole[q - w] || hole[q + w]) continue;
                        const float gx = 0.5f * (gray[q + 1] - gray[q - 1]);
                        const float gy = 0.5f * (gray[q + w] - gray[q - w]);
                        const float mag = gx * gx + gy * gy;
                        if (mag > bestMag) { bestMag = mag; gxBest = gx; gyBest = gy; }
                    }
                }
                const float C = csum / static_cast<float>(cnt);
                float nx = (M(x + 1, y - 1) + 2 * M(x + 1, y) + M(x + 1, y + 1)) - (M(x - 1, y - 1) + 2 * M(x - 1, y) + M(x - 1, y + 1));
                float ny = (M(x - 1, y + 1) + 2 * M(x, y + 1) + M(x + 1, y + 1)) - (M(x - 1, y - 1) + 2 * M(x, y - 1) + M(x + 1, y - 1));
                const float nl = std::sqrt(nx * nx + ny * ny);
                float D = 0.0f;
                if (nl > 1e-6f) D = std::fabs((-gyBest) * (nx / nl) + gxBest * (ny / nl));  // |isophote . normal|
                const float P = C * (D + 1e-3f);
                if (P > bestP) { bestP = P; bestC = C; bx = x; by = y; }
            }
        }
        if (bx < 0) break;  // no front pixel: cannot happen while a known pixel exists

        // 2. Known pixels of the target patch.
        offs.clear(); tgt.clear();
        for (int dy = -r; dy <= r; ++dy) {
            const int py = by + dy;
            if (py < 0 || py >= h) continue;
            for (int dx = -r; dx <= r; ++dx) {
                const int px = bx + dx;
                if (px < 0 || px >= w) continue;
                const int q = py * w + px;
                if (hole[q]) continue;
                offs.push_back(dy * w + dx);
                tgt.push_back(rgb[3 * q]); tgt.push_back(rgb[3 * q + 1]); tgt.push_back(rgb[3 * q + 2]);
            }
        }
        const size_t K = offs.size();

        // 3. Best exemplar by SSD over the known pixels (early exit once a candidate cannot win).
        auto searchRows = [&](int cy1, int cx0, int cx1, int st, int firstRow, int rowStep, Candidate& out) {
            Candidate best;
            for (int cy = firstRow; cy <= cy1; cy += rowStep) {
                for (int cx = cx0; cx <= cx1; cx += st) {
                    const int ci = cy * w + cx;
                    if (!validCenter[ci]) continue;
                    int64_t ssd = 0;
                    for (size_t k = 0; k < K; ++k) {
                        const uint8_t* q = &rgb[3 * static_cast<size_t>(ci + offs[k])];
                        const int d0 = int(q[0]) - tgt[3 * k], d1 = int(q[1]) - tgt[3 * k + 1], d2 = int(q[2]) - tgt[3 * k + 2];
                        ssd += d0 * d0 + d1 * d1 + d2 * d2;
                        if (ssd > best.ssd) break;
                    }
                    Candidate c; c.ssd = ssd; c.x = cx; c.y = cy;
                    if (ssd <= best.ssd && better(c, best)) best = c;
                }
            }
            out = best;
        };
        auto runSearch = [&](int cx0, int cx1, int cy0, int cy1, int st) {
            std::vector<Candidate> res(T);
            const int rowStep = st * T;
            if (T == 1 || static_cast<long>(cx1 - cx0 + 1) * (cy1 - cy0 + 1) < 16384) {
                Candidate best;
                for (int t = 0; t < T; ++t) {
                    Candidate c;
                    searchRows(cy1, cx0, cx1, st, cy0 + t * st, rowStep, c);
                    if (better(c, best)) best = c;
                }
                return best;
            }
            std::vector<std::thread> pool;
            for (int t = 0; t < T; ++t)
                pool.emplace_back([&, t] { searchRows(cy1, cx0, cx1, st, cy0 + t * st, rowStep, res[t]); });
            for (auto& th : pool) th.join();
            Candidate best;
            for (int t = 0; t < T; ++t) if (better(res[t], best)) best = res[t];
            return best;
        };
        Candidate best = runSearch(std::max(r, bx - S), std::min(w - 1 - r, bx + S),
                                   std::max(r, by - S), std::min(h - 1 - r, by + S), stride);
        if (best.x < 0) best = runSearch(r, w - 1 - r, r, h - 1 - r, 1);
        if (best.x < 0) return -2;

        // 4. Copy the exemplar into the unknown pixels of the target patch and propagate the confidence.
        for (int dy = -r; dy <= r; ++dy) {
            const int py = by + dy;
            if (py < 0 || py >= h) continue;
            for (int dx = -r; dx <= r; ++dx) {
                const int px = bx + dx;
                if (px < 0 || px >= w) continue;
                const int q = py * w + px;
                if (!hole[q]) continue;
                const int s = (best.y + dy) * w + (best.x + dx);
                rgb[3 * q] = rgb[3 * s]; rgb[3 * q + 1] = rgb[3 * s + 1]; rgb[3 * q + 2] = rgb[3 * s + 2];
                gray[q] = gray[s];
                conf[q] = bestC;
                hole[q] = 0;
                --rowCount[py];
                --remaining;
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        if (holeIO[i] && !hole[i]) {
            uint8_t* p = rgba + 4 * static_cast<size_t>(i);
            p[0] = rgb[3 * i]; p[1] = rgb[3 * i + 1]; p[2] = rgb[3 * i + 2]; p[3] = 255;
        }
        holeIO[i] = hole[i];
    }
    return remaining;
}

}  // namespace qidk
