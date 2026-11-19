// biharmonic.cpp - biharmonic inpainting by banded Cholesky on B = L^2 (L = 4-neighbour graph Laplacian).
// In the interior B is the classic 13-point stencil; at the image border the graph Laplacian gives natural
// (replicate) boundary behaviour and keeps B symmetric positive semi-definite, so B restricted to the hole
// (a proper subset of the image) is symmetric positive definite and the Cholesky factorisation always exists.
#include "classical.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <vector>

namespace qidk {

int biharmonic_inpaint(uint8_t* rgba, const uint8_t* hole, int w, int h, int64_t max_cells) {
    if (!rgba || !hole || w < 5 || h < 5) return -1;
    const int N = w * h;
    std::vector<int32_t> idx(N, -1);
    int bx0 = w, bx1 = -1, by0 = h, by1 = -1, n = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (hole[y * w + x]) {
                bx0 = std::min(bx0, x); bx1 = std::max(bx1, x);
                by0 = std::min(by0, y); by1 = std::max(by1, y);
                ++n;
            }
    if (n == 0) return 0;
    if (n == N) return -1;

    // Number unknowns along the SHORTER side of the hole box, so the matrix band is about 2 * shorter side.
    int k = 0;
    if ((bx1 - bx0) <= (by1 - by0)) {
        for (int y = by0; y <= by1; ++y)
            for (int x = bx0; x <= bx1; ++x)
                if (hole[y * w + x]) idx[y * w + x] = k++;
    } else {
        for (int x = bx0; x <= bx1; ++x)
            for (int y = by0; y <= by1; ++y)
                if (hole[y * w + x]) idx[y * w + x] = k++;
    }

    auto inImg = [&](int x, int y) { return x >= 0 && x < w && y >= 0 && y < h; };
    auto deg = [&](int x, int y) { return (x > 0) + (x < w - 1) + (y > 0) + (y < h - 1); };
    static const int DX[4] = {1, -1, 0, 0}, DY[4] = {0, 0, 1, -1};

    // Row of B = L*L at pixel (x,y): coefficients c[dy+2][dx+2] on the 13 pixels with |dx|+|dy| <= 2.
    auto stencil = [&](int x, int y, double c[5][5]) {
        for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b) c[a][b] = 0.0;
        const int dp = deg(x, y);
        for (int a = 0; a < 4; ++a) {
            const int qx = x + DX[a], qy = y + DY[a];
            if (!inImg(qx, qy)) continue;
            const int dq = deg(qx, qy);
            for (int b = 0; b < 4; ++b) {
                const int sx = qx + DX[b], sy = qy + DY[b];
                if (inImg(sx, sy)) c[sy - y + 2][sx - x + 2] += 1.0;
            }
            c[qy - y + 2][qx - x + 2] -= static_cast<double>(dq + dp);
        }
        c[2][2] += static_cast<double>(dp) * dp;
    };

    // Half bandwidth.
    int64_t bw = 0;
    for (int y = by0; y <= by1; ++y)
        for (int x = bx0; x <= bx1; ++x) {
            const int i = idx[y * w + x];
            if (i < 0) continue;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    if (std::abs(dx) + std::abs(dy) > 2 || !inImg(x + dx, y + dy)) continue;
                    const int j = idx[(y + dy) * w + (x + dx)];
                    if (j >= 0) bw = std::max<int64_t>(bw, std::abs(i - j));
                }
        }
    const int64_t width = bw + 1;
    if (static_cast<int64_t>(n) * width > max_cells) return -2;
    if (static_cast<double>(n) * static_cast<double>(bw) * static_cast<double>(bw) > 1.2e10) return -2;

    std::vector<double> A;
    std::vector<double> rhs[3];
    try {
        A.assign(static_cast<size_t>(n) * width, 0.0);
        for (auto& v : rhs) v.assign(n, 0.0);
    } catch (const std::bad_alloc&) {
        return -3;
    }
    // A[i*width + (j - i + bw)] holds B[i][j] for i-bw <= j <= i.
    for (int y = by0; y <= by1; ++y)
        for (int x = bx0; x <= bx1; ++x) {
            const int i = idx[y * w + x];
            if (i < 0) continue;
            double c[5][5];
            stencil(x, y, c);
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx) {
                    const double v = c[dy + 2][dx + 2];
                    if (v == 0.0) continue;
                    const int sx = x + dx, sy = y + dy;
                    const int j = idx[sy * w + sx];
                    if (j >= 0) {
                        if (j <= i) A[static_cast<size_t>(i) * width + (j - i + bw)] += v;
                    } else {
                        const uint8_t* p = rgba + 4 * static_cast<size_t>(sy * w + sx);
                        for (int ch = 0; ch < 3; ++ch) rhs[ch][i] -= v * p[ch];
                    }
                }
        }

    // Banded Cholesky B = L L^T, in place.
    for (int64_t i = 0; i < n; ++i) {
        const int64_t jmin = std::max<int64_t>(0, i - bw);
        for (int64_t j = jmin; j <= i; ++j) {
            double sum = A[static_cast<size_t>(i * width + (j - i + bw))];
            const double* pi = &A[static_cast<size_t>(i * width + (jmin - i + bw))];
            const double* pj = &A[static_cast<size_t>(j * width + (jmin - j + bw))];
            for (int64_t t = jmin; t < j; ++t) sum -= (*pi++) * (*pj++);
            if (j == i) {
                if (!(sum > 1e-9)) return -3;
                A[static_cast<size_t>(i * width + bw)] = std::sqrt(sum);
            } else {
                A[static_cast<size_t>(i * width + (j - i + bw))] = sum / A[static_cast<size_t>(j * width + bw)];
            }
        }
    }

    // Solve for the three colour channels with the same factor.
    std::vector<double> y(n), x(n);
    for (int ch = 0; ch < 3; ++ch) {
        for (int64_t i = 0; i < n; ++i) {
            double s = rhs[ch][i];
            const int64_t jmin = std::max<int64_t>(0, i - bw);
            const double* row = &A[static_cast<size_t>(i * width + (jmin - i + bw))];
            for (int64_t t = jmin; t < i; ++t) s -= (*row++) * y[t];
            y[i] = s / A[static_cast<size_t>(i * width + bw)];
        }
        for (int64_t i = n - 1; i >= 0; --i) {
            double s = y[i];
            const int64_t jmax = std::min<int64_t>(n - 1, i + bw);
            for (int64_t t = i + 1; t <= jmax; ++t) s -= A[static_cast<size_t>(t * width + (i - t + bw))] * x[t];
            x[i] = s / A[static_cast<size_t>(i * width + bw)];
        }
        for (int yy = by0; yy <= by1; ++yy)
            for (int xx = bx0; xx <= bx1; ++xx) {
                const int i = idx[yy * w + xx];
                if (i < 0) continue;
                const double v = std::min(255.0, std::max(0.0, std::floor(x[i] + 0.5)));
                rgba[4 * static_cast<size_t>(yy * w + xx) + ch] = static_cast<uint8_t>(v);
            }
    }
    for (int yy = by0; yy <= by1; ++yy)
        for (int xx = bx0; xx <= bx1; ++xx)
            if (idx[yy * w + xx] >= 0) rgba[4 * static_cast<size_t>(yy * w + xx) + 3] = 255;
    return 0;
}

}  // namespace qidk
