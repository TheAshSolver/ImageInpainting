#!/usr/bin/env python3
"""Objective checks of one super-resolution result against its input (needs numpy + opencv-python).

  sr_metrics.py --input in.png --output out.png --scale 2 [--json]

Prints one JSON object and exits 0 (PASS) or 1 (FAIL). The numbers are proxies, not a judgement of beauty: they catch the four
failures seen in the field (identity output, screen-door lattice, geometric drift, garbage) and report how much detail was added.

  lr_psnr            PSNR between the input and the output box-averaged back down (the output must still BE this picture)
  lr_psnr_bicubic    the same number for plain bicubic: the reference a neural result has to stay close to
  nyq_share          share of the diagonal 2x2 Hadamard energy in (output - bicubic): 1/3 for natural texture, ~1 for a pixel lattice
  mae_vs_bicubic     mean |output - bicubic| in 8-bit levels: ~0 = the engine added nothing, > 25 = it changed the picture
  lap_gain           Laplacian variance of the output / of bicubic (> 1 = sharper); informational
  periodic_peak      strongest isolated peak of the spectrum of (output - bicubic) relative to its median; > 60 = periodic mesh
"""
import argparse
import json
import sys

import cv2
import numpy as np


def box_down(img, s):
    h, w = img.shape[:2]
    h2, w2 = h // s, w // s
    return img[: h2 * s, : w2 * s].astype(np.float64).reshape(h2, s, w2, s, -1).mean(axis=(1, 3))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)


def hadamard_shares(res):
    g = res.mean(axis=2) if res.ndim == 3 else res
    h, w = (g.shape[0] // 2) * 2, (g.shape[1] // 2) * 2
    g = g[:h, :w]
    a, b, c, d = g[0::2, 0::2], g[0::2, 1::2], g[1::2, 0::2], g[1::2, 1::2]
    hx = 0.25 * (a - b + c - d)
    hy = 0.25 * (a + b - c - d)
    hxy = 0.25 * (a - b - c + d)
    tot = float((hx ** 2).sum() + (hy ** 2).sum() + (hxy ** 2).sum())
    return float((hxy ** 2).sum() / tot) if tot > 1e-9 else 0.0


def periodic_peak(res):
    g = res.mean(axis=2) if res.ndim == 3 else res
    n = 1024
    cy, cx = g.shape[0] // 2, g.shape[1] // 2
    g = g[max(0, cy - n // 2): cy + n // 2, max(0, cx - n // 2): cx + n // 2]
    if min(g.shape) < 64:
        return 0.0
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    f = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win)))
    f[g.shape[0] // 2 - 3: g.shape[0] // 2 + 4, g.shape[1] // 2 - 3: g.shape[1] // 2 + 4] = 0     # DC and its leakage
    med = float(np.median(f)) + 1e-9
    return float(f.max() / med)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    src = cv2.imread(a.input, cv2.IMREAD_COLOR)
    out = cv2.imread(a.output, cv2.IMREAD_COLOR)
    if src is None or out is None:
        print(json.dumps({"verdict": "FAIL", "reason": "cannot read images"}))
        return 1
    s = a.scale
    r = {"input": list(src.shape[:2]), "output": list(out.shape[:2]), "scale": s}
    if out.shape[0] != src.shape[0] * s or out.shape[1] != src.shape[1] * s:
        r.update(verdict="FAIL", reason="output size is not input x scale")
        print(json.dumps(r))
        return 1
    bic = cv2.resize(src, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC)
    r["lr_psnr"] = round(psnr(box_down(out, s), src), 2)
    r["lr_psnr_bicubic"] = round(psnr(box_down(bic, s), src), 2)
    res = out.astype(np.float32) - bic.astype(np.float32)
    r["mae_vs_bicubic"] = round(float(np.abs(res).mean()), 3)
    r["nyq_share"] = round(hadamard_shares(res), 3)
    lap = lambda im: float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    r["lap_gain"] = round(lap(out) / max(lap(bic), 1e-9), 3)
    r["periodic_peak"] = round(periodic_peak(res), 1)
    r["detail_added"] = r["mae_vs_bicubic"] >= 0.3
    problems = []
    if r["lr_psnr"] < r["lr_psnr_bicubic"] - 3.0 and r["lr_psnr"] < 38.0:      # a smooth picture makes bicubic look perfect; 38 dB (3 levels rms) is still the same picture
        problems.append("output drifted from the input (LR consistency %.1f dB vs bicubic %.1f dB)" % (r["lr_psnr"], r["lr_psnr_bicubic"]))
    if r["nyq_share"] > 0.55:
        problems.append("pixel-lattice texture (diagonal share %.2f)" % r["nyq_share"])
    if r["periodic_peak"] > 60 and r["mae_vs_bicubic"] > 1.0:
        problems.append("periodic mesh in the spectrum (peak %.0fx median)" % r["periodic_peak"])
    if r["mae_vs_bicubic"] > 25:
        problems.append("the picture was changed, not upscaled (%.1f levels from bicubic)" % r["mae_vs_bicubic"])
    r["verdict"] = "FAIL" if problems else "PASS"
    r["problems"] = problems
    print(json.dumps(r))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
