#!/usr/bin/env python3
"""compare_images.py A.png B.png [--max-abs N] [--min-psnr DB]

Prints size, max abs difference, mean abs difference and PSNR between two images and exits 1 when a limit is broken.
Used for: port parity (e.g. Tarsh reconstructor: --max-abs 1), regression checks, "did the prompt change the output".
"""
import argparse, sys
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--max-abs", type=float); p.add_argument("--min-psnr", type=float)
    p.add_argument("--min-diff", type=float, help="require mean abs diff >= this (e.g. 'different prompts must differ')")
    a = p.parse_args()
    x = cv2.imread(a.a, cv2.IMREAD_COLOR); y = cv2.imread(a.b, cv2.IMREAD_COLOR)
    if x is None or y is None:
        print("cannot read an input image"); return 2
    if x.shape != y.shape:
        print(f"FAIL size differs: {x.shape} vs {y.shape}"); return 1
    d = np.abs(x.astype(np.int16) - y.astype(np.int16))
    mx, mean = int(d.max()), float(d.mean())
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    print(f"size={x.shape[1]}x{x.shape[0]} max_abs={mx} mean_abs={mean:.4f} psnr={psnr:.2f} dB identical={mx == 0}")
    bad = False
    if a.max_abs is not None and mx > a.max_abs: print(f"FAIL max_abs {mx} > {a.max_abs}"); bad = True
    if a.min_psnr is not None and psnr < a.min_psnr: print(f"FAIL psnr {psnr:.2f} < {a.min_psnr}"); bad = True
    if a.min_diff is not None and mean < a.min_diff: print(f"FAIL mean_abs {mean:.4f} < {a.min_diff}"); bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
