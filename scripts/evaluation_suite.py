#!/usr/bin/env python3
"""
Comprehensive Perceptual Metric Evaluation Suite for Inpainting Models.

Evaluates inpainting predictions against ground-truth images using:
  - PSNR (Peak Signal-to-Noise Ratio in dB)
  - SSIM (Structural Similarity Index Measure)
  - LPIPS (Learned Perceptual Image Patch Similarity with VGG backbone)

Features:
  - Strict {idx}_{model}.png file naming support
  - Clean offset handling: auto-detects 0-indexed vs 1-indexed predictions
  - Aspect ratio / resolution harmonization (Bicubic matching to GT)
  - Formatted CSV export with per-pair metrics and dataset averages
  - Multi-model cross-comparison report support
"""

import os
import re
import csv
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
import lpips
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def extract_integer_index(filename):
    """Extracts integer index from filenames like '1.png', '0_migan.png'."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    # Check leading integer (e.g. '0_migan' -> 0, '1' -> 1)
    match = re.match(r"^(\d+)", stem)
    if match:
        return int(match.group(1))
    # Fallback to any integer
    search = re.search(r"\d+", stem)
    return int(search.group(0)) if search else None


def infer_model_from_path_or_files(pred_dir):
    """Infers model name from directory path or directory contents."""
    path_lower = os.path.abspath(pred_dir).lower()
    for m in ["sd", "migan", "aotgan", "lama"]:
        if f"/{m}/" in path_lower or path_lower.endswith(f"/{m}") or path_lower.endswith(f"/{m}/results"):
            return m

    try:
        files = os.listdir(pred_dir)
        for m in ["sd", "migan", "aotgan", "lama"]:
            if any(f.lower().endswith(f"_{m}.png") for f in files):
                return m
    except Exception:
        pass
    return "unknown"


def auto_detect_offset(gt_indices, pred_dir, model=None):
    """
    Detects whether prediction files are offset-aligned:
      offset 1: GT 1 matches Pred 0 (e.g., 0_{model}.png)
      offset 0: GT 1 matches Pred 1 (e.g., 1_{model}.png)
    """
    if not os.path.isdir(pred_dir):
        return 0

    pred_files = os.listdir(pred_dir)

    # Count matches under offset=1 vs offset=0
    count_offset_0 = 0
    count_offset_1 = 0

    for gt_idx in gt_indices:
        # Check offset 0 (gt_idx)
        idx0 = str(gt_idx)
        pattern0 = f"{idx0}_{model}.png" if model else f"{idx0}."
        if any(f.lower().startswith(f"{idx0}_") or f.lower().startswith(f"{idx0}.") for f in pred_files):
            count_offset_0 += 1

        # Check offset 1 (gt_idx - 1)
        idx1 = str(gt_idx - 1)
        if any(f.lower().startswith(f"{idx1}_") or f.lower().startswith(f"{idx1}.") for f in pred_files):
            count_offset_1 += 1

    if count_offset_1 > count_offset_0:
        return 1
    elif count_offset_0 > 0:
        return 0
    else:
        # Fallback default
        return 1


def find_matching_pred_file(pred_dir, gt_index, model=None, offset=1):
    """
    Strictly locates the prediction file for ground truth index:
      First checks: target_idx = gt_index - offset
      Fallback checks: target_idx = gt_index (if different)
    """
    valid_exts = VALID_IMAGE_EXTS
    cand_indices = [gt_index - offset]
    if offset != 0:
        cand_indices.append(gt_index)

    model_candidates = [model.lower()] if model else ["sd", "migan", "aotgan", "lama"]

    # 1. Exact match with model suffix: {idx}_{model}.ext
    for p_idx in cand_indices:
        for m in model_candidates:
            for ext in valid_exts:
                path = os.path.join(pred_dir, f"{p_idx}_{m}{ext}")
                if os.path.isfile(path):
                    return path

    # 2. Match exact index without model suffix: {idx}.ext
    for p_idx in cand_indices:
        for ext in valid_exts:
            path = os.path.join(pred_dir, f"{p_idx}{ext}")
            if os.path.isfile(path):
                return path

    return None


class PerceptualEvaluator:
    """Evaluates images using PSNR, SSIM, and LPIPS (VGG backbone)."""

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"📦 Initializing LPIPS with VGG backbone on device: {self.device}")
        # Initialize LPIPS VGG model
        self.lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        self.lpips_vgg.eval()

    def evaluate_pair(self, gt_path, pred_path):
        """
        Loads GT and Prediction images, harmonizes resolution if necessary,
        and computes PSNR, SSIM, and LPIPS.
        """
        gt_pil = Image.open(gt_path).convert("RGB")
        pred_pil = Image.open(pred_path).convert("RGB")

        # Harmonize resolution to GT resolution via Bicubic if they differ
        if gt_pil.size != pred_pil.size:
            pred_pil = pred_pil.resize(gt_pil.size, Image.Resampling.BICUBIC)

        gt_np = np.array(gt_pil).astype(np.float32) / 255.0
        pred_np = np.array(pred_pil).astype(np.float32) / 255.0

        # 1. PSNR (data_range=1.0)
        psnr_val = float(compute_psnr(gt_np, pred_np, data_range=1.0))

        # 2. SSIM (data_range=1.0, channel_axis=2)
        ssim_val = float(compute_ssim(gt_np, pred_np, data_range=1.0, channel_axis=2))

        # 3. LPIPS (VGG backbone expects tensors in [-1.0, 1.0] shape (1, 3, H, W))
        gt_tensor = (TF.to_tensor(gt_pil).unsqueeze(0).to(self.device) * 2.0) - 1.0
        pred_tensor = (TF.to_tensor(pred_pil).unsqueeze(0).to(self.device) * 2.0) - 1.0

        with torch.no_grad():
            lpips_val = float(self.lpips_vgg(gt_tensor, pred_tensor).item())

        return psnr_val, ssim_val, lpips_val


def run_evaluation(
    gt_dir="Benchmark/input/image",
    pred_dir="Benchmark/output/sd/results",
    model=None,
    offset="auto",
    output_csv=None,
    device=None
):
    """Runs batch perceptual metric evaluation and exports a CSV report."""
    if not os.path.isdir(gt_dir):
        print(f"❌ Error: Ground truth directory not found: {gt_dir}")
        return None

    if not os.path.isdir(pred_dir):
        print(f"❌ Error: Prediction directory not found: {pred_dir}")
        return None

    # Discover GT files
    gt_files = [
        os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.lower().endswith(VALID_IMAGE_EXTS)
    ]
    gt_files.sort(key=lambda x: extract_integer_index(x) or 0)

    if not gt_files:
        print(f"❌ No ground truth images found in {gt_dir}")
        return None

    gt_indices = [extract_integer_index(f) for f in gt_files if extract_integer_index(f) is not None]

    # Infer model if not specified
    if not model or model == "auto":
        model = infer_model_from_path_or_files(pred_dir)

    # Resolve offset
    if str(offset).lower() == "auto":
        detected_offset = auto_detect_offset(gt_indices, pred_dir, model=model)
        resolved_offset = detected_offset
    else:
        resolved_offset = int(offset)

    if output_csv is None:
        output_csv = f"{model}_metrics.csv" if model != "unknown" else "metrics_summary.csv"

    print("=" * 68)
    print("        COMPREHENSIVE PERCEPTUAL METRIC EVALUATION SUITE        ")
    print("=" * 68)
    print(f"  Ground Truth Dir : {gt_dir} ({len(gt_files)} images)")
    print(f"  Prediction Dir   : {pred_dir}")
    print(f"  Target Model     : {model.upper()}")
    print(f"  Resolved Offset  : {resolved_offset} (Pred = GT - {resolved_offset})")
    print(f"  CSV Report       : {output_csv}")
    print("=" * 68 + "\n")

    evaluator = PerceptualEvaluator(device=device)

    records = []
    psnr_list = []
    ssim_list = []
    lpips_list = []

    print(f"{'Idx':<5} | {'GT File':<12} | {'Pred File':<16} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'LPIPS (VGG)':<11}")
    print("-" * 72)

    for gt_path in gt_files:
        idx = extract_integer_index(gt_path)
        if idx is None:
            continue

        pred_path = find_matching_pred_file(pred_dir, idx, model=model, offset=resolved_offset)
        if not pred_path:
            expected_name = f"{idx - resolved_offset}_{model}.png"
            print(f"{idx:<5} | {os.path.basename(gt_path):<12} | [MISSING: {expected_name}]")
            continue

        try:
            psnr_val, ssim_val, lpips_val = evaluator.evaluate_pair(gt_path, pred_path)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)

            records.append({
                "Index": idx,
                "GT_File": os.path.basename(gt_path),
                "Pred_File": os.path.basename(pred_path),
                "Model": model,
                "PSNR_dB": round(psnr_val, 4),
                "SSIM": round(ssim_val, 4),
                "LPIPS_VGG": round(lpips_val, 4)
            })

            gt_name = os.path.basename(gt_path)
            pred_name = os.path.basename(pred_path)
            print(f"{idx:<5} | {gt_name:<12} | {pred_name:<16} | {psnr_val:>9.4f}  | {ssim_val:>7.4f} | {lpips_val:>10.4f}")

        except Exception as e:
            print(f"{idx:<5} | {os.path.basename(gt_path):<12} | ERROR: {e}")

    if not records:
        print("\n❌ No valid image pairs evaluated. Check prediction directory and naming conventions.")
        return None

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_lpips = float(np.mean(lpips_list))

    print("-" * 72)
    print(f"{'AVG':<5} | {f'{len(records)} pairs':<12} | {'-':<16} | {avg_psnr:>9.4f}  | {avg_ssim:>7.4f} | {avg_lpips:>10.4f}")
    print("=" * 72 + "\n")

    # Write CSV report
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        fields = ["Index", "GT_File", "Pred_File", "Model", "PSNR_dB", "SSIM", "LPIPS_VGG"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
        writer.writerow({
            "Index": "AVERAGE",
            "GT_File": f"{len(records)} pairs",
            "Pred_File": "-",
            "Model": model,
            "PSNR_dB": round(avg_psnr, 4),
            "SSIM": round(avg_ssim, 4),
            "LPIPS_VGG": round(avg_lpips, 4),
        })

    print(f"✅ Formatted CSV report successfully saved to: {os.path.abspath(output_csv)}\n")

    return {
        "model": model,
        "pairs_count": len(records),
        "psnr_avg": avg_psnr,
        "ssim_avg": avg_ssim,
        "lpips_avg": avg_lpips,
        "csv_path": os.path.abspath(output_csv)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive Perceptual Metric Evaluation Suite (PSNR, SSIM, LPIPS VGG)."
    )
    parser.add_argument("--gt_dir", type=str, default="Benchmark/input/image",
                        help="Path to ground-truth images (default: Benchmark/input/image)")
    parser.add_argument("--pred_dir", type=str, default="Benchmark/output/sd/results",
                        help="Path to prediction images (default: Benchmark/output/sd/results)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model name to match predictions (e.g. sd, migan, aotgan, lama)")
    parser.add_argument("--offset", type=str, default="auto",
                        help="Index offset between GT and Pred ('auto', 0, 1). Default: auto")
    parser.add_argument("--output_csv", "-o", type=str, default=None,
                        help="Path to save output CSV summary report")
    parser.add_argument("--device", type=str, default=None,
                        help="Evaluation device ('cuda' or 'cpu')")

    args = parser.parse_args()

    run_evaluation(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        model=args.model,
        offset=args.offset,
        output_csv=args.output_csv,
        device=args.device
    )
