#!/usr/bin/env python3
"""
scripts/benchmarks/evaluate_6000_quality.py
================================================================================
⚡ 6,000-IMAGE BATCH PERCEPTUAL QUALITY PIPELINE (PSNR, SSIM, LPIPS)
================================================================================
Calculates rigorous, standardized image restoration metrics for the 6,000-image
dataset across inpainting predictions (MIGAN, LaMa-Dilated, AOT-GAN, Stable Diffusion).

Evaluated Metrics:
  1. psnr_full  : Full-canvas Peak Signal-to-Noise Ratio (dB)
  2. psnr_hole  : Hole-only PSNR (evaluating strictly masked pixels) (dB)
  3. ssim       : Structural Similarity Index Measure (0.0 to 1.0)
  4. lpips_vgg  : Learned Perceptual Image Patch Similarity (VGG-16 backbone)

Pipeline Highlights:
  - Alpha Compositing: I_comp = M * I_pred + (1 - M) * I_gt
  - Automatic Resolution & Offset Matching (handles {idx}_{model}.png or {idx:04d}.png)
  - Seamless Metadata Fusion with dataset_manifest.csv and masks_manifest.csv
  - High-Throughput Batched PyTorch LPIPS Evaluation
  - Resilient Checkpointing: Automatically resumes if interrupted
================================================================================
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF
import lpips
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ==============================================================================
# DEFAULT PATHS
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES_DIR = REPO_ROOT / "Benchmark" / "6000_images" / "compiled" / "images"
DEFAULT_MASKS_DIR = REPO_ROOT / "Benchmark" / "6000_images" / "compiled" / "masks"
DEFAULT_IMG_MANIFEST = REPO_ROOT / "Benchmark" / "6000_images" / "compiled" / "dataset_manifest.csv"
DEFAULT_MASK_MANIFEST = REPO_ROOT / "Benchmark" / "6000_images" / "compiled" / "masks_manifest.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Benchmark" / "output" / "6000_benchmark"


class QualityEvaluator:
    """Computes PSNR (Full & Hole), SSIM, and batched LPIPS with VGG backbone."""

    def __init__(self, device: Optional[str] = None, lpips_batch_size: int = 16):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.batch_size = lpips_batch_size
        print(f"📦 Initializing Perceptual Evaluator on device: {self.device}")
        print("   Loading LPIPS VGG-16 network weights...")
        self.lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        self.lpips_fn.eval()

    def compute_psnr_and_ssim(self, gt_rgb: np.ndarray, pred_rgb: np.ndarray,
                              mask_bin: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes PSNR Full, PSNR Hole, and SSIM.
        Input arrays are uint8 in range [0, 255] with shape (512, 512, 3).
        mask_bin is float32/uint8 with shape (512, 512) where hole=1.
        """
        # Alpha compositing: replace unmasked pixels with exact ground truth
        mask_3d = (mask_bin > 0.5)[:, :, None].astype(np.float32)
        comp_rgb = np.clip(mask_3d * pred_rgb + (1.0 - mask_3d) * gt_rgb, 0, 255).astype(np.uint8)

        # 1. Full Image PSNR
        psnr_full = float(compute_psnr(gt_rgb, comp_rgb, data_range=255))

        # 2. Hole-Only PSNR (strictly inside masked defect)
        hole_mask_2d = (mask_bin > 0.5)
        if np.any(hole_mask_2d):
            pred_hole = pred_rgb[hole_mask_2d].astype(np.float64)
            gt_hole = gt_rgb[hole_mask_2d].astype(np.float64)
            mse_hole = np.mean((pred_hole - gt_hole) ** 2)
            psnr_hole = float(10.0 * np.log10((255.0 ** 2) / mse_hole)) if mse_hole > 1e-10 else 99.0
        else:
            psnr_hole = psnr_full

        # 3. SSIM (channel_axis=2 for RGB)
        ssim_val = float(compute_ssim(gt_rgb, comp_rgb, data_range=255, channel_axis=2))

        return round(psnr_full, 4), round(psnr_hole, 4), round(ssim_val, 4)

    def compute_lpips_batch(self, gt_list: List[np.ndarray], comp_list: List[np.ndarray]) -> List[float]:
        """
        Computes LPIPS for a batch of images.
        Converts uint8 [0, 255] arrays to torch float32 [-1.0, 1.0] tensors.
        """
        tensors_gt = []
        tensors_comp = []

        for gt_np, comp_np in zip(gt_list, comp_list):
            t_gt = (torch.from_numpy(gt_np).permute(2, 0, 1).float() / 127.5 - 1.0)
            t_comp = (torch.from_numpy(comp_np).permute(2, 0, 1).float() / 127.5 - 1.0)
            tensors_gt.append(t_gt)
            tensors_comp.append(t_comp)

        batch_gt = torch.stack(tensors_gt, dim=0).to(self.device)
        batch_comp = torch.stack(tensors_comp, dim=0).to(self.device)

        with torch.no_grad():
            lpips_tensor = self.lpips_fn(batch_gt, batch_comp)
            lpips_vals = [round(float(v.item()), 4) for v in lpips_tensor]

        return lpips_vals


def resolve_prediction_file(pred_dir: Path, sample_id: str, model_name: str) -> Optional[Path]:
    """Finds matching prediction file handling zero-padding and model suffixes."""
    stem = sample_id
    numeric_id = int(sample_id) if sample_id.isdigit() else None

    candidate_stems = [stem]
    if numeric_id is not None:
        candidate_stems.extend([f"{numeric_id}", f"{numeric_id:04d}", f"{numeric_id:03d}"])

    extensions = [".png", ".jpg", ".jpeg", ".webp"]
    model_tags = [f"_{model_name}", f"-{model_name}", ""]

    for s in candidate_stems:
        for tag in model_tags:
            for ext in extensions:
                p = pred_dir / f"{s}{tag}{ext}"
                if p.exists():
                    return p
    return None


def run_quality_pipeline(
    pred_dir: Path,
    model_name: str,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    masks_dir: Path = DEFAULT_MASKS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit: Optional[int] = None,
    batch_size: int = 16,
    device: Optional[str] = None
):
    """Executes the batch perceptual quality pipeline across the 6,000 dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{model_name}_quality_metrics.csv"
    ckpt_file = output_dir / f"{model_name}_quality_checkpoint.json"

    # 1. Load Manifests
    img_meta = {}
    if DEFAULT_IMG_MANIFEST.exists():
        df_img = pd.read_csv(DEFAULT_IMG_MANIFEST, dtype={"sample_id": str})
        for _, r in df_img.iterrows():
            img_meta[r["sample_id"]] = r.to_dict()

    mask_meta = {}
    if DEFAULT_MASK_MANIFEST.exists():
        df_mask = pd.read_csv(DEFAULT_MASK_MANIFEST, dtype={"sample_id": str})
        for _, r in df_mask.iterrows():
            mask_meta[r["sample_id"]] = r.to_dict()

    # 2. Discover Ground Truth Samples
    gt_files = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    if limit is not None:
        gt_files = gt_files[:limit]

    total_samples = len(gt_files)
    print("\n" + "=" * 80)
    print(f"🚀 INPAINTING QUALITY EVALUATION PIPELINE: {model_name.upper()}")
    print("=" * 80)
    print(f" Ground Truth Images : {images_dir} ({total_samples} samples)")
    print(f" Synthetic Masks     : {masks_dir}")
    print(f" Prediction Directory: {pred_dir}")
    print(f" Output Metrics CSV  : {out_csv}")
    print("=" * 80 + "\n")

    # 3. Checkpoint Resumption
    completed_ids = set()
    records = []
    if ckpt_file.exists() and out_csv.exists():
        try:
            with open(ckpt_file, "r") as f:
                ckpt_data = json.load(f)
                completed_ids = set(ckpt_data.get("completed_sample_ids", []))
            df_existing = pd.read_csv(out_csv, dtype={"sample_id": str})
            records = df_existing.to_dict("records")
            print(f"🔄 Resuming from checkpoint: {len(completed_ids)} samples already evaluated.\n")
        except Exception as e:
            print(f"⚠️ Warning loading checkpoint: {e}. Starting fresh.")
            completed_ids = set()
            records = []

    # 4. Initialize Evaluator
    evaluator = QualityEvaluator(device=device, lpips_batch_size=batch_size)

    # 5. Process in Batches
    samples_to_eval = [p for p in gt_files if p.stem not in completed_ids]
    if not samples_to_eval:
        print("✅ All samples already evaluated!")
        return

    pbar = tqdm(total=len(samples_to_eval), desc=f"Evaluating {model_name}", unit="img")

    # Accumulator for batch LPIPS
    batch_gt_np = []
    batch_comp_np = []
    batch_info = []

    def flush_batch():
        nonlocal batch_gt_np, batch_comp_np, batch_info, records
        if not batch_gt_np:
            return

        # Compute batch LPIPS
        lpips_scores = evaluator.compute_lpips_batch(batch_gt_np, batch_comp_np)

        for info, lp_score in zip(batch_info, lpips_scores):
            info["lpips_vgg"] = lp_score
            records.append(info)
            completed_ids.add(info["sample_id"])

        # Write intermediate CSV and checkpoint
        pd.DataFrame(records).to_csv(out_csv, index=False)
        with open(ckpt_file, "w") as f:
            json.dump({"completed_sample_ids": list(completed_ids), "last_update": time.time()}, f)

        # Clear batch
        batch_gt_np.clear()
        batch_comp_np.clear()
        batch_info.clear()

    for gt_path in samples_to_eval:
        stem = gt_path.stem
        mask_path = masks_dir / f"{stem}.png"
        pred_path = resolve_prediction_file(pred_dir, stem, model_name)

        if not pred_path or not pred_path.exists():
            pbar.update(1)
            continue

        try:
            gt_pil = Image.open(gt_path).convert("RGB")
            pred_pil = Image.open(pred_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L") if mask_path.exists() else None

            # Ensure matching 512x512
            if pred_pil.size != gt_pil.size:
                pred_pil = pred_pil.resize(gt_pil.size, Image.Resampling.BICUBIC)

            gt_np = np.array(gt_pil, dtype=np.uint8)
            pred_np = np.array(pred_pil, dtype=np.uint8)
            mask_np = (np.array(mask_pil) > 127).astype(np.float32) if mask_pil else np.zeros((512, 512), dtype=np.float32)

            # Compute PSNR & SSIM
            psnr_full, psnr_hole, ssim_val = evaluator.compute_psnr_and_ssim(gt_np, pred_np, mask_np)

            # Prepare alpha composite for LPIPS
            mask_3d = mask_np[:, :, None]
            comp_np = np.clip(mask_3d * pred_np + (1.0 - mask_3d) * gt_np, 0, 255).astype(np.uint8)

            # Pack record metadata
            i_meta = img_meta.get(stem, {})
            m_meta = mask_meta.get(stem, {})

            record = {
                "sample_id": stem,
                "model": model_name,
                "category": i_meta.get("category", "unknown"),
                "mask_type": m_meta.get("mask_type", "unknown"),
                "coverage_pct": m_meta.get("coverage_pct", round(float(mask_np.mean() * 100), 2)),
                "hole_count": m_meta.get("hole_count", 1),
                "dist_to_border": m_meta.get("dist_to_border", 0.0),
                "psnr_full": psnr_full,
                "psnr_hole": psnr_hole,
                "ssim": ssim_val
            }

            batch_gt_np.append(gt_np)
            batch_comp_np.append(comp_np)
            batch_info.append(record)

            if len(batch_gt_np) >= batch_size:
                flush_batch()

        except Exception as e:
            print(f"\n⚠️ Error processing {stem}: {e}")

        pbar.update(1)

    # Flush final remainder
    flush_batch()
    pbar.close()

    # Final summary statistics
    df_final = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print(f"📊 EVALUATION COMPLETED: {len(df_final)} SAMPLES EVALUATED")
    print("=" * 80)
    print(f"  • Mean PSNR (Full) : {df_final['psnr_full'].mean():.2f} dB (Median: {df_final['psnr_full'].median():.2f} dB)")
    print(f"  • Mean PSNR (Hole) : {df_final['psnr_hole'].mean():.2f} dB (Median: {df_final['psnr_hole'].median():.2f} dB)")
    print(f"  • Mean SSIM        : {df_final['ssim'].mean():.4f}")
    print(f"  • Mean LPIPS (VGG) : {df_final['lpips_vgg'].mean():.4f}")
    print(f"  • Output CSV       : {out_csv}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="6,000-Image Inpainting Perceptual Metric Pipeline")
    parser.add_argument("-p", "--pred-dir", required=True, type=Path, help="Directory containing prediction images")
    parser.add_argument("-m", "--model", required=True, type=str, help="Model name (migan, lama, aotgan, sd)")
    parser.add_argument("-i", "--images-dir", default=DEFAULT_IMAGES_DIR, type=Path, help="GT images directory")
    parser.add_argument("-k", "--masks-dir", default=DEFAULT_MASKS_DIR, type=Path, help="Masks directory")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path, help="Output directory")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="Batch size for LPIPS evaluation")
    parser.add_argument("-n", "--limit", default=None, type=int, help="Limit number of samples to evaluate")
    parser.add_argument("--device", default=None, type=str, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    run_quality_pipeline(
        pred_dir=args.pred_dir,
        model_name=args.model,
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()
