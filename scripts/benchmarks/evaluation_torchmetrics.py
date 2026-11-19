import os
import re
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.functional as TF
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def extract_number(path: Path):
    """Extracts the first integer found in the filename (e.g., 'frame_004.png' -> 4)."""
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[0])

def index_files_by_number(directory: str):
    """Maps extracted integers to their file paths."""
    p = Path(directory)
    num_to_path = {}
    for f in p.iterdir():
        if f.suffix.lower() in VALID_EXTENSIONS:
            num = extract_number(f)
            if num is not None:
                num_to_path[num] = f
    return num_to_path

def load_image_tensor(image_path: Path, target_size=None) -> torch.Tensor:
    """Loads an image into a float tensor of shape (1, 3, H, W) in range [0.0, 1.0]."""
    img = Image.open(image_path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.BICUBIC)
    return TF.to_tensor(img).unsqueeze(0)

def benchmark_datasets(gen_dir: str, ref_dir: str, offset: int = 1, device: str = None):
    """
    Args:
        gen_dir: Directory with generated files (e.g., 0, 1, 2, ...)
        ref_dir: Directory with reference files (e.g., 1, 2, 3, ...)
        offset: Added to gen index to get ref index (default = 1: gen 0 -> ref 1)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Running evaluation on: {device}")

    gen_map = index_files_by_number(gen_dir)
    ref_map = index_files_by_number(ref_dir)

    # Perform strict sorted one-to-one numeric matching
    matched_pairs = []
    for gen_idx in sorted(gen_map.keys()):
        expected_ref_idx = gen_idx + offset
        if expected_ref_idx in ref_map:
            matched_pairs.append((gen_map[gen_idx], ref_map[expected_ref_idx], gen_idx, expected_ref_idx))
        else:
            print(f"[Warning] No reference match found for Gen idx {gen_idx} (expected Ref idx {expected_ref_idx})")

    if not matched_pairs:
        raise ValueError("No matching index pairs could be aligned between the directories.")

    print(f"Successfully matched {len(matched_pairs)} image pairs in sequential sorted order.\n")
    # Preview first couple matches
    for g, r, gi, ri in matched_pairs[:3]:
        print(f"  [Match Sample] Gen (idx {gi}): {g.name}  <-->  Ref (idx {ri}): {r.name}")
    print("  ...")

    # -------------------------------------------------------------------------
    # 1. Initialize Metrics
    # -------------------------------------------------------------------------
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    # -------------------------------------------------------------------------
    # 2. Sequential Evaluation Loop
    # -------------------------------------------------------------------------
    for gen_path, ref_path, _, _ in tqdm(matched_pairs, desc="Evaluating pairs"):
        ref_tensor = load_image_tensor(ref_path).to(device)
        _, _, h, w = ref_tensor.shape
        gen_tensor = load_image_tensor(gen_path, target_size=(w, h)).to(device)

        # PSNR & SSIM ([0, 1] range)
        total_psnr += psnr_metric(gen_tensor, ref_tensor).item()
        total_ssim += ssim_metric(gen_tensor, ref_tensor).item()

        # LPIPS ([-1, 1] range)
        gen_lpips = torch.clamp(gen_tensor * 2.0 - 1.0, -1.0, 1.0)
        ref_lpips = torch.clamp(ref_tensor * 2.0 - 1.0, -1.0, 1.0)
        total_lpips += lpips_metric(gen_lpips, ref_lpips).item()

        # FID (uint8 [0, 255])
        ref_uint8 = (ref_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        gen_uint8 = (gen_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        fid_metric.update(ref_uint8, real=True)
        fid_metric.update(gen_uint8, real=False)

    # -------------------------------------------------------------------------
    # 3. Final Summary
    # -------------------------------------------------------------------------
    n = len(matched_pairs)
    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    avg_lpips = total_lpips / n
    
    print("\nComputing final Frechet Inception Distance (FID)...")
    final_fid = fid_metric.compute().item()

    print("\n" + "=" * 45)
    print("        INPAINTING BENCHMARK RESULTS         ")
    print("=" * 45)
    print(f" Total Matched Pairs    : {n}")
    print(f" PSNR  (Higher is better): {avg_psnr:.3f} dB")
    print(f" SSIM  (Higher is better): {avg_ssim:.4f}")
    print(f" LPIPS (Lower is better) : {avg_lpips:.4f}")
    print(f" FID   (Lower is better) : {final_fid:.3f}")
    print("=" * 45 + "\n")

    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "fid": final_fid,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sorted 1-to-1 Index Matching Image Benchmark")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated images (0 to N)")
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory with reference images (1 to N+1)")
    parser.add_argument("--offset", type=int, default=1, help="Index offset (gen_idx + offset = ref_idx). Default: 1")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'")

    args = parser.parse_args()
    benchmark_datasets(args.gen_dir, args.ref_dir, offset=args.offset, device=args.device)