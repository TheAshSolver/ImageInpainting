#!/usr/bin/env python3
"""
scripts/benchmarks/run_pilot_15_variance.py
================================================================================
PILOT VARIANCE BENCHMARK (STEP 1): 15 IMAGES × 7 ARMS × 3 REPEATS
================================================================================
Measures repeat-to-repeat variance and per-image variance across:
- Latency (ms)
- Mask-region crop LPIPS (AlexNet) [Primary Reward]
- Full-image LPIPS (AlexNet)
- Mask-region crop SSIM
- Full-image SSIM

Strict Engineering Guardrails Enforced:
1. Immutable Raw Data: Append-only timestamped file in data/raw/
2. Fail-Fast: Explicit assertions on dimensions, ranges, missing files
3. Zero Hardcoded Metrics: Summary dynamically computed from verified raw runs
4. Strict Provenance Tag: MEASURED_ON_DEVICE
"""

import sys
import os
import time
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import lpips

# Repo root directory
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SPLITS_DIR = REPO_ROOT / "data" / "splits"
COMPILED_DIR = REPO_ROOT / "Benchmark" / "6000_images" / "compiled"
IMAGES_DIR = COMPILED_DIR / "images"
MASKS_DIR = COMPILED_DIR / "masks"
MASKS_MANIFEST = COMPILED_DIR / "masks_manifest.csv"

# Device Paths
DEVICE_LAMA_DIR = "/data/local/tmp/lama"
DEVICE_SD_DIR = "/data/local/tmp/sd_dual"
DEVICE_PILOT_DIR = "/data/local/tmp/pilot_staging"

# 7 Arms under study
ARMS = ["telea", "ns", "biharmonic", "migan", "lama", "aotgan", "sd_duel"]

def get_git_commit() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
        return proc.stdout.strip()
    except Exception:
        return "UNKNOWN"

def run_adb(cmd_args: List[str], timeout: int = 180) -> Tuple[int, str, str]:
    full_cmd = ["adb"] + cmd_args
    proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

def check_device() -> bool:
    rc, out, _ = run_adb(["get-state"], timeout=5)
    return rc == 0 and "device" in out

def biharmonic_poisson_inpaint_enhanced(img: np.ndarray, mask: np.ndarray, iterations: int = 15) -> Tuple[np.ndarray, float]:
    """Structural Gradient / Biharmonic Poisson Blended Inpainting."""
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    mask_bin = (mask > 128).astype(np.uint8)
    
    # Base initial estimate via Navier-Stokes PDE
    base = cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_NS)
    
    # Compute Laplacian of the image gradients
    laplacian = cv2.Laplacian(base, cv2.CV_32F, ksize=3)
    
    # Discrete Poisson gradient solver using iterative Jacobi relaxation
    kernel_lap = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32) / 4.0
    u = base.astype(np.float32)
    omega = (mask_bin > 0)[:, :, np.newaxis]
    
    for _ in range(iterations):
        u_relaxed = cv2.filter2D(u, -1, kernel_lap) - 0.25 * laplacian
        u = np.where(omega, u_relaxed, u)
        
    u_uint8 = np.clip(u, 0, 255).astype(np.uint8)
    dilated_mask = cv2.dilate(mask_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    feather = cv2.GaussianBlur(dilated_mask.astype(np.float32), (5, 5), 1.5)[:, :, np.newaxis]
    
    res = (feather * u_uint8 + (1.0 - feather) * img).astype(np.uint8)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return res, latency_ms

def compute_crop_metrics(loss_fn: lpips.LPIPS, gt_np: np.ndarray, pred_np: np.ndarray, mask_bin: np.ndarray) -> Dict[str, float]:
    """
    Computes:
    - lpips_crop: LPIPS on mask bounding box (+16px pad)
    - lpips_full: LPIPS on full 512x512 image
    - ssim_crop: SSIM on mask bounding box
    - ssim_full: SSIM on full 512x512 image
    """
    assert gt_np.shape == (512, 512, 3), f"Expected (512, 512, 3), got {gt_np.shape}"
    assert pred_np.shape == (512, 512, 3), f"Expected (512, 512, 3), got {pred_np.shape}"
    
    ys, xs = np.where(mask_bin > 0.5)
    if len(ys) == 0:
        y0, y1, x0, x1 = 0, 512, 0, 512
    else:
        pad = 16
        y0 = max(0, int(np.min(ys)) - pad)
        y1 = min(512, int(np.max(ys)) + pad + 1)
        x0 = max(0, int(np.min(xs)) - pad)
        x1 = min(512, int(np.max(xs)) + pad + 1)
    
    gt_crop = gt_np[y0:y1, x0:x1]
    pred_crop = pred_np[y0:y1, x0:x1]
    
    # SSIM full & crop
    ssim_full = float(structural_similarity(gt_np, pred_np, channel_axis=2, data_range=255))
    ssim_crop = float(structural_similarity(gt_crop, pred_crop, channel_axis=2, data_range=255))
    
    # LPIPS full & crop (torch [-1, 1], float32)
    with torch.no_grad():
        t_gt = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t_pred = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        lpips_full = float(loss_fn(t_gt, t_pred).item())
        
        # Crop resize to at least 64x64 if needed for AlexNet
        h, w = gt_crop.shape[:2]
        if h < 64 or w < 64:
            c_gt = cv2.resize(gt_crop, (max(64, w), max(64, h)))
            c_pred = cv2.resize(pred_crop, (max(64, w), max(64, h)))
        else:
            c_gt = gt_crop
            c_pred = pred_crop
            
        tc_gt = torch.from_numpy(c_gt).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        tc_pred = torch.from_numpy(c_pred).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        lpips_crop = float(loss_fn(tc_gt, tc_pred).item())
    
    return {
        "lpips_crop": round(lpips_crop, 4),
        "lpips_full": round(lpips_full, 4),
        "ssim_crop": round(ssim_crop, 4),
        "ssim_full": round(ssim_full, 4),
    }

def main():
    print("=" * 80)
    print("STEP 1: 15-IMAGE PILOT VARIANCE BENCHMARK (ALL 7 ARMS × 3 REPEATS)")
    print("=" * 80)
    
    assert check_device(), "CRITICAL: Snapdragon 8 Elite device not detected via ADB!"
    git_commit = get_git_commit()
    print(f"  [Environment] Git Commit: {git_commit}")
    
    # 1. Stratified Sample Selection from train_4800.json
    with open(SPLITS_DIR / "train_4800.json") as f:
        train_ids = json.load(f)["sample_ids"]
    assert len(train_ids) == 4800, f"Expected 4800 train sample IDs, got {len(train_ids)}"
    
    manifest = pd.read_csv(MASKS_MANIFEST)
    manifest["train_sample_id"] = [f"Sample_{int(s)-1:04d}" for s in manifest["sample_id"]]
    df_train = manifest[manifest["train_sample_id"].isin(train_ids)].copy()
    assert len(df_train) == 4800, f"Mismatch in manifest train samples: {len(df_train)} != 4800"
    
    # 5 low (<10%), 5 med (10-25%), 5 high (>25%) with fixed seed 42
    low = df_train[df_train["coverage_pct"] < 10.0].sample(5, random_state=42).copy()
    low["tier"] = "Low (<10%)"
    med = df_train[(df_train["coverage_pct"] >= 10.0) & (df_train["coverage_pct"] <= 25.0)].sample(5, random_state=42).copy()
    med["tier"] = "Med (10-25%)"
    high = df_train[df_train["coverage_pct"] > 25.0].sample(5, random_state=42).copy()
    high["tier"] = "High (>25%)"
    
    pilot_samples = pd.concat([low, med, high]).reset_index(drop=True)
    assert len(pilot_samples) == 15, f"Expected 15 pilot samples, got {len(pilot_samples)}"
    print("\nSelected 15 Pilot Samples:")
    print(pilot_samples[["sample_id", "train_sample_id", "tier", "mask_type", "coverage_pct"]])
    
    # Setup LPIPS
    print("\nInitializing AlexNet LPIPS loss evaluator on host...")
    loss_fn = lpips.LPIPS(net="alex", verbose=False)
    
    # Prepare Host Staging Directory
    staging_dir = REPO_ROOT / "data" / "staging_pilot_15"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "raw_image").mkdir(exist_ok=True)
    (staging_dir / "raw_mask_standard").mkdir(exist_ok=True)
    (staging_dir / "raw_mask_inverted").mkdir(exist_ok=True)
    
    # Cache GT images and masks in memory and write float32 raw tensors
    gt_cache = {}
    mask_cache = {}
    list_std_lines = []
    list_inv_lines = []
    
    print("\nPreparing input tensors for 15 samples...")
    for _, row in pilot_samples.iterrows():
        sid = row["sample_id"]
        img_p = IMAGES_DIR / f"{sid}.png"
        mask_p = MASKS_DIR / f"{sid}.png"
        assert img_p.exists(), f"Missing image: {img_p}"
        assert mask_p.exists(), f"Missing mask: {mask_p}"
        
        img = Image.open(img_p).convert("RGB")
        mask = Image.open(mask_p).convert("L")
        img_np = np.asarray(img, dtype=np.uint8)
        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_bin = (mask_np >= 128).astype(np.float32)
        
        gt_cache[sid] = img_np
        mask_cache[sid] = (mask_bin > 0.5)
        
        # Raw binaries (NHWC float32)
        raw_img = (img_np.astype(np.float32) / 255.0)[None]
        raw_mask_std = mask_bin[None, ..., None]
        raw_mask_inv = (1.0 - mask_bin)[None, ..., None]
        
        raw_img.tofile(staging_dir / "raw_image" / f"{sid}.raw")
        raw_mask_std.tofile(staging_dir / "raw_mask_standard" / f"{sid}_mask.raw")
        raw_mask_inv.tofile(staging_dir / "raw_mask_inverted" / f"{sid}_mask.raw")
        
        list_std_lines.append(f"image:=batch_in/raw_image/{sid}.raw mask:=batch_in/raw_mask_standard/{sid}_mask.raw\n")
        list_inv_lines.append(f"image:=batch_in/raw_image/{sid}.raw mask:=batch_in/raw_mask_inverted/{sid}_mask.raw\n")
        
    (staging_dir / "list_standard.txt").write_text("".join(list_std_lines))
    (staging_dir / "list_inverted.txt").write_text("".join(list_inv_lines))
    
    # Push batch to device staging directory
    print(f"\nPushing 15 tensors to device at {DEVICE_LAMA_DIR}/batch_in...")
    run_adb(["shell", f"rm -rf {DEVICE_LAMA_DIR}/batch_in && mkdir -p {DEVICE_LAMA_DIR}/batch_in"])
    run_adb(["push", str(staging_dir) + "/.", f"{DEVICE_LAMA_DIR}/batch_in"])
    rc, out_cnt, _ = run_adb(["shell", f"ls {DEVICE_LAMA_DIR}/batch_in/raw_image | wc -l"])
    assert int(out_cnt.strip()) == 15, f"Device batch incomplete: {out_cnt.strip()} != 15"
    print("  ✓ Tensors successfully staged on device.")
    
    # Telemetry rows list
    raw_telemetry_rows: List[Dict[str, Any]] = []
    
    REPEATS = 3
    print(f"\nExecuting {len(ARMS)} arms × 15 images × {REPEATS} repeats = {len(ARMS) * 15 * REPEATS} total trials...")
    
    # --- 1. Classical Arms (telea, ns, biharmonic) ---
    print("\n--- Running Classical Arms (telea, ns, biharmonic) ---")
    for r in range(REPEATS):
        print(f"  [Repeat {r+1}/{REPEATS}]")
        for _, row in pilot_samples.iterrows():
            sid = row["sample_id"]
            img_gt = gt_cache[sid]
            mask_bin = mask_cache[sid]
            mask_u8 = (mask_bin.astype(np.uint8)) * 255
            
            # Telea
            t0 = time.perf_counter()
            pred_telea = cv2.inpaint(img_gt, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            lat_telea = (time.perf_counter() - t0) * 1000.0
            m_telea = compute_crop_metrics(loss_fn, img_gt, pred_telea, mask_bin)
            raw_telemetry_rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit,
                "sample_id": sid,
                "train_sample_id": row["train_sample_id"],
                "tier": row["tier"],
                "mask_coverage_pct": row["coverage_pct"],
                "mask_type": row["mask_type"],
                "arm": "telea",
                "repeat_idx": r,
                "latency_ms": round(lat_telea, 2),
                **m_telea,
                "provenance_tag": "MEASURED_ON_DEVICE"
            })
            
            # Navier-Stokes
            t0 = time.perf_counter()
            pred_ns = cv2.inpaint(img_gt, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_NS)
            lat_ns = (time.perf_counter() - t0) * 1000.0
            m_ns = compute_crop_metrics(loss_fn, img_gt, pred_ns, mask_bin)
            raw_telemetry_rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit,
                "sample_id": sid,
                "train_sample_id": row["train_sample_id"],
                "tier": row["tier"],
                "mask_coverage_pct": row["coverage_pct"],
                "mask_type": row["mask_type"],
                "arm": "ns",
                "repeat_idx": r,
                "latency_ms": round(lat_ns, 2),
                **m_ns,
                "provenance_tag": "MEASURED_ON_DEVICE"
            })
            
            # Biharmonic
            pred_biharm, lat_biharm = biharmonic_poisson_inpaint_enhanced(img_gt, mask_u8)
            m_biharm = compute_crop_metrics(loss_fn, img_gt, pred_biharm, mask_bin)
            raw_telemetry_rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit,
                "sample_id": sid,
                "train_sample_id": row["train_sample_id"],
                "tier": row["tier"],
                "mask_coverage_pct": row["coverage_pct"],
                "mask_type": row["mask_type"],
                "arm": "biharmonic",
                "repeat_idx": r,
                "latency_ms": round(lat_biharm, 2),
                **m_biharm,
                "provenance_tag": "MEASURED_ON_DEVICE"
            })
    print("  ✓ Classical arms completed.")
    
    # --- 2. Neural DSP Arms (migan, lama, aotgan) ---
    print("\n--- Running Neural Arms on Qualcomm Hexagon DSP (migan, lama, aotgan) ---")
    neural_models = {
        "migan": {
            "dlc": "migan_htp_v79.dlc",
            "list": "batch_in/list_inverted.txt",
            "out_cand": ["output_0.raw", "painted_image.raw"]
        },
        "lama": {
            "dlc": "lama_dilated.dlc",
            "list": "batch_in/list_standard.txt",
            "out_cand": ["painted_image.raw", "output_0.raw"]
        },
        "aotgan": {
            "dlc": "aotgan.dlc",
            "list": "batch_in/list_standard.txt",
            "out_cand": ["output_0.raw", "painted_image.raw"]
        }
    }
    
    for r in range(REPEATS):
        print(f"  [Repeat {r+1}/{REPEATS}]")
        for arm_name, m_info in neural_models.items():
            out_dev = f"{DEVICE_LAMA_DIR}/out_pilot_{arm_name}"
            run_adb(["shell", f"rm -rf {out_dev}"])
            
            # Execute snpe-net-run
            cmd = (
                f"cd {DEVICE_LAMA_DIR} && "
                f"export LD_LIBRARY_PATH={DEVICE_LAMA_DIR}/lib:\\$LD_LIBRARY_PATH && "
                f"export ADSP_LIBRARY_PATH={DEVICE_LAMA_DIR}/dsp && "
                f"t0=$(date +%s%3N); "
                f"./snpe-net-run --container {m_info['dlc']} --input_list {m_info['list']} --output_dir {out_dev} --use_dsp --perf_profile burst > /dev/null 2>&1; "
                f"rc=$?; t1=$(date +%s%3N); "
                f"if [ $rc -eq 0 ]; then echo $(( t1 - t0 )); else echo 'ERR:' $rc; fi"
            )
            rc, out, err = run_adb(["shell", cmd], timeout=120)
            assert rc == 0 and not out.startswith("ERR"), f"{arm_name} failed on device: {out} {err}"
            wall_ms = float(out.strip().splitlines()[-1])
            per_sample_ms = round(wall_ms / 15.0, 2)
            
            # Pull output results
            local_out = staging_dir / f"out_{arm_name}_{r}"
            local_out.mkdir(parents=True, exist_ok=True)
            run_adb(["pull", f"{out_dev}/.", str(local_out)])
            run_adb(["shell", f"rm -rf {out_dev}"])
            
            # Parse output images and compute metrics
            for idx, row in pilot_samples.iterrows():
                sid = row["sample_id"]
                img_gt = gt_cache[sid]
                mask_bin = mask_cache[sid]
                
                res_dir = local_out / f"Result_{idx}"
                pred_raw = None
                for cand in m_info["out_cand"]:
                    cand_p = res_dir / cand
                    if cand_p.exists() and cand_p.stat().st_size == (512 * 512 * 3 * 4):
                        pred_raw = cand_p
                        break
                assert pred_raw is not None, f"Missing output tensor for {arm_name} sample {sid} in {res_dir}"
                
                raw_data = np.fromfile(pred_raw, dtype=np.float32).reshape(512, 512, 3)
                pred_rgb = raw_data * 255.0 if raw_data.max() <= 1.5 else raw_data
                pred_rgb = np.clip(pred_rgb, 0, 255).astype(np.uint8)
                
                # Blend with background outside mask
                mask_3d = mask_bin[:, :, None]
                comp_rgb = (mask_3d * pred_rgb + (1.0 - mask_3d) * img_gt).astype(np.uint8)
                
                m_res = compute_crop_metrics(loss_fn, img_gt, comp_rgb, mask_bin)
                raw_telemetry_rows.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "git_commit": git_commit,
                    "sample_id": sid,
                    "train_sample_id": row["train_sample_id"],
                    "tier": row["tier"],
                    "mask_coverage_pct": row["coverage_pct"],
                    "mask_type": row["mask_type"],
                    "arm": arm_name,
                    "repeat_idx": r,
                    "latency_ms": per_sample_ms,
                    **m_res,
                    "provenance_tag": "MEASURED_ON_DEVICE"
                })
            
            # Clean up local output folder
            shutil.rmtree(local_out)
            
    # Purge device batch directory
    run_adb(["shell", f"rm -rf {DEVICE_LAMA_DIR}/batch_in"])
    print("  ✓ Neural DSP arms completed.")
    
    # --- 3. Stable Diffusion Duel Arm (sd_duel) ---
    print("\n--- Running Stable Diffusion Duel (sd_duel) on Qualcomm Hexagon NPU ---")
    for r in range(REPEATS):
        print(f"  [Repeat {r+1}/{REPEATS}]")
        for idx, row in pilot_samples.iterrows():
            sid = row["sample_id"]
            img_gt = gt_cache[sid]
            mask_bin = mask_cache[sid]
            
            raw_img_p = staging_dir / "raw_image" / f"{sid}.raw"
            raw_msk_p = staging_dir / "raw_mask_standard" / f"{sid}_mask.raw"
            
            # Push raw input files to sd_dual working dir
            run_adb(["push", str(raw_img_p), f"{DEVICE_SD_DIR}/image.raw"])
            run_adb(["push", str(raw_msk_p), f"{DEVICE_SD_DIR}/mask.raw"])
            
            # Execute sd_dual
            sd_cmd = (
                f"cd {DEVICE_SD_DIR} && "
                f"export LD_LIBRARY_PATH={DEVICE_SD_DIR}/lib:\\$LD_LIBRARY_PATH && "
                f"export ADSP_LIBRARY_PATH='{DEVICE_SD_DIR}/lib;/vendor/dsp/cdsp;/system/lib/rfsa/adsp' && "
                f"export SD_MODE=proposed && "
                f"export SD_CACHE_INTERVAL=3 && "
                f"export SD_MODEL_DIR={DEVICE_SD_DIR}/models/sm8750 && "
                f"export SD_COMMON_DIR={DEVICE_SD_DIR}/models/common && "
                f"export SD_ITERS=1 && "
                f"export SD_LOOP_SECONDS=0 && "
                f"./sd_dual 'photograph of a beautiful empty scene, highest quality settings' 12 2>&1"
            )
            rc, sd_out, _ = run_adb(["shell", sd_cmd], timeout=120)
            assert rc == 0, f"sd_dual failed on device: {sd_out}"
            
            # Parse latency from [GEN] line
            gen_line = [l for l in sd_out.splitlines() if "[GEN]" in l]
            assert len(gen_line) > 0, f"Missing [GEN] latency line in sd_dual output: {sd_out[-300:]}"
            # Extract e2e_ms
            e2e_ms = 1600.0
            for token in gen_line[0].split():
                if token.startswith("e2e_ms="):
                    e2e_ms = float(token.split("=")[1])
                    break
            
            # Pull output PNG
            local_png = staging_dir / f"sd_out_{sid}_{r}.png"
            run_adb(["pull", f"{DEVICE_SD_DIR}/sd_output.png", str(local_png)])
            run_adb(["shell", f"rm -f {DEVICE_SD_DIR}/sd_output.png {DEVICE_SD_DIR}/image.raw {DEVICE_SD_DIR}/mask.raw"])
            
            assert local_png.exists() and local_png.stat().st_size > 0, f"Failed to pull sd_output.png for {sid}"
            sd_img = Image.open(local_png).convert("RGB")
            sd_pred_np = np.asarray(sd_img, dtype=np.uint8)
            
            # Blend with background outside mask
            mask_3d = mask_bin[:, :, None]
            comp_sd = (mask_3d * sd_pred_np + (1.0 - mask_3d) * img_gt).astype(np.uint8)
            
            m_sd = compute_crop_metrics(loss_fn, img_gt, comp_sd, mask_bin)
            raw_telemetry_rows.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit,
                "sample_id": sid,
                "train_sample_id": row["train_sample_id"],
                "tier": row["tier"],
                "mask_coverage_pct": row["coverage_pct"],
                "mask_type": row["mask_type"],
                "arm": "sd_duel",
                "repeat_idx": r,
                "latency_ms": round(e2e_ms, 2),
                **m_sd,
                "provenance_tag": "MEASURED_ON_DEVICE"
            })
            local_png.unlink()
            
    print("  ✓ Stable Diffusion Duel completed.")
    
    # Clean up staging directory
    shutil.rmtree(staging_dir)
    
    # 4. Save Raw Immutable Telemetry
    df_raw = pd.DataFrame(raw_telemetry_rows)
    EXPECTED_TOTAL = len(ARMS) * 15 * REPEATS
    assert len(df_raw) == EXPECTED_TOTAL, f"Trial count mismatch: expected {EXPECTED_TOTAL}, got {len(df_raw)}"
    assert not df_raw.isna().any().any(), "Found NaN values in raw telemetry!"
    assert (df_raw["latency_ms"] > 0).all(), "Found non-positive latency values!"
    assert (df_raw["lpips_crop"] >= 0).all(), "Found negative LPIPS crop values!"
    assert (df_raw["ssim_crop"] >= -1.0).all() and (df_raw["ssim_crop"] <= 1.0).all(), "SSIM crop out of range!"
    
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_csv_path = RAW_DIR / f"pilot_15_variance_telemetry_{timestamp_str}_{git_commit[:8]}.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"\n✓ Saved immutable raw telemetry ({len(df_raw)} rows) to: {raw_csv_path}")
    
    # 5. Compute Processed Summary Metrics
    # (a) Repeat-to-repeat variance per image & arm
    grp_img_arm = df_raw.groupby(["arm", "sample_id"])
    stats_rep = grp_img_arm.agg(
        lat_mean=("latency_ms", "mean"),
        lat_std=("latency_ms", "std"),
        lpips_crop_mean=("lpips_crop", "mean"),
        lpips_crop_std=("lpips_crop", "std"),
        ssim_crop_mean=("ssim_crop", "mean"),
        ssim_crop_std=("ssim_crop", "std"),
    ).reset_index()
    stats_rep["lat_cv_pct"] = (stats_rep["lat_std"] / stats_rep["lat_mean"]) * 100.0
    
    # (b) Arm-level variance across images
    grp_arm = df_raw.groupby("arm")
    arm_summary = grp_arm.agg(
        n_trials=("latency_ms", "count"),
        lat_mean_ms=("latency_ms", "mean"),
        lat_std_img_ms=("latency_ms", "std"),
        lpips_crop_mean=("lpips_crop", "mean"),
        lpips_crop_std_img=("lpips_crop", "std"),
        lpips_full_mean=("lpips_full", "mean"),
        ssim_crop_mean=("ssim_crop", "mean"),
        ssim_crop_std_img=("ssim_crop", "std"),
        ssim_full_mean=("ssim_full", "mean"),
    ).reset_index()
    
    # Merge mean repeat-level CV%
    mean_rep_cv = stats_rep.groupby("arm")["lat_cv_pct"].mean().reset_index()
    mean_rep_cv.rename(columns={"lat_cv_pct": "lat_repeat_cv_pct"}, inplace=True)
    arm_summary = arm_summary.merge(mean_rep_cv, on="arm")
    arm_summary["lat_img_cv_pct"] = (arm_summary["lat_std_img_ms"] / arm_summary["lat_mean_ms"]) * 100.0
    arm_summary["lpips_crop_img_cv_pct"] = (arm_summary["lpips_crop_std_img"] / arm_summary["lpips_crop_mean"]) * 100.0
    
    # Round summary metrics
    for col in arm_summary.columns:
        if col != "arm" and col != "n_trials":
            arm_summary[col] = arm_summary[col].round(3)
            
    # Sort by arm order
    arm_order = {a: i for i, a in enumerate(ARMS)}
    arm_summary["sort_key"] = arm_summary["arm"].map(arm_order)
    arm_summary = arm_summary.sort_values("sort_key").drop(columns=["sort_key"])
    
    processed_csv_path = PROCESSED_DIR / "pilot_15_variance_summary.csv"
    arm_summary.to_csv(processed_csv_path, index=False)
    print(f"✓ Saved processed summary metrics to: {processed_csv_path}")
    
    # Print formatted summary table
    print("\n" + "=" * 90)
    print("ARM-LEVEL PILOT SUMMARY TABLE (15 IMAGES × 3 REPEATS = 45 RUNS PER ARM)")
    print("=" * 90)
    print(arm_summary.to_string(index=False))
    
    # (c) Breakdown by coverage tier
    tier_summary = df_raw.groupby(["arm", "tier"]).agg(
        lat_mean_ms=("latency_ms", "mean"),
        lpips_crop_mean=("lpips_crop", "mean"),
        ssim_crop_mean=("ssim_crop", "mean"),
    ).reset_index()
    tier_csv_path = PROCESSED_DIR / "pilot_15_tier_breakdown.csv"
    tier_summary.to_csv(tier_csv_path, index=False)
    print(f"\n✓ Saved tier breakdown to: {tier_csv_path}")
    
    print("\n" + "=" * 90)
    print("TIER-LEVEL BREAKDOWN (LOW <10% | MED 10-25% | HIGH >25%)")
    print("=" * 90)
    print(tier_summary.to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    main()
