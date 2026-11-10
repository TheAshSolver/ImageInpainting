#!/usr/bin/env python3
"""
scripts/run_sd_benchmark_live.py
================================================================================
⚡ QUALCOMM SNAPDRAGON 8 ELITE — LIVE STABLE DIFFUSION INPAINTING BENCHMARK
================================================================================
Interactive Live Benchmark Runner for Qualcomm QIDK Stable Diffusion Inpainting
Featuring:
  - tqdm Progress bar with live ETA, sample metrics, and dynamic sub-step status
  - Auto-reconnect on USB disconnection (resilient to wire pulls)
  - Smart Resume / Checkpointing (skips already generated images unless --force-rerun)
  - Real-time thermal telemetry polling from Snapdragon 8 Elite thermal zones
  - Exact NHWC float32 raw tensor staging (image.raw + mask.raw) for native runners
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    print("[!] 'tqdm' is required for the live progress bar. Install with: pip install tqdm")
    sys.exit(1)

# Default paths
DEFAULT_DATASET = Path("Benchmark/dataset_previous")
DEFAULT_OUTPUT_BASE = Path("Benchmark/output/sd_comparison")
DEVICE_SD_DIR = "/data/local/tmp/sd_runtime"
CANONICAL_SIZE = (512, 512)
IMAGE_RAW_BYTES = 512 * 512 * 3 * 4   # 3,145,728 bytes (float32 NHWC)
MASK_RAW_BYTES = 512 * 512 * 1 * 4    # 1,048,576 bytes (float32 NHWC)


# ---------------------------------------------------------------------------
# ADB Helper Utilities with Auto-Reconnect
# ---------------------------------------------------------------------------

def run_adb(cmd_args: List[str], timeout: int = 180) -> Tuple[int, str, str]:
    """Execute an ADB command with timeout."""
    full_cmd = ["adb"] + cmd_args
    try:
        proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_device_connected() -> bool:
    """Return True if device is attached and authorized."""
    rc, out, _ = run_adb(["get-state"], timeout=5)
    return rc == 0 and "device" in out


def wait_for_device_reconnection(pbar: Optional[tqdm] = None):
    """Pause execution and block until USB cable is plugged back in."""
    if check_device_connected():
        return

    msg = "⚠️  USB DISCONNECTED! Waiting for reconnection..."
    if pbar:
        pbar.set_description(f"\033[91m{msg}\033[0m")
    else:
        print(f"\n{msg}")

    dots = 0
    while not check_device_connected():
        time.sleep(1.5)
        dots = (dots + 1) % 4
        sys.stdout.write(f"\r\033[33m[Waiting for Snapdragon 8 Elite ADB device{'.' * dots}   ]\033[0m ")
        sys.stdout.flush()

    sys.stdout.write("\r\033[K")
    print("✅ Device reconnected! Resuming benchmark pipeline.\n")
    if pbar:
        pbar.set_description("Benchmarking")
    time.sleep(1.0)


def get_device_peak_temp() -> Optional[float]:
    """Read highest thermal zone temp in degrees Celsius."""
    if not check_device_connected():
        return None
    rc, out, _ = run_adb(["shell", "cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -nr | head -n 1"])
    if rc == 0 and out.strip().isdigit():
        return int(out.strip()) / 1000.0  # milli-Celsius to Celsius
    return None


# ---------------------------------------------------------------------------
# Tensor Preprocessing for Snapdragon 8 Elite NPU
# ---------------------------------------------------------------------------

def encode_raw_tensors(img_path: Path, mask_path: Path) -> Tuple[bytes, bytes]:
    """
    Encodes image and mask into exact Snapdragon NPU float32 NHWC raw tensor binaries:
      - Image: 512x512 RGB normalized to [0.0, 1.0] -> 3,145,728 bytes
      - Mask:  512x512 float32 binary (1.0 = hole, 0.0 = keep) -> 1,048,576 bytes
    """
    img = Image.open(img_path).convert("RGB")
    if img.size != CANONICAL_SIZE:
        img = img.resize(CANONICAL_SIZE, Image.Resampling.BILINEAR)
    img_np = (np.array(img, dtype=np.float32) / 255.0)
    img_bytes = img_np.tobytes()

    mask = Image.open(mask_path).convert("L")
    if mask.size != CANONICAL_SIZE:
        mask = mask.resize(CANONICAL_SIZE, Image.Resampling.NEAREST)
    mask_np = np.array(mask, dtype=np.float32)
    mask_binary = (mask_np >= 128.0).astype(np.float32)[:, :, np.newaxis]
    mask_bytes = mask_binary.tobytes()

    assert len(img_bytes) == IMAGE_RAW_BYTES, f"Invalid image raw bytes: {len(img_bytes)}"
    assert len(mask_bytes) == MASK_RAW_BYTES, f"Invalid mask raw bytes: {len(mask_bytes)}"
    return img_bytes, mask_bytes


def stage_sample_tensors(stem: str, img_path: Path, mask_path: Path, staging_dir: Path) -> Tuple[Path, Path]:
    """Caches encoded raw float32 NHWC binaries on host to eliminate redundant conversions."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    local_img_raw = staging_dir / f"{stem}_img.raw"
    local_mask_raw = staging_dir / f"{stem}_mask.raw"

    if not (local_img_raw.exists() and local_mask_raw.exists() and local_img_raw.stat().st_size == IMAGE_RAW_BYTES):
        img_bytes, mask_bytes = encode_raw_tensors(img_path, mask_path)
        with open(local_img_raw, "wb") as f:
            f.write(img_bytes)
        with open(local_mask_raw, "wb") as f:
            f.write(mask_bytes)

    return local_img_raw, local_mask_raw


# ---------------------------------------------------------------------------
# Benchmark Engine
# ---------------------------------------------------------------------------

def discover_samples(dataset_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Scan dataset for (stem, image_path, mask_path)."""
    img_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    if not img_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing images/ or masks/ directory in {dataset_dir}")

    samples = []
    for img_p in sorted(img_dir.glob("*.png")):
        mask_p = mask_dir / img_p.name
        if mask_p.exists():
            samples.append((img_p.stem, img_p, mask_p))
    return samples


def run_benchmark():
    parser = argparse.ArgumentParser(
        description="Qualcomm Snapdragon 8 Elite Live SD Inpainting Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", choices=["sd_inpaint", "sd_inefficient"], default="sd_inpaint",
                        help="Runner target binary (default: sd_inpaint)")
    parser.add_argument("--limit", "--max-samples", dest="limit", type=int, default=None,
                        help="Number of samples to process (default: all 102)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Conditioning text prompt (default: '' to eliminate hallucinations)")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Overwrite existing outputs")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Path to dataset directory containing images/ and masks/")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE,
                        help="Directory to save generated inpainting results")
    args = parser.parse_args()

    # Model runner configuration
    is_inpaint = (args.model == "sd_inpaint")
    runner_bin = "./sd_qidk_runner_inpaint" if is_inpaint else "./sd_qidk_runner_inefficient"
    steps_label = "12-Step DPM++" if is_inpaint else "20-Step Euler"
    out_dir = args.output_base / f"results_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    host_staging_dir = Path("/tmp/sd_live_staging")

    print("=" * 70)
    print("  ⚡ QUALCOMM SNAPDRAGON 8 ELITE — LIVE SD INPAINTING BENCHMARK")
    print("=" * 70)
    print(f" Target Model     : {args.model} ({runner_bin} | {steps_label})")
    print(f" Target Hardware  : Qualcomm Hexagon HTP v79 NPU")
    print(f" Local Output Dir : {out_dir}")
    print(f" Dataset Path     : {args.dataset}")
    print(f" Prompt           : \"{args.prompt}\"")
    print(f" Smart Resume     : {'Disabled (--force-rerun)' if args.force_rerun else 'Enabled'}")
    print("=" * 70)

    # Sanity check initial device connection
    wait_for_device_reconnection()

    all_samples = discover_samples(args.dataset)
    if args.limit:
        all_samples = all_samples[:args.limit]

    total_samples = len(all_samples)
    print(f"\nDiscovered {total_samples} samples to evaluate.\n")

    latencies = []
    skipped_count = 0

    pbar = tqdm(all_samples, desc="Benchmarking", unit="img", dynamic_ncols=True)

    for stem, img_path, mask_path in pbar:
        local_out_png = out_dir / f"{stem}.png"

        # Checkpoint / Resume check
        if local_out_png.exists() and local_out_png.stat().st_size > 1024 and not args.force_rerun:
            skipped_count += 1
            pbar.set_postfix_str(f"Skipped {stem} (Cached) | Skipped: {skipped_count}")
            continue

        # Ensure device is still alive before execution
        wait_for_device_reconnection(pbar)

        temp_c = get_device_peak_temp()
        temp_str = f"{temp_c:.1f}°C" if temp_c else "N/A"

        # -------------------------------------------------------------
        # Step 1: Pre-encode & push raw tensors to device runtime directory
        # -------------------------------------------------------------
        pbar.set_postfix_str(f"Staging {stem} | NPU Temp: {temp_str}")
        local_img_raw, local_mask_raw = stage_sample_tensors(stem, img_path, mask_path, host_staging_dir)

        # Push raw tensors (required by C++ runners) and PNG references
        while True:
            rc_img, _, _ = run_adb(["push", str(local_img_raw), f"{DEVICE_SD_DIR}/image.raw"])
            rc_mask, _, _ = run_adb(["push", str(local_mask_raw), f"{DEVICE_SD_DIR}/mask.raw"])
            run_adb(["push", str(img_path), f"{DEVICE_SD_DIR}/input.png"])
            run_adb(["push", str(mask_path), f"{DEVICE_SD_DIR}/mask.png"])
            if rc_img == 0 and rc_mask == 0:
                break
            wait_for_device_reconnection(pbar)

        # -------------------------------------------------------------
        # Step 2: Execute On-Device Inpainting
        # -------------------------------------------------------------
        pbar.set_postfix_str(f"⚡ Inpainting {stem} ({steps_label}) | NPU Temp: {temp_str}")

        cmd = (
            f"cd {DEVICE_SD_DIR} && "
            f"export LD_LIBRARY_PATH={DEVICE_SD_DIR}:$LD_LIBRARY_PATH && "
            f"export ADSP_LIBRARY_PATH='{DEVICE_SD_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && "
            f"rm -f sd_output.png 2>/dev/null; "
            f"{runner_bin} \"{args.prompt}\" > sd_stdout_{stem}.txt 2>&1"
        )

        t_start = time.time()
        rc, _, _ = run_adb(["shell", cmd], timeout=300)
        t_elapsed = time.time() - t_start

        if rc != 0:
            # Check if failure was due to device disconnect
            if not check_device_connected():
                wait_for_device_reconnection(pbar)
                # Re-stage tensors and retry current sample after reconnection
                run_adb(["push", str(local_img_raw), f"{DEVICE_SD_DIR}/image.raw"])
                run_adb(["push", str(local_mask_raw), f"{DEVICE_SD_DIR}/mask.raw"])
                t_start = time.time()
                rc, _, _ = run_adb(["shell", cmd], timeout=300)
                t_elapsed = time.time() - t_start

        # -------------------------------------------------------------
        # Step 3: Pull Result and Log Timing
        # -------------------------------------------------------------
        pbar.set_postfix_str(f"Pulling {stem}.png ({t_elapsed:.1f}s)")
        pull_rc, _, _ = run_adb(["pull", f"{DEVICE_SD_DIR}/sd_output.png", str(local_out_png)])

        temp_after = get_device_peak_temp()
        temp_after_str = f"{temp_after:.1f}°C" if temp_after else temp_str

        if pull_rc == 0 and local_out_png.exists() and local_out_png.stat().st_size > 1024:
            latencies.append(t_elapsed)
            avg_lat = sum(latencies) / len(latencies)
            pbar.set_postfix_str(f"Last: {t_elapsed:.2f}s | Avg: {avg_lat:.2f}s | NPU Temp: {temp_after_str}")
        else:
            pbar.set_postfix_str(f"❌ Failed {stem} (rc={rc})")

    pbar.close()

    # -----------------------------------------------------------------
    # Final Summary Report
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ⚡ BENCHMARK RUN COMPLETE")
    print("=" * 70)
    print(f" Model Evaluated    : {args.model} ({steps_label})")
    print(f" Processed Samples  : {len(latencies)}")
    print(f" Skipped (Cached)   : {skipped_count}")
    if latencies:
        print(f" Average Latency    : {sum(latencies) / len(latencies):.2f} s / sample")
        print(f" Fastest Sample     : {min(latencies):.2f} s")
        print(f" Slowest Sample     : {max(latencies):.2f} s")
        print(f" Total Active Time  : {sum(latencies):.1f} s (~{sum(latencies)/60:.1f} min)")
    print(f" Output Directory   : {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n\n[!] Benchmark aborted by user.")
        sys.exit(0)
