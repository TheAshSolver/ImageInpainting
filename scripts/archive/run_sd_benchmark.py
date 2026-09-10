#!/usr/bin/env python3
"""
Automated Batch Runner for Stable Diffusion 1.5 RePaint on Snapdragon 8 Elite (QIDK).

Executes the native sd_qidk_runner_encoder pipeline on Hexagon NPU (HTP v79),
processes all 17 benchmark pairs with strict NHWC float32 formatting,
measures latency per sample, pulls outputs to Benchmark/output/sd/results/{idx}_sd.png,
and logs a latency report.
"""

import os
import sys
import time
import glob
import re
import csv
import argparse
import subprocess
import numpy as np
from PIL import Image

# Default device paths
DEVICE_SD_DIR = "/data/local/tmp/sd_runtime"
DEVICE_IMAGE_RAW = f"{DEVICE_SD_DIR}/image.raw"
DEVICE_MASK_RAW = f"{DEVICE_SD_DIR}/mask.raw"
DEVICE_OUTPUT_PNG = f"{DEVICE_SD_DIR}/sd_output.png"
DEVICE_RUNNER = "./sd_qidk_runner_encoder"

# Default local paths
LOCAL_BASE_DIR = "Benchmark"
LOCAL_INPUT_DIR = os.path.join(LOCAL_BASE_DIR, "input")
LOCAL_OUTPUT_DIR = os.path.join(LOCAL_BASE_DIR, "output", "sd")
LOCAL_RESULTS_DIR = os.path.join(LOCAL_OUTPUT_DIR, "results")


def run_adb(cmd, check=True, capture_output=True, timeout=300):
    """Executes an adb command list or string."""
    if isinstance(cmd, str):
        cmd_list = ["adb", "shell", cmd]
    else:
        cmd_list = ["adb"] + cmd
    
    try:
        res = subprocess.run(
            cmd_list,
            check=check,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            timeout=timeout
        )
        return res.returncode == 0, res.stdout if capture_output else "", res.stderr if capture_output else ""
    except subprocess.TimeoutExpired:
        print(f"❌ ADB command timed out after {timeout}s: {' '.join(cmd_list[:5])}...")
        return False, "", "Timeout"
    except subprocess.CalledProcessError as e:
        return False, e.stdout or "", e.stderr or ""


def convert_pair_to_raw(image_path, mask_path, target_size=(512, 512)):
    """
    Converts image and mask pairs to exact binary layout required by sd_runner_encoder:
      - image.raw: NHWC (512, 512, 3) float32 normalized in [0.0, 1.0] (3,145,728 bytes)
      - mask.raw:  (512, 512, 1) float32 binary (1.0=hole, 0.0=keep) (1,048,576 bytes)
    """
    # 1. Image processing: RGB -> 512x512 -> float32 [0.0, 1.0] -> NHWC
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    img_arr = (np.array(img, dtype=np.float32) / 255.0)  # Shape (512, 512, 3)
    img_bytes = img_arr.tobytes()

    # 2. Mask processing: L -> 512x512 -> threshold -> float32 (1.0=hole, 0.0=keep)
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(target_size, Image.Resampling.NEAREST)
    mask_np = np.array(mask, dtype=np.float32)
    # Threshold at 128: 255 -> 1.0 (hole), 0 -> 0.0 (keep)
    mask_binary = (mask_np >= 128.0).astype(np.float32)
    mask_bytes = mask_binary.tobytes()

    return img_bytes, mask_bytes


def find_benchmark_pairs(input_dir=LOCAL_INPUT_DIR, max_index=17):
    """Discovers and pairs input images and masks (1 to max_index)."""
    img_dir = os.path.join(input_dir, "image")
    mask_dir = os.path.join(input_dir, "mask")
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")

    pairs = []
    for idx in range(1, max_index + 1):
        # Locate image
        img_path = None
        for ext in valid_exts:
            p = os.path.join(img_dir, f"{idx}{ext}")
            if os.path.isfile(p):
                img_path = p
                break

        if not img_path:
            continue

        # Locate mask
        mask_path = None
        candidates = [
            os.path.join(mask_dir, f"{idx}{ext}"),
            os.path.join(mask_dir, f"{idx}_mask{ext}"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                mask_path = c
                break

        if not mask_path:
            print(f"⚠️ [Warning] Mask for image index {idx} not found in {mask_dir}. Skipping.")
            continue

        pairs.append((idx, img_path, mask_path))

    return pairs


def run_sd_batch(
    prompt="A cinematic shot of sunset",
    start_idx=1,
    end_idx=17,
    input_dir=LOCAL_INPUT_DIR,
    output_dir=LOCAL_OUTPUT_DIR,
    results_dir=LOCAL_RESULTS_DIR,
    zero_indexed=False,
    clean_device_steps=True,
    device_dir=DEVICE_SD_DIR
):
    """Runs Stable Diffusion batch inpainting on connected QIDK device."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Verify device connection
    ok, out, _ = run_adb(["get-state"])
    if not ok or "device" not in out:
        print("❌ Error: No ADB device connected or authorized. Aborting.")
        return False

    print("=" * 65)
    print("      Snapdragon 8 Elite SD 1.5 RePaint Batch Benchmarking      ")
    print("=" * 65)
    print(f"  Device Working Directory : {device_dir}")
    print(f"  Local Results Directory  : {results_dir}")
    print(f"  Conditioning Prompt      : \"{prompt}\"")
    print(f"  Zero-Indexed Naming      : {zero_indexed} (e.g. {'0_sd.png' if zero_indexed else '1_sd.png'})")
    print("=" * 65 + "\n")

    # Discover benchmark pairs
    all_pairs = find_benchmark_pairs(input_dir, max_index=end_idx)
    selected_pairs = [p for p in all_pairs if start_idx <= p[0] <= end_idx]

    if not selected_pairs:
        print("❌ No matching image/mask pairs found.")
        return False

    print(f"Found {len(selected_pairs)} benchmark pairs to evaluate (indices {start_idx}..{end_idx}).\n")

    # Clean any stale outputs on device
    run_adb(f"rm -f {device_dir}/sd_output.png {device_dir}/step_*.png")

    latency_records = []
    total_start = time.perf_counter()

    for i, (pair_idx, img_path, mask_path) in enumerate(selected_pairs, 1):
        target_name_idx = (pair_idx - 1) if zero_indexed else pair_idx
        local_output_png = os.path.join(results_dir, f"{target_name_idx}_sd.png")

        print(f"[{i:02d}/{len(selected_pairs):02d}] Processing Pair {pair_idx}:")
        print(f"   Image: {os.path.basename(img_path)}  |  Mask: {os.path.basename(mask_path)}")

        # 1. Format raw buffers
        conv_start = time.perf_counter()
        img_bytes, mask_bytes = convert_pair_to_raw(img_path, mask_path)
        conv_time = time.perf_counter() - conv_start

        # 2. Push raw buffers to device
        push_start = time.perf_counter()
        temp_img = f"/tmp/sd_img_{pair_idx}.raw"
        temp_mask = f"/tmp/sd_mask_{pair_idx}.raw"
        with open(temp_img, "wb") as f:
            f.write(img_bytes)
        with open(temp_mask, "wb") as f:
            f.write(mask_bytes)

        ok_img, _, _ = run_adb(["push", temp_img, DEVICE_IMAGE_RAW])
        ok_mask, _, _ = run_adb(["push", temp_mask, DEVICE_MASK_RAW])
        try:
            os.remove(temp_img)
            os.remove(temp_mask)
        except OSError:
            pass

        if not (ok_img and ok_mask):
            print(f"   ❌ Failed to push raw buffers for pair {pair_idx} to device. Skipping.")
            continue
        push_time = time.perf_counter() - push_start

        # 3. Execute runner on device
        # Note: /data/local/tmp/sd_runtime contains QNN HTP runtime libs.
        # System linker path handles standard libc/libm without namespace collisions.
        cmd = (
            f"cd {device_dir} && "
            f"export LD_LIBRARY_PATH={device_dir}:$LD_LIBRARY_PATH && "
            f"export ADSP_LIBRARY_PATH='{device_dir};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && "
            f"{DEVICE_RUNNER} \"{prompt}\""
        )

        exec_start = time.perf_counter()
        ok_run, stdout, stderr = run_adb(cmd, timeout=600)
        exec_time = time.perf_counter() - exec_start

        if not ok_run:
            print(f"   ❌ Inference failed for pair {pair_idx} (took {exec_time:.2f}s).")
            if stderr:
                print(f"      Stderr: {stderr.strip()[:300]}")
            if stdout:
                print(f"      Stdout: {stdout.strip()[:300]}")
            continue

        # 4. Pull output image
        pull_start = time.perf_counter()
        ok_pull, _, _ = run_adb(["pull", DEVICE_OUTPUT_PNG, local_output_png])
        pull_time = time.perf_counter() - pull_start

        if ok_pull and os.path.isfile(local_output_png):
            file_size_kb = os.path.getsize(local_output_png) / 1024.0
            print(f"   ✅ Saved: {os.path.basename(local_output_png)} ({file_size_kb:.1f} KB)")
            print(f"   ⏱️  Timings: Exec={exec_time:.2f}s | Push={push_time:.2f}s | Pull={pull_time:.2f}s")
            latency_records.append({
                "Index": pair_idx,
                "Output_File": os.path.basename(local_output_png),
                "Prompt": prompt,
                "Inference_Latency_sec": round(exec_time, 3),
                "Push_Latency_sec": round(push_time, 3),
                "Pull_Latency_sec": round(pull_time, 3),
                "Status": "SUCCESS"
            })
        else:
            print(f"   ❌ Failed to pull {DEVICE_OUTPUT_PNG} for pair {pair_idx}.")
            latency_records.append({
                "Index": pair_idx,
                "Output_File": os.path.basename(local_output_png),
                "Prompt": prompt,
                "Inference_Latency_sec": round(exec_time, 3),
                "Push_Latency_sec": round(push_time, 3),
                "Pull_Latency_sec": round(pull_time, 3),
                "Status": "PULL_FAILED"
            })

        # 5. Clean intermediate step images on device to preserve storage
        if clean_device_steps:
            run_adb(f"rm -f {device_dir}/step_*.png {device_dir}/sd_output.png")

    total_time = time.perf_counter() - total_start

    # Save latency report
    report_path = os.path.join(output_dir, "sd_latency.csv")
    if latency_records:
        avg_latency = np.mean([r["Inference_Latency_sec"] for r in latency_records if r["Status"] == "SUCCESS"])
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            fields = ["Index", "Output_File", "Prompt", "Inference_Latency_sec", "Push_Latency_sec", "Pull_Latency_sec", "Status"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(latency_records)
            writer.writerow({
                "Index": "AVERAGE",
                "Output_File": f"{len(latency_records)} samples",
                "Prompt": "-",
                "Inference_Latency_sec": round(float(avg_latency), 3),
                "Push_Latency_sec": "-",
                "Pull_Latency_sec": "-",
                "Status": "-"
            })

    print("\n" + "=" * 65)
    print("                     BATCH EXECUTION SUMMARY                     ")
    print("=" * 65)
    successful = sum(1 for r in latency_records if r["Status"] == "SUCCESS")
    print(f"  Total Processed Pairs    : {len(selected_pairs)}")
    print(f"  Successfully Completed   : {successful}/{len(selected_pairs)}")
    if successful > 0:
        avg_exec = np.mean([r["Inference_Latency_sec"] for r in latency_records if r["Status"] == "SUCCESS"])
        min_exec = np.min([r["Inference_Latency_sec"] for r in latency_records if r["Status"] == "SUCCESS"])
        max_exec = np.max([r["Inference_Latency_sec"] for r in latency_records if r["Status"] == "SUCCESS"])
        print(f"  Avg Inference Latency    : {avg_exec:.2f} s / sample")
        print(f"  Min Inference Latency    : {min_exec:.2f} s")
        print(f"  Max Inference Latency    : {max_exec:.2f} s")
    print(f"  Total Batch Elapsed Time : {total_time:.2f} s")
    print(f"  Latency Report Saved To  : {report_path}")
    print(f"  Outputs Saved To         : {results_dir}")
    print("=" * 65 + "\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Stable Diffusion 1.5 RePaint Inpainting Batch Benchmark on Snapdragon 8 Elite."
    )
    parser.add_argument("--prompt", "-p", type=str, default="A cinematic shot of sunset",
                        help="Text prompt passed to sd_qidk_runner_encoder (default: 'A cinematic shot of sunset')")
    parser.add_argument("--start", type=int, default=1, help="Start pair index (1 to 17, default: 1)")
    parser.add_argument("--end", type=int, default=17, help="End pair index (1 to 17, default: 17)")
    parser.add_argument("--input_dir", type=str, default=LOCAL_INPUT_DIR, help="Base input folder (contains image/ and mask/)")
    parser.add_argument("--output_dir", type=str, default=LOCAL_OUTPUT_DIR, help="Output directory for SD benchmark")
    parser.add_argument("--results_dir", type=str, default=LOCAL_RESULTS_DIR, help="Directory to pull results ({idx}_sd.png)")
    parser.add_argument("--zero_indexed", action="store_true",
                        help="Save outputs with 0-indexed naming (0_sd.png..16_sd.png) instead of 1-indexed (1_sd.png..17_sd.png)")
    parser.add_argument("--no_clean_steps", action="store_true",
                        help="Keep step_*.png debug images on device between iterations")
    parser.add_argument("--device_dir", type=str, default=DEVICE_SD_DIR,
                        help="Target directory on device with serialized binaries and runner")

    args = parser.parse_args()

    run_sd_batch(
        prompt=args.prompt,
        start_idx=args.start,
        end_idx=args.end,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        zero_indexed=args.zero_indexed,
        clean_device_steps=not args.no_clean_steps,
        device_dir=args.device_dir
    )
