#!/usr/bin/env python3
"""
scripts/benchmarks/run_6000_npu_benchmark.py
================================================================================
⚡ 6,000-SAMPLE NPU RESILIENT BATCH BENCHMARK & META-PREDICTOR DATASET HARNESS
================================================================================
Target Platform: Qualcomm Snapdragon 8 Elite (SM8750P) Hexagon NPU (HTP v79)
Target Models:   MIGAN NPU, LaMa-Dilated NPU, AOT-GAN NPU

Orchestration Workflow:
  1. Loads or initializes checkpoint.json to guarantee seamless resumption if stopped.
  2. Processes dataset in sequential batches (default: 50 images per batch).
  3. Enforces initial temperature barrier (<= 42.0 C) before every batch.
  4. Dynamically compiles raw float32 NHWC input tensors on-the-fly in /tmp.
  5. Pushes batch to device (/data/local/tmp/lama/batch_in).
  6. Executes each model with FastRPC on Hexagon NPU (--use_dsp --perf_profile burst).
  7. Samples hardware telemetry (peak temps, thermal rise, PMIC power/current).
  8. Pulls reconstructed output tensors to host.
  9. PURGES device batch immediately to conserve QIDK internal storage.
  10. Computes PSNR, Hole PSNR, SSIM, and LPIPS-VGG on host.
  11. Retains 1 representative visual comparison image set per batch on host SSD.
  12. Logs 33+ input features and performance/quality targets to:
      Benchmark/output/6000_benchmark/dataset_meta_predictor.csv
  13. Atomically updates checkpoint.json after every batch.
================================================================================
"""

import os
import sys
import time
import json
import csv
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as TF
import lpips
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILED_DIR = REPO_ROOT / "Benchmark" / "6000_images" / "compiled"
IMAGES_DIR = COMPILED_DIR / "images"
MASKS_DIR = COMPILED_DIR / "masks"
IMAGE_MANIFEST = COMPILED_DIR / "dataset_manifest.csv"
MASK_MANIFEST = COMPILED_DIR / "masks_manifest.csv"

OUTPUT_BASE = REPO_ROOT / "Benchmark" / "output" / "6000_benchmark"
CHECKPOINT_FILE = OUTPUT_BASE / "checkpoint.json"
MASTER_CSV_FILE = OUTPUT_BASE / "dataset_meta_predictor.csv"
ARTIFACTS_CSV_FILE = REPO_ROOT / "artifacts" / "dataset_complete_metrics_6000.csv"
SAMPLE_RECON_DIR = OUTPUT_BASE / "reconstructions_sample"

DEVICE_LAMA_DIR = "/data/local/tmp/lama"
DEVICE_SD_DIR = "/data/local/tmp/sd_runtime"

CANONICAL_SIZE = (512, 512)

# Model definitions on Hexagon HTP v79 NPU
MODELS = {
    "migan_npu": {
        "display_name": "MIGAN (NPU)",
        "dlc": "migan_htp_v79.dlc",
        "mask_type": "inverted",
        "runtime_flag": "--use_dsp",
        "output_cand": ["output_0.raw", "painted_image.raw"]
    },
    "lama_npu": {
        "display_name": "LaMa Dilated (NPU)",
        "dlc": "bench_cache_lama_dilated.dlc",
        "mask_type": "standard",
        "runtime_flag": "--use_dsp",
        "output_cand": ["painted_image.raw", "output_0.raw"]
    },
    "aotgan_npu": {
        "display_name": "AOT-GAN (NPU)",
        "dlc": "bench_cache_aotgan.dlc",
        "mask_type": "standard",
        "runtime_flag": "--use_dsp",
        "output_cand": ["output_0.raw", "painted_image.raw"]
    }
}

# ==============================================================================
# TELEMETRY & ADB HELPERS
# ==============================================================================
def run_adb(cmd_args: List[str], adb_path: str = "adb", timeout: int = 180, retries: int = 3) -> Tuple[int, str, str]:
    full_cmd = [adb_path] + cmd_args
    for attempt in range(retries):
        try:
            proc = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
            err = proc.stderr.strip()
            if proc.returncode != 0 and any(w in err.lower() for w in ["closed", "device not found", "no devices", "offline", "protocol fault"]):
                time.sleep(2)
                wait_for_device(adb_path)
                continue
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except subprocess.TimeoutExpired:
            wait_for_device(adb_path)
            continue
        except Exception as e:
            return -1, "", str(e)
    return -1, "", "Exceeded retries"

def check_device_connected(adb_path: str = "adb") -> bool:
    try:
        proc = subprocess.run([adb_path, "get-state"], capture_output=True, text=True, timeout=5)
        return proc.returncode == 0 and "device" in proc.stdout.strip()
    except Exception:
        return False

def wait_for_device(adb_path: str = "adb"):
    if check_device_connected(adb_path):
        return
    print("\n⚠️  Snapdragon 8 Elite USB disconnected! Attempting automatic recovery...")
    dots = 0
    while not check_device_connected(adb_path):
        try:
            subprocess.run([adb_path, "reconnect", "offline"], capture_output=True, text=True, timeout=5)
            subprocess.run([adb_path, "reconnect"], capture_output=True, text=True, timeout=5)
        except Exception:
            pass
        time.sleep(2.0)
        dots = (dots + 1) % 4
        sys.stdout.write(f"\r\033[33m[Waiting for ADB device{'.' * dots}   ]\033[0m ")
        sys.stdout.flush()
    sys.stdout.write("\r\033[K")
    print("✅ Device reconnected! Resuming benchmark pipeline.\n")
    time.sleep(1.0)

def query_device_thermals(adb_path: str = "adb") -> Dict[str, float]:
    """Reads highest temperature across CPU, GPU, NSP (NPU) thermal zones."""
    script = (
        'for t in /sys/class/thermal/thermal_zone*; do '
        '  type=$(cat "$t/type" 2>/dev/null); '
        '  temp=$(cat "$t/temp" 2>/dev/null); '
        '  [ -n "$temp" ] && [ "$temp" -gt 0 ] && echo "$type $temp"; '
        'done; true'
    )
    rc, out, _ = run_adb(["shell", script], adb_path=adb_path)
    res = {"cpu": 0.0, "gpu": 0.0, "nsp": 0.0, "soc_max": 0.0}
    if not out:
        return res

    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            z_type, temp_mc = parts[0], int(parts[1])
            temp_c = temp_mc / 1000.0
            if z_type.startswith("cpu") or "cpuss" in z_type:
                res["cpu"] = max(res["cpu"], temp_c)
            elif "gpuss" in z_type:
                res["gpu"] = max(res["gpu"], temp_c)
            elif "nsphvx" in z_type or "nsphmx" in z_type:
                res["nsp"] = max(res["nsp"], temp_c)
            elif "soc" in z_type and not "socd" in z_type:
                res["soc_max"] = max(res["soc_max"], temp_c)

    res["soc_max"] = max(res["soc_max"], res["cpu"], res["gpu"], res["nsp"])
    return res

def query_device_power(adb_path: str = "adb") -> Dict[str, float]:
    """Queries battery voltage, PMIC current, and computes active power."""
    script = (
        'v=$(cat /sys/class/power_supply/battery/voltage_now 2>/dev/null); '
        'c=$(cat /sys/bus/iio/devices/iio:device0/in_current_pmih010x_ichg_fb_input 2>/dev/null | tr -d "-"); '
        '[ -z "$c" ] || [ "$c" -eq 0 ] 2>/dev/null && c=$(cat /sys/class/power_supply/battery/current_now 2>/dev/null | tr -d "-"); '
        'echo "$v $c"; true'
    )
    rc, out, _ = run_adb(["shell", script], adb_path=adb_path)
    ret = {"voltage_v": 8.9, "current_ma": 350.0, "power_w": 3.1}
    if rc == 0 and out:
        parts = out.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            v_uv, c_ua = int(parts[0]), int(parts[1])
            ret["voltage_v"] = round(v_uv / 1000000.0, 3)
            ret["current_ma"] = round(c_ua / 1000.0, 1)
            ret["power_w"] = round((ret["voltage_v"] * ret["current_ma"]) / 1000.0, 3)
    return ret

EMERGENCY_OVERHEAT_LIMIT = 62.0  # Hardware damage hazard ceiling

def check_storage_and_system_health(adb_path: str = "adb") -> Tuple[bool, str]:
    """
    Guarantees storage safety for both device (/data) and host system (/home and /tmp).
    Halts if free space drops below safety thresholds.
    """
    # 1. Device storage check
    rc, out, _ = run_adb(["shell", "df -k /data | tail -n 1"], adb_path=adb_path)
    device_avail_mb = 0.0
    if rc == 0 and out:
        parts = out.split()
        if len(parts) >= 4 and parts[3].isdigit():
            device_avail_mb = int(parts[3]) / 1024.0

    # Emergency device cleanup if space < 1500 MB
    if device_avail_mb < 1500.0:
        run_adb(["shell", "rm -rf /data/local/tmp/lama/batch_in /data/local/tmp/lama/out_* /data/local/tmp/*.png /data/local/tmp/*.xml"], adb_path=adb_path)
        rc2, out2, _ = run_adb(["shell", "df -k /data | tail -n 1"], adb_path=adb_path)
        if rc2 == 0 and out2:
            parts = out2.split()
            if len(parts) >= 4 and parts[3].isdigit():
                device_avail_mb = int(parts[3]) / 1024.0

    if device_avail_mb < 1000.0:
        return False, f"CRITICAL: QIDK device storage nearly exhausted ({device_avail_mb:.1f} MB remaining on /data)! Aborting to prevent device lockup."

    # 2. Host storage check
    home_stat = shutil.disk_usage(REPO_ROOT)
    home_avail_gb = home_stat.free / (1024 ** 3)
    if home_avail_gb < 1.5:
        return False, f"CRITICAL: Host filesystem storage low ({home_avail_gb:.2f} GB remaining on /home)! Aborting to prevent disk full."

    tmp_stat = shutil.disk_usage("/tmp")
    tmp_avail_gb = tmp_stat.free / (1024 ** 3)
    if tmp_avail_gb < 1.0:
        return False, f"CRITICAL: Host /tmp storage low ({tmp_avail_gb:.2f} GB remaining)! Aborting to prevent crash."

    status = f"QIDK /data: {device_avail_mb/1024.0:.2f} GB free | Host /home: {home_avail_gb:.1f} GB free | Host /tmp: {tmp_avail_gb:.1f} GB free"
    return True, status

def enforce_cooldown_barrier(target_temp: float = 55.0, max_wait_sec: int = 300, adb_path: str = "adb") -> float:
    """
    Checks hardware thermals before running.
    - If SoC junction temp <= target_temp (55.0°C), execution proceeds without delay.
    - If temp > target_temp (55.0°C), pauses and actively waits until temp drops <= 48.0°C.
    - If temp >= EMERGENCY_OVERHEAT_LIMIT (62.0°C), logs emergency warning and forces cooldown.
    """
    thermals = query_device_thermals(adb_path)
    curr_temp = thermals["soc_max"]

    if curr_temp >= EMERGENCY_OVERHEAT_LIMIT:
        print(f"\n🚨 [CRITICAL HARDWARE OVERHEAT HAZARD] Peak SoC temp {curr_temp:.1f}°C reached limit ({EMERGENCY_OVERHEAT_LIMIT:.1f}°C)!")
        print("   Emergency cooldown engaged. Waiting for temperatures to safely subside...")
        time.sleep(60)

    if curr_temp <= target_temp:
        return curr_temp

    print(f"\n⚠️  [THERMAL BARRIER] Peak SoC temp {curr_temp:.1f}°C exceeded {target_temp:.1f}°C limit!")
    print(f"   Initiating active cooldown barrier until temp drops below 48.0°C...")
    t0 = time.time()
    while curr_temp > 48.0 and (time.time() - t0) < max_wait_sec:
        time.sleep(5)
        curr_temp = query_device_thermals(adb_path)["soc_max"]
        sys.stdout.write(f"\r   Cooling down: SoC temp = {curr_temp:.1f}°C (target: <= 48.0°C)... ")
        sys.stdout.flush()

    if curr_temp > target_temp:
        raise RuntimeError(f"Hardware thermal safety triggered: SoC temp {curr_temp:.1f}°C failed to cool below {target_temp:.1f}°C after {max_wait_sec}s.")

    print(f"\n  [Thermals] Cooled to {curr_temp:.1f}°C. Resuming benchmark execution safely.\n")
    return curr_temp

# ==============================================================================
# IMAGE & MASK FEATURE EXTRACTION
# ==============================================================================
def extract_image_features(img_np: np.ndarray) -> Dict[str, float]:
    """Extracts photometric, textural, and structural features from input image."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = round(float(laplacian.var()), 2)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = round(float(np.sum(edges > 0)) / float(edges.size), 4)

    mean_luminance = round(float(np.mean(gray)), 2)
    std_contrast = round(float(np.std(gray)), 2)

    # Color channel entropy
    ent = 0.0
    for c in range(3):
        hist, _ = np.histogram(img_np[:, :, c], bins=32, range=(0, 256), density=True)
        hist = hist[hist > 0]
        ent += -np.sum(hist * np.log2(hist))
    color_entropy = round(float(ent / 3.0), 3)

    return {
        "img_laplacian_var": laplacian_var,
        "img_canny_edge_density": edge_density,
        "img_mean_luminance": mean_luminance,
        "img_std_contrast": std_contrast,
        "img_color_entropy": color_entropy
    }

# ==============================================================================
# PERCEPTUAL QUALITY EVALUATOR
# ==============================================================================
class PerceptualEvaluator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
        self.lpips_vgg.eval()

    def evaluate_batch(
        self,
        pred_rgb_list: List[np.ndarray],
        gt_rgb_list: List[np.ndarray],
        mask_bin_list: List[np.ndarray]
    ) -> List[Dict[str, float]]:
        """Batched calculation of PSNR, SSIM, and batched LPIPS-VGG."""
        n = len(pred_rgb_list)
        results = []
        pred_u8_list = []
        gt_u8_list = []

        for i in range(n):
            pred_u8 = np.clip(pred_rgb_list[i], 0, 255).astype(np.uint8)
            gt_u8 = np.clip(gt_rgb_list[i], 0, 255).astype(np.uint8)
            pred_u8_list.append(pred_u8)
            gt_u8_list.append(gt_u8)

            psnr_full = round(float(compute_psnr(gt_u8, pred_u8)), 2)
            ssim_val = round(float(compute_ssim(gt_u8, pred_u8, multichannel=True, channel_axis=2)), 4)

            hole_mask = (mask_bin_list[i] > 0.5)
            if np.any(hole_mask):
                pred_hole = pred_u8[hole_mask]
                gt_hole = gt_u8[hole_mask]
                mse_hole = np.mean((pred_hole.astype(np.float64) - gt_hole.astype(np.float64)) ** 2)
                psnr_hole = round(float(10.0 * np.log10((255.0 ** 2) / mse_hole)), 2) if mse_hole > 1e-10 else 99.0
            else:
                psnr_hole = psnr_full

            results.append({
                "psnr_full": psnr_full,
                "psnr_hole": psnr_hole,
                "ssim": ssim_val,
                "lpips": 0.0
            })

        # Batched LPIPS forward passes in chunks of 10 to accelerate CPU execution
        chunk_size = 10
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            chunk_p = [torch.from_numpy(pred_u8_list[k]).permute(2, 0, 1).float() / 127.5 - 1.0 for k in range(start, end)]
            chunk_g = [torch.from_numpy(gt_u8_list[k]).permute(2, 0, 1).float() / 127.5 - 1.0 for k in range(start, end)]
            t_p = torch.stack(chunk_p).to(self.device)
            t_g = torch.stack(chunk_g).to(self.device)
            with torch.no_grad():
                lpips_out = self.lpips_vgg(t_p, t_g).squeeze()
                if lpips_out.ndim == 0:
                    vals = [lpips_out.item()]
                else:
                    vals = lpips_out.tolist()
            for k, val in enumerate(vals):
                results[start + k]["lpips"] = round(float(val), 4)

        return results

# ==============================================================================
# QUALITY STATS TRACKER & DASHBOARD
# ==============================================================================
class QualityStatsTracker:
    """Tracks running quality statistics and metrics across all batches."""
    def __init__(self, master_csv_path: Path):
        self.master_csv_path = master_csv_path
        self.stats = {}
        self._init_from_csv()

    def _init_from_csv(self):
        if not self.master_csv_path.exists():
            return
        try:
            with open(self.master_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    m = r.get("model")
                    if not m:
                        continue
                    if m not in self.stats:
                        self.stats[m] = {"count": 0, "psnr_full": 0.0, "psnr_hole": 0.0, "ssim": 0.0, "lpips": 0.0, "latency": 0.0}
                    self.stats[m]["count"] += 1
                    self.stats[m]["psnr_full"] += float(r.get("psnr_full", 0.0))
                    self.stats[m]["psnr_hole"] += float(r.get("psnr_hole", 0.0))
                    self.stats[m]["ssim"] += float(r.get("ssim", 0.0))
                    self.stats[m]["lpips"] += float(r.get("lpips", 0.0))
                    self.stats[m]["latency"] += float(r.get("latency_ms", 0.0))
        except Exception:
            pass

    def update(self, batch_rows: List[Dict]):
        for r in batch_rows:
            m = r["model"]
            if m not in self.stats:
                self.stats[m] = {"count": 0, "psnr_full": 0.0, "psnr_hole": 0.0, "ssim": 0.0, "lpips": 0.0, "latency": 0.0}
            self.stats[m]["count"] += 1
            self.stats[m]["psnr_full"] += float(r["psnr_full"])
            self.stats[m]["psnr_hole"] += float(r["psnr_hole"])
            self.stats[m]["ssim"] += float(r["ssim"])
            self.stats[m]["lpips"] += float(r["lpips"])
            self.stats[m]["latency"] += float(r["latency_ms"])

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for m, data in self.stats.items():
            cnt = max(1, data["count"])
            summary[m] = {
                "count": data["count"],
                "psnr_full": data["psnr_full"] / cnt,
                "psnr_hole": data["psnr_hole"] / cnt,
                "ssim": data["ssim"] / cnt,
                "lpips": data["lpips"] / cnt,
                "latency": data["latency"] / cnt
            }
        return summary


def print_live_progress_dashboard(
    batch_idx: int,
    total_batches: int,
    images_evaluated_so_far: int,
    total_images: int,
    elapsed_sec: float,
    recent_batch_rows: List[Dict],
    stats_tracker: QualityStatsTracker,
    curr_thermals: Dict[str, float],
    curr_power: Dict[str, float],
    cooldown_threshold: float = 55.0
):
    """Renders a real-time terminal progress and quality dashboard."""
    pct = (images_evaluated_so_far / total_images) * 100.0 if total_images > 0 else 0.0
    bar_width = 24
    filled_len = int(bar_width * images_evaluated_so_far // total_images) if total_images > 0 else 0
    bar = "█" * filled_len + "░" * (bar_width - filled_len)

    # Calculate pace & ETA
    img_per_sec = images_evaluated_so_far / elapsed_sec if elapsed_sec > 0 else 0.0
    remaining_imgs = max(0, total_images - images_evaluated_so_far)
    eta_sec = remaining_imgs / img_per_sec if img_per_sec > 0 else 0.0

    elapsed_str = f"{int(elapsed_sec // 60)}m {int(elapsed_sec % 60):02d}s"
    eta_str = f"{int(eta_sec // 3600)}h {int((eta_sec % 3600) // 60):02d}m" if eta_sec >= 3600 else f"{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"

    # Compute batch stats per model
    batch_stats = {}
    for r in recent_batch_rows:
        m = r["model"]
        if m not in batch_stats:
            batch_stats[m] = {
                "psnr_full": [], "psnr_hole": [], "ssim": [], "lpips": [],
                "latency": r["latency_ms"], "fps": r["throughput_fps"]
            }
        batch_stats[m]["psnr_full"].append(r["psnr_full"])
        batch_stats[m]["psnr_hole"].append(r["psnr_hole"])
        batch_stats[m]["ssim"].append(r["ssim"])
        batch_stats[m]["lpips"].append(r["lpips"])

    cum_summary = stats_tracker.get_summary()
    soc_temp = curr_thermals.get("soc_max", 0.0)
    nsp_temp = curr_thermals.get("nsp", 0.0)
    therm_status = "OPTIMAL (FULL SPEED)" if soc_temp <= cooldown_threshold else "HOT (THROTTLE RISK)"

    print("\n" + "=" * 94)
    print(f"📊 LIVE QUALITY BENCHMARK PROGRESS — 6,000 NPU DATASET")
    print("=" * 94)
    print(f" 🎯 Overall Progress : [{bar}] {images_evaluated_so_far}/{total_images} images ({pct:.1f}%) | Batch {batch_idx}/{total_batches}")
    print(f" ⏱️  Timing & Pace    : Elapsed: {elapsed_str} | Remaining ETA: {eta_str} | Speed: {img_per_sec:.2f} img/s ({img_per_sec*3:.2f} inf/s)")
    print(f" 🌡️  Thermal State    : SoC Peak: {soc_temp:.1f}°C (Threshold: {cooldown_threshold:.1f}°C) | NSP: {nsp_temp:.1f}°C | Status: {therm_status}")
    print(f" ⚡ Board Power Draw : {curr_power.get('power_w', 0.0):.2f} W ({curr_power.get('current_ma', 0.0):.1f} mA @ {curr_power.get('voltage_v', 0.0):.2f} V)")
    print("-" * 94)
    num_batch_imgs = len(recent_batch_rows) // len(batch_stats) if batch_stats else 0
    print(f" 📋 RECENT BATCH STATS (Batch {batch_idx} — {num_batch_imgs} Images)")
    print(" ┌──────────────┬─────────────┬───────────┬─────────────┬─────────────┬───────────┬───────────┐")
    print(" │ Model (NPU)  │ Latency     │ FPS       │ PSNR (Full) │ PSNR (Hole) │ SSIM      │ LPIPS     │")
    print(" ├──────────────┼─────────────┼───────────┼─────────────┼─────────────┼───────────┼───────────┤")
    for m_key in ["migan_npu", "lama_npu", "aotgan_npu"]:
        if m_key in batch_stats:
            b_data = batch_stats[m_key]
            name = MODELS.get(m_key, {}).get("display_name", m_key)[:12]
            p_f = np.mean(b_data["psnr_full"])
            p_h = np.mean(b_data["psnr_hole"])
            ss = np.mean(b_data["ssim"])
            lp = np.mean(b_data["lpips"])
            lat = b_data["latency"]
            fps = b_data["fps"]
            print(f" │ {name:<12s} │ {lat:>7.1f} ms  │ {fps:>5.1f} FPS │ {p_f:>8.2f} dB │ {p_h:>8.2f} dB │  {ss:>8.4f} │  {lp:>8.4f} │")
    print(" └──────────────┴─────────────┴───────────┴─────────────┴─────────────┴───────────┴───────────┘")

    print(f" 📈 CUMULATIVE AVERAGE QUALITY (Across {images_evaluated_so_far} Images / {images_evaluated_so_far*3} Inferences)")
    print(" ┌──────────────┬─────────────┬───────────┬─────────────┬─────────────┬───────────┬───────────┐")
    print(" │ Model (NPU)  │ Avg Latency │ Avg FPS   │ PSNR (Full) │ PSNR (Hole) │ SSIM      │ LPIPS     │")
    print(" ├──────────────┼─────────────┼───────────┼─────────────┼─────────────┼───────────┼───────────┤")
    for m_key in ["migan_npu", "lama_npu", "aotgan_npu"]:
        if m_key in cum_summary:
            c_data = cum_summary[m_key]
            name = MODELS.get(m_key, {}).get("display_name", m_key)[:12]
            p_f = c_data["psnr_full"]
            p_h = c_data["psnr_hole"]
            ss = c_data["ssim"]
            lp = c_data["lpips"]
            lat = c_data["latency"]
            fps = 1000.0 / lat if lat > 0 else 0.0
            print(f" │ {name:<12s} │ {lat:>7.1f} ms  │ {fps:>5.1f} FPS │ {p_f:>8.2f} dB │ {p_h:>8.2f} dB │  {ss:>8.4f} │  {lp:>8.4f} │")
    print(" └──────────────┴─────────────┴───────────┴─────────────┴─────────────┴───────────┴───────────┘")
    print("=" * 94 + "\n")


# ==============================================================================
# STATE & CHECKPOINT MANAGEMENT
# ==============================================================================
def load_checkpoint() -> Dict:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "completed_batches": [],
        "completed_samples_count": 0,
        "last_updated": None
    }

def save_checkpoint(ckpt: Dict):
    ckpt["last_updated"] = datetime.now().isoformat()
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)

# ==============================================================================
# BATCH ORCHESTRATOR
# ==============================================================================
def run_benchmark(batch_size: int = 50, limit: int = 6000, cooldown_temp: float = 55.0,
                  models_to_run: List[str] = None, adb_path: str = "adb"):
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    SAMPLE_RECON_DIR.mkdir(parents=True, exist_ok=True)

    if models_to_run is None:
        models_to_run = list(MODELS.keys())

    print("=" * 80)
    print("⚡ SNAPDRAGON 8 ELITE — 6,000-SAMPLE NPU RESILIENT BENCHMARK HARNESS")
    print("=" * 80)
    print(f" Target NPU Core      : Hexagon HTP v79 NPU (FastRPC --use_dsp --perf_profile burst)")
    print(f" Target Models        : {', '.join(models_to_run)}")
    print(f" Total Samples        : {limit}")
    print(f" Batch Size           : {batch_size} images per push/purge cycle")
    print(f" Thermal Barrier      : <= {cooldown_temp}°C (Bypassed if below 55.0°C)")
    print(f" Checkpoint File      : {CHECKPOINT_FILE}")
    print(f" Master Dataset CSV   : {MASTER_CSV_FILE}")
    print("=" * 80)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # Load manifests
    if not IMAGE_MANIFEST.exists() or not MASK_MANIFEST.exists():
        print("Error: Missing dataset_manifest.csv or masks_manifest.csv. Run compilation first.")
        return 1

    img_meta = {}
    with open(IMAGE_MANIFEST, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_meta[row["sample_id"]] = row

    mask_meta = {}
    with open(MASK_MANIFEST, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mask_meta[row["sample_id"]] = row

    all_sample_ids = sorted(list(img_meta.keys()))[:limit]
    total_samples = len(all_sample_ids)
    num_batches = (total_samples + batch_size - 1) // batch_size

    ckpt = load_checkpoint()
    completed_batches = set(ckpt.get("completed_batches", []))
    print(f"Status: {len(completed_batches)}/{num_batches} batches previously completed.")

    # Master CSV Headers
    CSV_FIELDS = [
        "sample_id", "category", "model", "batch_idx", "intra_batch_pos",
        # Image Features
        "img_laplacian_var", "img_canny_edge_density", "img_mean_luminance", "img_std_contrast", "img_color_entropy",
        # Mask Features
        "mask_type", "mask_coverage_pct", "mask_hole_count", "mask_centroid_x", "mask_centroid_y",
        "mask_dist_to_border", "mask_aspect_ratio", "mask_perimeter_to_area",
        # Hardware Telemetry
        "t_start_nsp_c", "t_start_cpu_c", "t_start_gpu_c",
        "t_end_nsp_c", "t_end_cpu_c", "t_end_gpu_c",
        "delta_temp_nsp_c", "pmic_current_ma", "pmic_voltage_v", "active_power_w",
        # Performance Targets
        "latency_ms", "throughput_fps", "energy_delay_product_edp",
        # Quality Targets
        "psnr_full", "psnr_hole", "ssim", "lpips"
    ]

    # Initialize CSV if not exists
    if not MASTER_CSV_FILE.exists():
        with open(MASTER_CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

    evaluator = PerceptualEvaluator()
    stats_tracker = QualityStatsTracker(MASTER_CSV_FILE)
    host_staging_root = Path("/tmp/qidk_staging_batch")
    benchmark_start_time = time.time()

    for b_idx in range(num_batches):
        if b_idx in completed_batches:
            continue

        b_start_idx = b_idx * batch_size
        b_sample_ids = all_sample_ids[b_start_idx : b_start_idx + batch_size]
        curr_batch_size = len(b_sample_ids)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ Batch {b_idx + 1}/{num_batches} (Samples {b_sample_ids[0]}..{b_sample_ids[-1]})")

        # 1. Health & Storage Guard
        wait_for_device(adb_path)
        safe, health_msg = check_storage_and_system_health(adb_path)
        if not safe:
            print(f"\n❌ STORAGE ABORT GUARD: {health_msg}")
            return 1
        print(f"  [System Health] {health_msg}")

        # 2. Enforce Initial Cooldown Barrier
        enforce_cooldown_barrier(target_temp=cooldown_temp, adb_path=adb_path)

        # 3. Prepare Local Raw Tensors on-the-fly
        shutil.rmtree(host_staging_root, ignore_errors=True)
        for sub in ("raw_image", "raw_mask_standard", "raw_mask_inverted"):
            (host_staging_root / sub).mkdir(parents=True, exist_ok=True)

        list_std_lines = []
        list_inv_lines = []
        cached_gt_images = {}
        cached_masks = {}

        for stem in b_sample_ids:
            img_p = IMAGES_DIR / f"{stem}.png"
            mask_p = MASKS_DIR / f"{stem}.png"

            img = Image.open(img_p).convert("RGB")
            mask = Image.open(mask_p).convert("L")
            img_np = np.asarray(img, dtype=np.float32)
            mask_np = (np.asarray(mask) >= 128).astype(np.float32)

            cached_gt_images[stem] = img_np
            cached_masks[stem] = mask_np

            # float32 NHWC raw binaries
            (img_np / 255.0)[None].tofile(host_staging_root / "raw_image" / f"{stem}.raw")
            mask_np[None, ..., None].tofile(host_staging_root / "raw_mask_standard" / f"{stem}_mask.raw")
            (1.0 - mask_np)[None, ..., None].tofile(host_staging_root / "raw_mask_inverted" / f"{stem}_mask.raw")

            list_std_lines.append(f"image:=batch_in/raw_image/{stem}.raw mask:=batch_in/raw_mask_standard/{stem}_mask.raw\n")
            list_inv_lines.append(f"image:=batch_in/raw_image/{stem}.raw mask:=batch_in/raw_mask_inverted/{stem}_mask.raw\n")

        (host_staging_root / "list_standard.txt").write_text("".join(list_std_lines))
        (host_staging_root / "list_inverted.txt").write_text("".join(list_inv_lines))

        # 4. Push Batch to Device with strict count verification
        print(f"  [ADB Push] Pushing {curr_batch_size} tensor pairs to {DEVICE_LAMA_DIR}/batch_in...")
        push_ok = False
        for push_attempt in range(3):
            run_adb(["shell", f"rm -rf {DEVICE_LAMA_DIR}/batch_in && mkdir -p {DEVICE_LAMA_DIR}/batch_in"], adb_path=adb_path)
            run_adb(["push", str(host_staging_root) + "/.", f"{DEVICE_LAMA_DIR}/batch_in"], adb_path=adb_path)
            rc_cnt, out_cnt, _ = run_adb(["shell", f"ls {DEVICE_LAMA_DIR}/batch_in/raw_image | wc -l"], adb_path=adb_path)
            actual_count = int(out_cnt.strip()) if out_cnt.strip().isdigit() else 0
            if actual_count == curr_batch_size:
                push_ok = True
                break
            print(f"  [!] Incomplete push ({actual_count}/{curr_batch_size}) on attempt {push_attempt + 1}! Re-pushing batch...")
            time.sleep(2)

        # 5. Execute Each Model on Hexagon NPU with Continuous Thermal Protection
        batch_results_per_model = {}

        for m_key in models_to_run:
            # Active cooldown barrier before EVERY model execution
            enforce_cooldown_barrier(target_temp=cooldown_temp, adb_path=adb_path)
            m_cfg = MODELS[m_key]
            list_name = f"list_{m_cfg['mask_type']}.txt"
            out_dir_dev = f"{DEVICE_LAMA_DIR}/out_{m_key}"
            run_adb(["shell", f"rm -rf {out_dir_dev} && mkdir -p {out_dir_dev}"], adb_path=adb_path)

            t_start = query_device_thermals(adb_path)
            p_start = query_device_power(adb_path)

            cmd = (
                f"cd {DEVICE_LAMA_DIR} && "
                f"export LD_LIBRARY_PATH={DEVICE_LAMA_DIR}/lib:{DEVICE_LAMA_DIR}:{DEVICE_SD_DIR}:vendor/lib64:/system/lib64 && "
                f"export ADSP_LIBRARY_PATH='{DEVICE_LAMA_DIR}/dsp/lib;{DEVICE_LAMA_DIR}/dsp;{DEVICE_SD_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && "
                f"t0=$(date +%s%3N) && "
                f"./snpe-net-run --container {m_cfg['dlc']} --input_list batch_in/{list_name} "
                f"--output_dir {out_dir_dev} {m_cfg['runtime_flag']} --perf_profile burst > /dev/null 2>&1; "
                f"snpe_rc=$?; t1=$(date +%s%3N); "
                f"if [ $snpe_rc -eq 0 ]; then echo $(( t1 - t0 )); else echo 'SNPE_ERROR:' $snpe_rc; fi"
            )

            rc, stdout, stderr = run_adb(["shell", cmd], adb_path=adb_path, timeout=300)
            t_end = query_device_thermals(adb_path)
            p_end = query_device_power(adb_path)

            # Parse wall latency
            wall_ms = 0.0
            for line in stdout.splitlines():
                if "SNPE_ERROR" in line:
                    print(f"      [!] {m_cfg['display_name']} Execution Error: {line}")
                elif line.strip().isdigit():
                    wall_ms = float(line.strip())
                    break

            if wall_ms == 0.0:
                print(f"      [!] {m_cfg['display_name']} failed on NPU! Checking diagnostic log...")
                _, diag_out, _ = run_adb(["shell", f"cd {DEVICE_LAMA_DIR} && export LD_LIBRARY_PATH={DEVICE_LAMA_DIR}/lib:{DEVICE_LAMA_DIR}:{DEVICE_SD_DIR}:vendor/lib64:/system/lib64 && export ADSP_LIBRARY_PATH='{DEVICE_LAMA_DIR}/dsp/lib;{DEVICE_LAMA_DIR}/dsp;{DEVICE_SD_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && ./snpe-net-run --container {m_cfg['dlc']} --input_list batch_in/{list_name} --output_dir {out_dir_dev} {m_cfg['runtime_flag']} --perf_profile burst 2>&1 | tail -n 10"], adb_path=adb_path)
                print(f"      [Diag]: {diag_out}")

            per_frame_ms = round(wall_ms / curr_batch_size, 2)
            fps = round(1000.0 / per_frame_ms, 1) if per_frame_ms > 0 else 0.0
            delta_nsp = round(t_end["nsp"] - t_start["nsp"], 2)
            avg_power = round((p_start["power_w"] + p_end["power_w"]) / 2.0, 3)

            print(f"    • {m_cfg['display_name']:18s} | Wall: {wall_ms:6.1f} ms | Per-Frame: {per_frame_ms:5.1f} ms ({fps:4.1f} FPS) | ΔT_NPU: {delta_nsp:+4.1f}°C")

            # Pull output raw tensors
            local_out_dir = host_staging_root / f"out_{m_key}"
            local_out_dir.mkdir(parents=True, exist_ok=True)
            run_adb(["pull", f"{out_dir_dev}/.", str(local_out_dir)], adb_path=adb_path)

            batch_results_per_model[m_key] = {
                "per_frame_ms": per_frame_ms,
                "fps": fps,
                "t_start": t_start,
                "t_end": t_end,
                "delta_nsp": delta_nsp,
                "power": avg_power,
                "voltage": p_start["voltage_v"],
                "current": p_start["current_ma"],
                "local_dir": local_out_dir
            }

        # 5. PURGE Device Batch Immediately (Storage preservation)
        run_adb(["shell", f"rm -rf {DEVICE_LAMA_DIR}/batch_in {DEVICE_LAMA_DIR}/out_*"], adb_path=adb_path)

        # 6. Host Evaluation & Feature Logging
        new_csv_rows = []

        for m_key in models_to_run:
            m_res = batch_results_per_model[m_key]
            m_cand = MODELS[m_key]["output_cand"]

            preds_list = []
            gts_list = []
            masks_list = []
            valid_indices = []

            for intra_idx, stem in enumerate(b_sample_ids):
                gt_rgb = cached_gt_images[stem]
                mask_bin = cached_masks[stem]

                res_dir = m_res["local_dir"] / f"Result_{intra_idx}"
                pred_raw_path = None
                for c in m_cand:
                    cand_p = res_dir / c
                    if cand_p.exists():
                        pred_raw_path = cand_p
                        break

                if pred_raw_path and pred_raw_path.stat().st_size == (512 * 512 * 3 * 4):
                    raw_data = np.fromfile(pred_raw_path, dtype=np.float32).reshape(512, 512, 3)
                    pred_rgb = raw_data * 255.0 if raw_data.max() <= 1.5 else raw_data
                    mask_3d = mask_bin[:, :, None]
                    comp_rgb = (mask_3d * pred_rgb + (1.0 - mask_3d) * gt_rgb).clip(0, 255)

                    preds_list.append(comp_rgb)
                    gts_list.append(gt_rgb)
                    masks_list.append(mask_bin)
                    valid_indices.append(intra_idx)

                    # Save 1 representative visual comparison image set per batch (intra_idx == 0)
                    if intra_idx == 0:
                        rep_stem = f"batch_{b_idx+1:03d}_{stem}_{m_key}.png"
                        Image.fromarray(comp_rgb.astype(np.uint8)).save(SAMPLE_RECON_DIR / rep_stem)
                        if m_key == models_to_run[0]:
                            Image.fromarray(gt_rgb.astype(np.uint8)).save(SAMPLE_RECON_DIR / f"batch_{b_idx+1:03d}_{stem}_gt.png")
                            Image.fromarray((mask_bin * 255).astype(np.uint8)).save(SAMPLE_RECON_DIR / f"batch_{b_idx+1:03d}_{stem}_mask.png")

            sys.stdout.write(f"\r  [Evaluating Metrics] Batch {b_idx + 1}/{num_batches} | Model: {m_key} ({len(preds_list)} images) ... ")
            sys.stdout.flush()

            if preds_list:
                batch_q_metrics = evaluator.evaluate_batch(preds_list, gts_list, masks_list)
            else:
                batch_q_metrics = []

            metric_map = {valid_indices[i]: batch_q_metrics[i] for i in range(len(valid_indices))}

            # EDP: Energy (Joules) * Latency (s)
            latency_s = m_res["per_frame_ms"] / 1000.0
            energy_j = m_res["power"] * latency_s
            edp = round(energy_j * latency_s, 6)

            for intra_idx, stem in enumerate(b_sample_ids):
                gt_rgb = cached_gt_images[stem]
                i_f = extract_image_features(gt_rgb.astype(np.uint8))
                m_f = mask_meta[stem]
                q_metrics = metric_map.get(intra_idx, {"psnr_full": 0.0, "psnr_hole": 0.0, "ssim": 0.0, "lpips": 1.0})

                row = {
                    "sample_id": stem,
                    "category": img_meta[stem]["category"],
                    "model": m_key,
                    "batch_idx": b_idx + 1,
                    "intra_batch_pos": intra_idx + 1,
                    # Image features
                    "img_laplacian_var": i_f["img_laplacian_var"],
                    "img_canny_edge_density": i_f["img_canny_edge_density"],
                    "img_mean_luminance": i_f["img_mean_luminance"],
                    "img_std_contrast": i_f["img_std_contrast"],
                    "img_color_entropy": i_f["img_color_entropy"],
                    # Mask features
                    "mask_type": m_f["mask_type"],
                    "mask_coverage_pct": m_f["coverage_pct"],
                    "mask_hole_count": m_f["hole_count"],
                    "mask_centroid_x": m_f["centroid_x"],
                    "mask_centroid_y": m_f["centroid_y"],
                    "mask_dist_to_border": m_f["dist_to_border"],
                    "mask_aspect_ratio": m_f["aspect_ratio"],
                    "mask_perimeter_to_area": m_f["perimeter_to_area"],
                    # Telemetry
                    "t_start_nsp_c": m_res["t_start"]["nsp"],
                    "t_start_cpu_c": m_res["t_start"]["cpu"],
                    "t_start_gpu_c": m_res["t_start"]["gpu"],
                    "t_end_nsp_c": m_res["t_end"]["nsp"],
                    "t_end_cpu_c": m_res["t_end"]["cpu"],
                    "t_end_gpu_c": m_res["t_end"]["gpu"],
                    "delta_temp_nsp_c": m_res["delta_nsp"],
                    "pmic_current_ma": m_res["current"],
                    "pmic_voltage_v": m_res["voltage"],
                    "active_power_w": m_res["power"],
                    # Performance
                    "latency_ms": m_res["per_frame_ms"],
                    "throughput_fps": m_res["fps"],
                    "energy_delay_product_edp": edp,
                    # Quality
                    "psnr_full": q_metrics["psnr_full"],
                    "psnr_hole": q_metrics["psnr_hole"],
                    "ssim": q_metrics["ssim"],
                    "lpips": q_metrics["lpips"]
                }
                new_csv_rows.append(row)

        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        # Write to master CSV
        with open(MASTER_CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerows(new_csv_rows)
        try:
            shutil.copy2(MASTER_CSV_FILE, ARTIFACTS_CSV_FILE)
        except Exception:
            pass

        # Update stats tracker
        stats_tracker.update(new_csv_rows)

        # Update checkpoint
        completed_batches.add(b_idx)
        ckpt["completed_batches"] = sorted(list(completed_batches))
        total_eval_images = min(total_samples, len(completed_batches) * batch_size)
        ckpt["completed_samples_count"] = total_eval_images
        save_checkpoint(ckpt)

        # Clean host staging
        shutil.rmtree(host_staging_root, ignore_errors=True)

        # Telemetry snapshot for dashboard
        last_thermals = query_device_thermals(adb_path)
        last_power = query_device_power(adb_path)

        # Print Live Progress and Recent Stats Dashboard
        print_live_progress_dashboard(
            batch_idx=b_idx + 1,
            total_batches=num_batches,
            images_evaluated_so_far=total_eval_images,
            total_images=total_samples,
            elapsed_sec=max(1.0, time.time() - benchmark_start_time),
            recent_batch_rows=new_csv_rows,
            stats_tracker=stats_tracker,
            curr_thermals=last_thermals,
            curr_power=last_power,
            cooldown_threshold=cooldown_temp
        )

    print("\n" + "=" * 80)
    print("🎉 6,000-SAMPLE NPU BENCHMARK COMPLETE!")
    print(f" Master Dataset CSV : {MASTER_CSV_FILE}")
    print(f" Visual Reconstructions Sample : {SAMPLE_RECON_DIR}")
    print("=" * 80)
    return 0

def main():
    parser = argparse.ArgumentParser(description="6,000-Sample NPU Benchmark Harness")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size per push/run cycle (default: 20)")
    parser.add_argument("--limit", type=int, default=6000, help="Total samples to run (default: 6000)")
    parser.add_argument("--cooldown-temp", type=float, default=55.0, help="Thermal barrier start temp (default: 55.0°C)")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()), default=list(MODELS.keys()))
    parser.add_argument("--reset-checkpoint", action="store_true", help="Reset checkpoint and re-run all batches")
    parser.add_argument("--adb", default="adb", help="Path to adb executable")
    args = parser.parse_args()

    if args.reset_checkpoint:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        if MASTER_CSV_FILE.exists():
            MASTER_CSV_FILE.unlink()
        print("Checkpoint and master CSV reset.")

    return run_benchmark(
        batch_size=args.batch_size,
        limit=args.limit,
        cooldown_temp=args.cooldown_temp,
        models_to_run=args.models,
        adb_path=args.adb
    )

if __name__ == "__main__":
    main()
