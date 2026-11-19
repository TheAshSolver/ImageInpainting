#!/usr/bin/env python3
"""
apps/qidk-load-generator/run_real_contention_benchmark.py
================================================================================
⚡ QUALCOMM SNAPDRAGON 8 ELITE CLOSED-LOOP CONTENTION HARNESS (L0 TO L4)
================================================================================
Runs on-device multi-workload contention across levels L0 to L4:
  - L0: Baseline Idle (0% CPU, 0 MB RAM)
  - L1: Light Multi-Tasking (15% CPU, 1024 MB LPDDR5x RAM)
  - L2: Medium Workload (35% CPU, 2560 MB RAM)
  - L3: Heavy Vision Pipeline (65% CPU, 4096 MB RAM)
  - L4: Extreme Thermal & Bus Saturation (85% CPU, 4096 MB RAM)

Strict Guardrails (GEMINI.md):
  - 100% measured on hardware (SM8750 / 8f27557f). Zero synthetic interpolation.
  - Fail-fast assertions on sample counts, temperature validity, and zero NaNs.
  - Generates immutable raw log: data/raw/real_contention_levels_l0_l4_telemetry.csv
  - Generates processed report: data/processed/load_generator_calibration_report.csv
================================================================================
"""

import os
import sys
import time
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_RAW_CSV = RAW_DATA_DIR / "real_contention_levels_l0_l4_telemetry.csv"
OUT_REPORT_CSV = PROCESSED_DIR / "load_generator_calibration_report.csv"
CHECKPOINT_FILE = REPO_ROOT / "apps" / "qidk-load-generator" / "checkpoint_phase2.json"
DEVICE_SERIAL = "8f27557f"
DEVICE_CSV = "/data/local/tmp/real_contention_telemetry.csv"
DEVICE_BIN = "/data/local/tmp/qidk_load_generator"

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
TRIALS_PER_LEVEL = 3
TRIAL_DURATION_S = 10


def run_adb(cmd: str) -> str:
    full_cmd = f"adb -s {DEVICE_SERIAL} {cmd}"
    res = subprocess.run(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ADB command failed ({res.returncode}): {full_cmd}\nStderr: {res.stderr}")
    return res.stdout.strip()


def get_peak_cpu_temp() -> float:
    cmd = "shell \"for z in 1 2 4 5 6 7 9 11 13 15 16 17 18 20 24 26 27 28 29 30; do cat /sys/class/thermal/thermal_zone\\$z/temp 2>/dev/null; done\""
    out = run_adb(cmd)
    temps = [float(x) / 1000.0 for x in out.split() if x.isdigit()]
    return max(temps) if temps else 32.0


def wait_for_thermal_settling(max_temp_c: float = 45.0, timeout_s: int = 30):
    print(f"  [Cooling] Waiting for thermal settling below {max_temp_c}°C...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        t = get_peak_cpu_temp()
        if t <= max_temp_c:
            print(f"  [Cooling] Settled at {t:.1f}°C.")
            return
        time.sleep(2)
    print(f"  [Cooling] Settling timeout ({timeout_s}s reached). Current temp: {get_peak_cpu_temp():.1f}°C. Proceeding.")


def main():
    print("=" * 80)
    print("⚡ SNAPDRAGON 8 ELITE ON-DEVICE CONTENTION BENCHMARK (L0 - L4)")
    print(f"  Device: {DEVICE_SERIAL}")
    print(f"  Levels: {LEVELS} ({TRIALS_PER_LEVEL} trials x {TRIAL_DURATION_S}s each)")
    print("=" * 80)

    # 1. Verify device connection
    devs = subprocess.run("adb devices -l", shell=True, capture_output=True, text=True).stdout
    if DEVICE_SERIAL not in devs:
        raise RuntimeError(f"Device {DEVICE_SERIAL} not connected! Run 'adb devices -l'.")

    # 2. Clear old telemetry on device
    run_adb(f"shell rm -f {DEVICE_CSV}")

    # 3. Execute sweep across levels and trials
    total_runs = len(LEVELS) * TRIALS_PER_LEVEL
    current_run = 0

    for level in LEVELS:
        for trial in range(1, TRIALS_PER_LEVEL + 1):
            current_run += 1
            print(f"\n>>> [{current_run}/{total_runs}] Running Contention {level} | Trial {trial}/{TRIALS_PER_LEVEL}...")

            # Wait for thermal settling before each trial
            wait_for_thermal_settling(max_temp_c=44.0, timeout_s=20)

            # Run load generator
            cmd = f"shell {DEVICE_BIN} {level} {TRIAL_DURATION_S} {DEVICE_CSV} {trial}"
            out = run_adb(cmd)
            # Print last telemetry line
            lines = [l for l in out.splitlines() if "[TELEMETRY]" in l]
            if lines:
                print(f"    Last reading: {lines[-1]}")

    # 4. Pull telemetry CSV to host
    print(f"\n[ADB] Pulling {DEVICE_CSV} -> {OUT_RAW_CSV}...")
    run_adb(f"pull {DEVICE_CSV} {OUT_RAW_CSV}")
    assert OUT_RAW_CSV.exists(), f"Failed to pull raw CSV to {OUT_RAW_CSV}"

    # 5. Ingestion & Fail-Fast Validation (GEMINI.md Guardrail)
    df = pd.read_csv(OUT_RAW_CSV)
    print(f"\n[Audit] Loaded {len(df)} telemetry samples from {OUT_RAW_CSV}.")

    # Assertions
    assert len(df) >= 200, f"Sample count too low: {len(df)} < 200"
    assert not df.isna().any().any(), "Found NaN values in on-device telemetry!"
    assert (df['cpu_temp_c'] > 0).all(), "Found non-positive CPU temperatures!"
    assert (df['power_w'] > 0).all(), "Found non-positive power values!"
    for lvl in LEVELS:
        assert lvl in df['level'].values, f"Missing level {lvl} in telemetry!"
        lvl_df = df[df['level'] == lvl]
        for tr in range(1, TRIALS_PER_LEVEL + 1):
            assert tr in lvl_df['trial'].values, f"Missing trial {tr} for level {lvl}!"

    print("  ✓ All fail-fast integrity assertions PASSED!")

    # 6. Aggregate Achieved vs Target Calibration Report
    summary = []
    for lvl in LEVELS:
        sub = df[df['level'] == lvl]
        target_cpu = sub['target_cpu_pct'].iloc[0]
        achieved_cpu = sub['achieved_cpu_pct'].mean()
        achieved_cpu_std = sub['achieved_cpu_pct'].std()

        target_ram = sub['target_ram_mb'].iloc[0]
        achieved_ram = sub['allocated_ram_mb'].iloc[0]

        target_temp = sub['target_temp_c'].iloc[0]
        peak_cpu_temp = sub['cpu_temp_c'].max()
        mean_cpu_temp = sub['cpu_temp_c'].mean()
        mean_npu_temp = sub['npu_temp_c'].mean()
        mean_gpu_temp = sub['gpu_temp_c'].mean()

        mean_power = sub['power_w'].mean()
        std_power = sub['power_w'].std()

        mean_bw = sub['mem_bw_gbps'].mean()

        summary.append({
            "level": lvl,
            "target_cpu_pct": target_cpu,
            "achieved_cpu_pct_mean": round(achieved_cpu, 2),
            "achieved_cpu_pct_std": round(achieved_cpu_std, 2),
            "target_ram_mb": target_ram,
            "achieved_ram_mb": achieved_ram,
            "target_temp_c": target_temp,
            "mean_cpu_temp_c": round(mean_cpu_temp, 2),
            "peak_cpu_temp_c": round(peak_cpu_temp, 2),
            "mean_npu_temp_c": round(mean_npu_temp, 2),
            "mean_gpu_temp_c": round(mean_gpu_temp, 2),
            "mean_power_w": round(mean_power, 2),
            "std_power_w": round(std_power, 2),
            "mean_mem_bw_gbps": round(mean_bw, 2),
            "sample_count": len(sub),
            "provenance": "MEASURED_ON_DEVICE"
        })

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(OUT_REPORT_CSV, index=False)
    print(f"\n[Summary] Calibration Report written to {OUT_REPORT_CSV}:\n")
    print(sum_df.to_string(index=False))

    # 7. Write Checkpoint
    checkpoint = {
        "status": "COMPLETED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device_serial": DEVICE_SERIAL,
        "platform": "Qualcomm Snapdragon 8 Elite (SM8750)",
        "provenance": "MEASURED_ON_DEVICE",
        "raw_telemetry_csv": str(OUT_RAW_CSV),
        "calibration_report_csv": str(OUT_REPORT_CSV),
        "total_samples": len(df),
        "levels": summary
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\n[Checkpoint] Saved {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
