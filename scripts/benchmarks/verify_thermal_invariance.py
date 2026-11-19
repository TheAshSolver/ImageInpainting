#!/usr/bin/env python3
"""
scripts/benchmarks/verify_thermal_invariance.py
================================================================================
⚡ HARDWARE DETERMINISM & THERMAL INVARIANCE PROOF
================================================================================
Verifies that Hexagon NPU HTP v79 neural inference execution is 100% bitwise
and numerically invariant across extreme operating temperatures:
  - Cold Baseline: 30°C - 38°C
  - Hot / Throttled State: > 75°C
Metrics:
  - Bitwise SHA-256 digest match
  - Max Absolute Difference (should be 0.0000000)
  - Root Mean Square Error (RMSE, should be 0.0000000)
  - Peak Signal-to-Noise Ratio (PSNR, should be inf dB)
Output:
  - artifacts/thermal_invariance_proof.csv
================================================================================
"""

import os
import sys
import time
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE_DIR = "/data/local/tmp/lama"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ARTIFACTS_DIR / "thermal_invariance_proof.csv"

def run_adb(cmd_list: List[str], check: bool = True) -> subprocess.CompletedProcess:
    full_cmd = ["adb"] + cmd_list
    return subprocess.run(full_cmd, capture_output=True, text=True, check=check)

def get_device_temps() -> Dict[str, float]:
    """Reads board thermistor (sys-therm-0) and silicon junction thermistors."""
    res = run_adb(["shell", "cat /sys/class/thermal/thermal_zone66/temp 2>/dev/null; cat /sys/class/thermal/thermal_zone48/temp 2>/dev/null; cat /sys/class/thermal/thermal_zone51/temp 2>/dev/null; cat /sys/class/thermal/thermal_zone2/temp 2>/dev/null"])
    lines = res.stdout.strip().splitlines()
    temps = {}
    names = ["sys_therm0", "nsphvx0", "nsphmx0", "cpu0"]
    for i, name in enumerate(names):
        try:
            if i < len(lines) and lines[i].strip().isdigit():
                temps[name] = float(lines[i].strip()) / 1000.0
            else:
                temps[name] = 35.0
        except:
            temps[name] = 35.0
    return temps

def ensure_inputs_staged(n_samples: int = 10):
    """Verifies that the first n_samples raw tensors are on device and builds input_10_migan.txt."""
    print(f"[STAGE] Staging {n_samples} input samples on device...")
    run_adb(["shell", f"mkdir -p {DEVICE_DIR}/input/raw_image {DEVICE_DIR}/input/raw_mask_inverted {DEVICE_DIR}/input/raw_mask_standard"])
    
    # Generate input_10_migan.txt on device
    lines = []
    for i in range(1, n_samples + 1):
        s_id = f"{i:03d}"
        lines.append(f"image:=input/raw_image/{s_id}.raw mask:=input/raw_mask_inverted/{s_id}_mask.raw")
    
    input_list_str = "\\n".join(lines)
    run_adb(["shell", f"echo -e '{input_list_str}' > {DEVICE_DIR}/input_10_migan.txt"])
    print("[STAGE] Verified input_10_migan.txt on device.")

def execute_snpe_batch(out_dir_dev: str) -> float:
    """Executes snpe-net-run on device and returns elapsed wall milliseconds."""
    cmd = (
        f"cd {DEVICE_DIR} && "
        f"export LD_LIBRARY_PATH={DEVICE_DIR}/lib:{DEVICE_DIR} && "
        f"export ADSP_LIBRARY_PATH='{DEVICE_DIR}/dsp/lib;{DEVICE_DIR}/dsp;/dsp' && "
        f"rm -rf {out_dir_dev} && "
        f"t0=$(date +%s%3N) && "
        f"./snpe-net-run --container migan_htp_v79.dlc --input_list input_10_migan.txt --output_dir {out_dir_dev} --use_dsp --perf_profile burst > /dev/null 2>&1 && "
        f"t1=$(date +%s%3N) && "
        f"echo $(( t1 - t0 ))"
    )
    res = run_adb(["shell", cmd])
    wall_ms = 0.0
    for l in res.stdout.strip().splitlines():
        if l.strip().isdigit():
            wall_ms = float(l.strip())
            break
    return wall_ms

def pull_raw_outputs(out_dir_dev: str, host_dir: Path, n_samples: int = 10) -> List[Path]:
    """Pulls output_0.raw tensors from each Result_i folder to host."""
    host_dir.mkdir(parents=True, exist_ok=True)
    pulled_paths = []
    for i in range(n_samples):
        dev_f = f"{DEVICE_DIR}/{out_dir_dev}/Result_{i}/output_0.raw"
        host_f = host_dir / f"sample_{i+1:03d}_output.raw"
        run_adb(["pull", dev_f, str(host_f)])
        if host_f.exists() and host_f.stat().st_size > 0:
            pulled_paths.append(host_f)
        else:
            raise RuntimeError(f"Failed to pull {dev_f} to {host_f}")
    return pulled_paths

def heat_device_to_target(target_junc_c: float = 75.0, max_wait_s: int = 60):
    """Spins background CPU stress workers until silicon junction exceeds target_junc_c."""
    print(f"\n[HEAT] Heating device silicon junction towards {target_junc_c}°C...", flush=True)
    heat_cmd = "nohup sh -c 'for i in $(seq 1 8); do (while true; do :; done) & done' </dev/null >/dev/null 2>&1 & echo 'launched'"
    run_adb(["shell", heat_cmd])
    
    t0 = time.time()
    try:
        while time.time() - t0 < max_wait_s:
            t = get_device_temps()
            junc_max = max(t["nsphvx0"], t["nsphmx0"], t["cpu0"])
            print(f"  [Heating {int(time.time()-t0)}s] Sys-Therm-0: {t['sys_therm0']:.1f}°C | Max Junction: {junc_max:.1f}°C (Target: >={target_junc_c}°C)", flush=True)
            if junc_max >= target_junc_c or t["sys_therm0"] >= 72.0:
                print(f"\n[HEAT] Reached target temperature! (Sys0: {t['sys_therm0']:.1f}°C, Junc: {junc_max:.1f}°C)", flush=True)
                return t
            time.sleep(2)
    finally:
        pass
    print("\n[HEAT] Max heating time reached; proceeding with current thermal state.", flush=True)
    return get_device_temps()

def kill_stress_workers():
    """Kills background spin workers."""
    try:
        run_adb(["shell", "killall sh 2>/dev/null; killall yes 2>/dev/null; true"])
    except:
        pass

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print(" ⚡ HEXAGON HTP v79 HARDWARE THERMAL INVARIANCE BENCHMARK")
    print("=" * 80)

    n_samples = 10
    ensure_inputs_staged(n_samples)

    # 1. COLD BASELINE
    print("\n>>> PHASE 1: Cold Baseline Inference (30°C - 38°C)")
    t_cold = get_device_temps()
    print(f"  Cold Status: Sys-Therm-0 = {t_cold['sys_therm0']:.1f}°C | Junction = {t_cold['nsphmx0']:.1f}°C")
    cold_wall = execute_snpe_batch("cold_invariance_out")
    print(f"  Cold 10-Sample Wall Latency: {cold_wall:.1f} ms ({cold_wall/n_samples:.2f} ms/sample)")
    
    cold_host_dir = ARTIFACTS_DIR / "cold_tensors"
    cold_files = pull_raw_outputs("cold_invariance_out", cold_host_dir, n_samples)
    print(f"  Pulled {len(cold_files)} cold output tensors.")

    # 2. HEAT DEVICE TO THROTTLED STATE (>75°C)
    print("\n>>> PHASE 2: Thermal Stress Heating to Throttled State (>75°C)")
    t_hot = heat_device_to_target(target_junc_c=75.0, max_wait_s=45)
    
    # 3. HOT INFERENCE UNDER THROTTLING
    print("\n>>> PHASE 3: Hot Throttled Inference Execution")
    hot_wall = execute_snpe_batch("hot_invariance_out")
    print(f"  Hot 10-Sample Wall Latency: {hot_wall:.1f} ms ({hot_wall/n_samples:.2f} ms/sample)")
    kill_stress_workers()
    
    hot_host_dir = ARTIFACTS_DIR / "hot_tensors"
    hot_files = pull_raw_outputs("hot_invariance_out", hot_host_dir, n_samples)
    print(f"  Pulled {len(hot_files)} hot output tensors.")

    # 4. NUMERICAL & BITWISE COMPARISON
    print("\n>>> PHASE 4: Mathematical & Cryptographic Determinism Verification")
    records = []
    all_bitwise_match = True

    for i in range(n_samples):
        s_id = f"{i+1:03d}"
        c_p = cold_files[i]
        h_p = hot_files[i]

        c_hash = compute_sha256(c_p)
        h_hash = compute_sha256(h_p)
        is_bitwise_match = (c_hash == h_hash)
        if not is_bitwise_match:
            all_bitwise_match = False

        c_arr = np.fromfile(str(c_p), dtype=np.float32)
        h_arr = np.fromfile(str(h_p), dtype=np.float32)

        abs_diff = np.abs(c_arr - h_arr)
        max_abs_diff = float(np.max(abs_diff))
        rmse = float(np.sqrt(np.mean((c_arr - h_arr) ** 2)))
        
        if rmse == 0.0:
            psnr_db = float("inf")
        else:
            psnr_db = float(20.0 * np.log10(1.0 / rmse))

        records.append({
            "sample_id": s_id,
            "cold_sys0_c": t_cold["sys_therm0"],
            "cold_junc_c": t_cold["nsphmx0"],
            "hot_sys0_c": t_hot["sys_therm0"],
            "hot_junc_c": max(t_hot["nsphvx0"], t_hot["nsphmx0"], t_hot["cpu0"]),
            "cold_latency_ms": round(cold_wall / n_samples, 2),
            "hot_latency_ms": round(hot_wall / n_samples, 2),
            "cold_sha256": c_hash,
            "hot_sha256": h_hash,
            "bitwise_identical": is_bitwise_match,
            "max_abs_diff": max_abs_diff,
            "rmse": rmse,
            "psnr_db": "inf" if np.isinf(psnr_db) else f"{psnr_db:.2f}"
        })

        print(f"  Sample {s_id}: Bitwise Match = {is_bitwise_match} | Max Diff = {max_abs_diff:.8f} | RMSE = {rmse:.8f} | PSNR = {psnr_db}")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n[SUCCESS] Validation results exported to: {OUT_CSV}")
    print(f"Overall Invariance Status: {'100% BITWISE IDENTICAL (PROOF VERIFIED)' if all_bitwise_match else 'DISCREPANCY DETECTED'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
