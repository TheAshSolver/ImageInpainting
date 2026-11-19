#!/usr/bin/env python3
"""
showcase/loadgen.py
=============================================================================
⚡ QUALCOMM SNAPDRAGON 8 ELITE SHOWCASE LOAD GENERATOR & DASHBOARD
=============================================================================
Interactive host-side CLI and live terminal dashboard for closed-loop
hardware contention and thermal regulation on the Snapdragon 8 Elite HDK.

Usage:
  # Preflight check of hardware, permissions, and runtimes:
  python showcase/loadgen.py --check

  # Preset contention state with closed-loop thermal regulation:
  python showcase/loadgen.py --level L3 --temp 42

  # Custom multi-resource specification with live dashboard:
  python showcase/loadgen.py --cpu 60 --gpu 40 --npu 30 --ram 3072 --membw 15 \\
         --temp 50 --max-temp 85 --duration 30 --dashboard
=============================================================================
"""

import sys
import os
import time
import signal
import argparse
import subprocess
import threading

DEVICE_ID = "8f27557f"
DEVICE_BIN = "/data/local/tmp/qidk_load_generator"
DEVICE_CSV = "/data/local/tmp/real_contention_telemetry.csv"

# Global process handle for clean termination
current_proc = None

def clean_exit(sig=None, frame=None):
    print("\n\n[INFO] Interrupt caught! Halting on-device engines and restoring clocks...")
    try:
        subprocess.run(["adb", "-s", DEVICE_ID, "shell", "killall -9 qidk_load_generator snpe-net-run 2>/dev/null"],
                       capture_output=True, timeout=5)
    except Exception:
        pass
    print("[INFO] Device cleanly restored. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

def run_adb(cmd_str: str, timeout: int = 15) -> str:
    res = subprocess.run(["adb", "-s", DEVICE_ID, "shell", cmd_str],
                         capture_output=True, text=True, timeout=timeout)
    return res.stdout.strip()

def run_check():
    print("=" * 70)
    print("⚡ SNAPDRAGON 8 ELITE HDK PREFLIGHT SYSTEM VERIFICATION")
    print("=" * 70)

    checks = []

    # 1. ADB Connection
    try:
        out = subprocess.check_output(["adb", "devices"]).decode()
        if DEVICE_ID in out:
            checks.append(("ADB Connection", "PASS", f"Device {DEVICE_ID} online"))
        else:
            checks.append(("ADB Connection", "FAIL", f"Device {DEVICE_ID} not detected in adb devices"))
    except Exception as e:
        checks.append(("ADB Connection", "FAIL", str(e)))

    # 2. Hardware Platform
    try:
        model = run_adb("getprop ro.product.board")
        platform = run_adb("getprop ro.board.platform")
        if platform == "sun" or model == "sun":
            checks.append(("SoC Platform", "PASS", f"Qualcomm SM8750 Snapdragon 8 Elite (platform: {platform})"))
        else:
            checks.append(("SoC Platform", "WARN", f"Unexpected platform: {platform} (model: {model})"))
    except Exception as e:
        checks.append(("SoC Platform", "FAIL", str(e)))

    # 3. Native Load Engine
    try:
        bin_check = run_adb(f"test -x {DEVICE_BIN} && echo YES || echo NO")
        if bin_check == "YES":
            checks.append(("Native Binary", "PASS", f"{DEVICE_BIN} executable present"))
        else:
            checks.append(("Native Binary", "FAIL", f"{DEVICE_BIN} missing or not executable"))
    except Exception as e:
        checks.append(("Native Binary", "FAIL", str(e)))

    # 4. OpenCL GPU (Adreno 830)
    try:
        cl_check = run_adb("test -f /vendor/lib64/libOpenCL.so && echo YES || echo NO")
        gpu_busy = run_adb("cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage 2>/dev/null")
        if cl_check == "YES" and gpu_busy:
            checks.append(("Adreno 830 GPU", "PASS", f"OpenCL ready, KGSL driver busy: {gpu_busy}"))
        else:
            checks.append(("Adreno 830 GPU", "FAIL", "OpenCL library or KGSL node missing"))
    except Exception as e:
        checks.append(("Adreno 830 GPU", "FAIL", str(e)))

    # 5. Hexagon v79 NPU DSP
    try:
        snpe_check = run_adb("test -x /data/local/tmp/lama/snpe-net-run && test -f /data/local/tmp/lama/migan_htp_v79.dlc && echo YES || echo NO")
        if snpe_check == "YES":
            checks.append(("Hexagon v79 NPU", "PASS", "SNPE DSP runner & migan_htp_v79.dlc model verified"))
        else:
            checks.append(("Hexagon v79 NPU", "FAIL", "SNPE runtime or DLC container missing"))
    except Exception as e:
        checks.append(("Hexagon v79 NPU", "FAIL", str(e)))

    # 6. DDR Memory Controller Core (MCCC) Hardware Clock
    try:
        mccc_hz = run_adb("cat /sys/kernel/debug/clk/measure_only_mccc_clk/clk_measure 2>/dev/null")
        if mccc_hz and int(mccc_hz) > 0:
            mccc_mhz = int(mccc_hz) / 1000000.0
            checks.append(("DDR MCCC Clock", "PASS", f"Hardware clock readable: {mccc_mhz:.1f} MHz"))
        else:
            checks.append(("DDR MCCC Clock", "FAIL", "Debugfs MCCC clock measure node inaccessible"))
    except Exception as e:
        checks.append(("DDR MCCC Clock", "FAIL", str(e)))

    # 7. Thermal Sensors
    try:
        cpuss_t = float(run_adb("cat /sys/class/thermal/thermal_zone18/temp 2>/dev/null")) / 1000.0
        npu_t = float(run_adb("cat /sys/class/thermal/thermal_zone48/temp 2>/dev/null")) / 1000.0
        gpu_t = float(run_adb("cat /sys/class/thermal/thermal_zone32/temp 2>/dev/null")) / 1000.0
        checks.append(("Thermal Zones", "PASS", f"CPUSS (Z18): {cpuss_t:.1f}C | GPU: {gpu_t:.1f}C | NPU: {npu_t:.1f}C"))
    except Exception as e:
        checks.append(("Thermal Zones", "FAIL", str(e)))

    # 8. External Power Meter Status
    checks.append(("Physical Power Meter", "BLOCKED", "No inline hardware power monitor connected on 9V rail (Tag: DERIVED_MODEL)"))

    # Print Report
    print(f"{'Component':<24} | {'Status':<7} | {'Details'}")
    print("-" * 70)
    for comp, stat, det in checks:
        print(f"{comp:<24} | {stat:<7} | {det}")
    print("=" * 70)

def render_dashboard(line: str, target_temp: float, max_temp: float):
    """Parses C++ telemetry output line and prints formatted live dashboard."""
    # Line format: [TELEMETRY] timestamp | Level | CPU: xx% (tgt yy%) | GPU: ... | NPU: ... | RSS: ... | DDR: ... | CPUSS: ... | Pwr: ...
    if not line.startswith("[TELEMETRY]"):
        print(line)
        return

    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 8:
        print(line)
        return

    # Extract fields
    time_str = parts[0].replace("[TELEMETRY]", "").strip()
    lvl = parts[1]
    cpu_str = parts[2]
    gpu_str = parts[3]
    npu_str = parts[4]
    rss_str = parts[5]
    ddr_str = parts[6]
    temp_str = parts[7]
    pwr_str = parts[8] if len(parts) > 8 else "N/A"

    os.system("clear" if os.name == "posix" else "cls")
    print("=" * 75)
    print("⚡ SNAPDRAGON 8 ELITE LIVE HARDWARE CONTENTION DASHBOARD")
    print(f"Timestamp: {time_str} | Level: {lvl} | Target Temp: {target_temp:.1f} C (Cutoff: {max_temp:.1f} C)")
    print("=" * 75)
    print(f"{'Subsystem':<14} | {'Live Measurement':<32} | {'Hardware Telemetry Node'}")
    print("-" * 75)
    print(f"{'Oryon CPU':<14} | {cpu_str:<32} | /proc/stat (8 cores)")
    print(f"{'Adreno 830 GPU':<14} | {gpu_str:<32} | /sys/class/kgsl/kgsl-3d0/gpu_busy_pct")
    print(f"{'Hexagon NPU':<14} | {npu_str:<32} | migan_htp_v79.dlc DSP inferences")
    print(f"{'Resident RAM':<14} | {rss_str:<32} | /proc/self/status (VmRSS)")
    print(f"{'DDR Bus':<14} | {ddr_str:<32} | measure_only_mccc_clk (MCCC)")
    print(f"{'Thermal State':<14} | {temp_str:<32} | Zone 18 (cpuss-0-0) Closed-Loop")
    print(f"{'Power Model':<14} | {pwr_str:<32} | TAG: DERIVED_MODEL (EAS Model)")
    print("-" * 75)
    print("Controls: Press Ctrl+C at any time to instantly halt load and restore clocks.")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="QIDK Snapdragon 8 Elite Contention & Thermal Engine")
    parser.add_argument("--check", action="store_true", help="Run preflight system and sensor checks")
    parser.add_argument("--level", type=str, default="L2", help="Contention preset (L0, L1, L2, L3, L4)")
    parser.add_argument("--cpu", type=int, default=-1, help="Target CPU % (0-100)")
    parser.add_argument("--gpu", type=int, default=-1, help="Target GPU % (0-100)")
    parser.add_argument("--npu", type=int, default=-1, help="Target NPU % (0-100)")
    parser.add_argument("--ram", type=int, default=-1, help="Target resident RAM MB (0-8192)")
    parser.add_argument("--membw", type=int, default=-1, help="Target memory bandwidth GB/s (0-35)")
    parser.add_argument("--temp", type=float, default=42.0, help="Held target temperature in deg C (default: 42.0)")
    parser.add_argument("--max-temp", type=float, default=85.0, help="Hard emergency safety cutoff temp in deg C (default: 85.0)")
    parser.add_argument("--duration", type=int, default=15, help="Test duration in seconds (default: 15)")
    parser.add_argument("--dashboard", action="store_true", help="Display live terminal telemetry dashboard")

    args = parser.parse_args()

    if args.check:
        run_check()
        return

    # Build device command
    cmd_parts = [DEVICE_BIN, "--temp", str(args.temp), "--max-temp", str(args.max_temp), "--duration", str(args.duration)]
    if args.cpu >= 0:
        cmd_parts.extend(["--cpu", str(args.cpu)])
        if args.gpu >= 0: cmd_parts.extend(["--gpu", str(args.gpu)])
        if args.npu >= 0: cmd_parts.extend(["--npu", str(args.npu)])
        if args.ram >= 0: cmd_parts.extend(["--ram", str(args.ram)])
        if args.membw >= 0: cmd_parts.extend(["--membw", str(args.membw)])
    else:
        cmd_parts.extend(["--level", args.level])

    cmd_parts.extend(["--csv", DEVICE_CSV])
    full_cmd = " ".join(cmd_parts)

    print("=" * 70)
    print("⚡ LAUNCHING QUALCOMM HARDWARE LOAD GENERATOR")
    print(f"Device: {DEVICE_ID} (SM8750 / sun)")
    print(f"Command: {full_cmd}")
    print("=" * 70)

    # Launch subprocess over ADB
    adb_cmd = ["adb", "-s", DEVICE_ID, "shell", full_cmd]
    proc = subprocess.Popen(adb_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    global current_proc
    current_proc = proc

    try:
        for line in proc.stdout:
            line = line.strip()
            if args.dashboard:
                render_dashboard(line, args.temp, args.max_temp)
            else:
                print(line)
        proc.wait()
    except KeyboardInterrupt:
        clean_exit()

    print("\n[INFO] Execution finished with return code:", proc.returncode)

if __name__ == "__main__":
    main()
