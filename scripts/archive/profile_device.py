#!/usr/bin/env python3
"""
Lightweight System State & Telemetry Profiler for Snapdragon 8 Elite (QIDK).

Records and analyzes:
  - Thermals: CPU, GPU (gpuss), NPU/HTP (nsphvx/nsphmx), DDR, Battery, and Peak SoC temperatures.
  - Power / Battery: current_now, voltage_now, instantaneous power (W), battery capacity.
  - Memory Footprint: System /proc/meminfo and process-specific VmRSS / dumpsys meminfo.

Modes:
  1. Single Snapshot: Quick health check and state dump.
  2. Continuous Monitoring: Concurrent background sampling during benchmark workloads with CSV export.
"""

import os
import sys
import time
import re
import csv
import argparse
import subprocess
import threading
from datetime import datetime


def execute_adb_shell(cmd_str, timeout=10):
    """Runs a shell command via ADB and returns trimmed stdout string."""
    try:
        res = subprocess.run(
            ["adb", "shell", cmd_str],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return res.stdout.strip()
    except Exception as e:
        return ""


class DeviceTelemetry:
    """Collects and aggregates hardware sensors from Snapdragon 8 Elite sysfs."""

    @staticmethod
    def get_thermals():
        """Reads all thermal zones and classifies into CPU, GPU, NPU, DDR, Battery."""
        script = (
            "for tz in /sys/class/thermal/thermal_zone*; do "
            "t=$(cat $tz/temp 2>/dev/null); "
            "n=$(cat $tz/type 2>/dev/null); "
            "if [ -n \"$t\" ]; then echo \"$n:$t\"; fi; "
            "done"
        )
        out = execute_adb_shell(script)
        if not out:
            return {}

        raw_temps = {}
        for line in out.splitlines():
            if ":" in line:
                name, val = line.split(":", 1)
                try:
                    t_mc = int(val.strip())
                    # Discard uninitialized or error values like -40960
                    if -30000 < t_mc < 150000:
                        raw_temps[name.strip()] = t_mc / 1000.0  # Convert to Celsius
                except ValueError:
                    pass

        # Classify sensors
        cpu_temps = [v for k, v in raw_temps.items() if k.startswith("cpu-") or k.startswith("cpuss-")]
        gpu_temps = [v for k, v in raw_temps.items() if k.startswith("gpuss-")]
        npu_temps = [v for k, v in raw_temps.items() if k.startswith("nsphvx-") or k.startswith("nsphmx-")]
        ddr_temps = [v for k, v in raw_temps.items() if "ddr" in k]
        bat_temps = [v for k, v in raw_temps.items() if k == "battery"]

        return {
            "all": raw_temps,
            "cpu_avg": round(sum(cpu_temps) / len(cpu_temps), 2) if cpu_temps else None,
            "cpu_max": round(max(cpu_temps), 2) if cpu_temps else None,
            "gpu_avg": round(sum(gpu_temps) / len(gpu_temps), 2) if gpu_temps else None,
            "gpu_max": round(max(gpu_temps), 2) if gpu_temps else None,
            "npu_avg": round(sum(npu_temps) / len(npu_temps), 2) if npu_temps else None,
            "npu_max": round(max(npu_temps), 2) if npu_temps else None,
            "ddr_temp": round(ddr_temps[0], 2) if ddr_temps else None,
            "battery_temp": round(bat_temps[0], 2) if bat_temps else None,
            "soc_max": round(max(raw_temps.values()), 2) if raw_temps else None,
        }

    @staticmethod
    def get_power():
        """Reads battery voltage, current, and computes instantaneous power."""
        cmd = (
            "cat /sys/class/power_supply/battery/current_now 2>/dev/null; "
            "echo '---'; "
            "cat /sys/class/power_supply/battery/voltage_now 2>/dev/null; "
            "echo '---'; "
            "cat /sys/class/power_supply/battery/capacity 2>/dev/null; "
            "echo '---'; "
            "cat /sys/class/power_supply/battery/status 2>/dev/null"
        )
        out = execute_adb_shell(cmd)
        parts = out.split("---")

        current_ua = None
        voltage_uv = None
        capacity_pct = None
        status = "Unknown"

        if len(parts) >= 1 and parts[0].strip().lstrip("-").isdigit():
            current_ua = int(parts[0].strip())
        if len(parts) >= 2 and parts[1].strip().isdigit():
            voltage_uv = int(parts[1].strip())
        if len(parts) >= 3 and parts[2].strip().isdigit():
            capacity_pct = int(parts[2].strip())
        if len(parts) >= 4 and parts[3].strip():
            status = parts[3].strip()

        # Compute power in Watts: abs(current * voltage)
        power_w = None
        if current_ua is not None and voltage_uv is not None:
            power_w = round(abs(current_ua * 1e-6 * voltage_uv * 1e-6), 3)

        return {
            "current_ma": round(current_ua / 1000.0, 2) if current_ua is not None else None,
            "voltage_v": round(voltage_uv / 1e6, 3) if voltage_uv is not None else None,
            "power_w": power_w,
            "capacity_pct": capacity_pct,
            "status": status,
        }

    @staticmethod
    def get_memory(proc_name=None):
        """Reads system /proc/meminfo and target process VmRSS."""
        meminfo_cmd = "head -n 5 /proc/meminfo"
        out_sys = execute_adb_shell(meminfo_cmd)

        mem_total = 0
        mem_free = 0
        mem_avail = 0

        for line in out_sys.splitlines():
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val_kb = int(re.findall(r"\d+", parts[1])[0]) if re.findall(r"\d+", parts[1]) else 0
                val_mb = round(val_kb / 1024.0, 1)
                if key == "MemTotal":
                    mem_total = val_mb
                elif key == "MemFree":
                    mem_free = val_mb
                elif key == "MemAvailable":
                    mem_avail = val_mb

        mem_used = round(mem_total - mem_avail, 1) if mem_total and mem_avail else None

        # Process-specific memory
        proc_rss_mb = None
        proc_vmsize_mb = None
        pid = None

        if proc_name:
            pid_out = execute_adb_shell(f"pidof {proc_name}")
            if pid_out and pid_out.strip().split():
                pid = pid_out.strip().split()[0]
                status_out = execute_adb_shell(f"grep -E 'VmRSS|VmSize' /proc/{pid}/status 2>/dev/null")
                for sline in status_out.splitlines():
                    if "VmRSS:" in sline:
                        rss_kb = int(re.findall(r"\d+", sline)[0]) if re.findall(r"\d+", sline) else 0
                        proc_rss_mb = round(rss_kb / 1024.0, 1)
                    elif "VmSize:" in sline:
                        vms_kb = int(re.findall(r"\d+", sline)[0]) if re.findall(r"\d+", sline) else 0
                        proc_vmsize_mb = round(vms_kb / 1024.0, 1)

        return {
            "system_total_mb": mem_total,
            "system_free_mb": mem_free,
            "system_available_mb": mem_avail,
            "system_used_mb": mem_used,
            "proc_name": proc_name,
            "proc_pid": pid,
            "proc_rss_mb": proc_rss_mb,
            "proc_vmsize_mb": proc_vmsize_mb,
        }

    @classmethod
    def capture_snapshot(cls, proc_name=None):
        """Captures a complete snapshot of all telemetry channels."""
        now = datetime.now()
        thermals = cls.get_thermals()
        power = cls.get_power()
        memory = cls.get_memory(proc_name=proc_name)

        return {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "epoch": time.time(),
            "cpu_avg_c": thermals.get("cpu_avg"),
            "cpu_max_c": thermals.get("cpu_max"),
            "gpu_avg_c": thermals.get("gpu_avg"),
            "gpu_max_c": thermals.get("gpu_max"),
            "npu_avg_c": thermals.get("npu_avg"),
            "npu_max_c": thermals.get("npu_max"),
            "ddr_temp_c": thermals.get("ddr_temp"),
            "battery_temp_c": thermals.get("battery_temp"),
            "soc_max_c": thermals.get("soc_max"),
            "battery_ma": power.get("current_ma"),
            "voltage_v": power.get("voltage_v"),
            "power_w": power.get("power_w"),
            "battery_pct": power.get("capacity_pct"),
            "battery_status": power.get("status"),
            "sys_mem_used_mb": memory.get("system_used_mb"),
            "sys_mem_avail_mb": memory.get("system_available_mb"),
            "proc_name": memory.get("proc_name"),
            "proc_pid": memory.get("proc_pid"),
            "proc_rss_mb": memory.get("proc_rss_mb"),
            "proc_vmsize_mb": memory.get("proc_vmsize_mb"),
        }


class DeviceProfiler:
    """Continuous background profiler."""

    def __init__(self, interval=1.0, proc_name=None, output_csv=None):
        self.interval = interval
        self.proc_name = proc_name
        self.output_csv = output_csv
        self._stop_event = threading.Event()
        self._thread = None
        self.records = []

    def start(self):
        """Starts monitoring in a background thread."""
        self._stop_event.clear()
        self.records = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops monitoring and flushes CSV."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self.output_csv and self.records:
            self._write_csv()

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            snap = DeviceTelemetry.capture_snapshot(self.proc_name)
            self.records.append(snap)
            time.sleep(self.interval)

    def _write_csv(self):
        if not self.records:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        keys = list(self.records[0].keys())
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.records)


def print_dashboard_snapshot(snap):
    """Pretty prints a snapshot dashboard to console."""
    print("=" * 66)
    print(f"      Snapdragon 8 Elite Telemetry Snapshot ({snap['timestamp']})")
    print("=" * 66)
    print("🌡️  THERMAL DYNAMICS (°C):")
    print(f"   CPU Max / Avg : {snap['cpu_max_c'] or 'N/A'}°C / {snap['cpu_avg_c'] or 'N/A'}°C")
    print(f"   GPU Max / Avg : {snap['gpu_max_c'] or 'N/A'}°C / {snap['gpu_avg_c'] or 'N/A'}°C")
    print(f"   NPU Max / Avg : {snap['npu_max_c'] or 'N/A'}°C / {snap['npu_avg_c'] or 'N/A'}°C")
    print(f"   DDR / Battery : {snap['ddr_temp_c'] or 'N/A'}°C / {snap['battery_temp_c'] or 'N/A'}°C")
    print(f"   Peak SoC Temp : {snap['soc_max_c'] or 'N/A'}°C")
    print("-" * 66)
    print("⚡ POWER & BATTERY:")
    print(f"   Voltage       : {snap['voltage_v'] or 'N/A'} V")
    print(f"   Current       : {snap['battery_ma'] or 'N/A'} mA")
    print(f"   Power Draw    : {snap['power_w'] or 'N/A'} W")
    print(f"   Battery Level : {snap['battery_pct'] or 'N/A'}% ({snap['battery_status']})")
    print("-" * 66)
    print("🧠 MEMORY (RAM):")
    print(f"   System RAM    : {snap['sys_mem_used_mb'] or 'N/A'} MB used / {snap['sys_mem_avail_mb'] or 'N/A'} MB available")
    if snap.get("proc_name"):
        proc_str = f"{snap['proc_name']} (PID: {snap['proc_pid'] or 'NOT RUNNING'})"
        print(f"   Process Target: {proc_str}")
        print(f"   Process VmRSS : {snap['proc_rss_mb'] or 'N/A'} MB")
        print(f"   Process VmSize: {snap['proc_vmsize_mb'] or 'N/A'} MB")
    print("=" * 66 + "\n")


def run_continuous_monitor(interval=1.0, duration=None, proc_name=None, output_csv=None):
    """Runs interactive continuous monitoring."""
    print("=" * 66)
    print("   Starting Snapdragon 8 Elite Continuous Hardware Profiler   ")
    print("=" * 66)
    print(f"  Sampling Interval : {interval}s")
    print(f"  Max Duration      : {f'{duration}s' if duration else 'Continuous (Ctrl+C to stop)'}")
    print(f"  Monitored Process : {proc_name or 'System-wide'}")
    print(f"  Output CSV Log    : {output_csv or 'None (console only)'}")
    print("=" * 66 + "\n")

    profiler = DeviceProfiler(interval=interval, proc_name=proc_name, output_csv=output_csv)
    profiler.start()

    start_time = time.time()
    try:
        print(f"{'Time':<12} | {'CPU Max':<8} | {'NPU Max':<8} | {'GPU Max':<8} | {'Power (W)':<9} | {'Proc RSS':<10} | {'Sys Used'}")
        print("-" * 75)

        while True:
            time.sleep(interval)
            if not profiler.records:
                continue
            snap = profiler.records[-1]
            time_str = snap["timestamp"].split(" ")[1]
            cpu_m = f"{snap['cpu_max_c']}°C" if snap['cpu_max_c'] else "N/A"
            npu_m = f"{snap['npu_max_c']}°C" if snap['npu_max_c'] else "N/A"
            gpu_m = f"{snap['gpu_max_c']}°C" if snap['gpu_max_c'] else "N/A"
            pwr = f"{snap['power_w']} W" if snap['power_w'] is not None else "N/A"
            rss = f"{snap['proc_rss_mb']} MB" if snap['proc_rss_mb'] is not None else "-"
            sys_u = f"{snap['sys_mem_used_mb']} MB" if snap['sys_mem_used_mb'] is not None else "N/A"

            print(f"{time_str:<12} | {cpu_m:<8} | {npu_m:<8} | {gpu_m:<8} | {pwr:<9} | {rss:<10} | {sys_u}")

            if duration and (time.time() - start_time) >= duration:
                break

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        profiler.stop()

    # Print summary statistics
    records = profiler.records
    if records:
        print("\n" + "=" * 66)
        print("                  PROFILING RUN SUMMARY                  ")
        print("=" * 66)
        cpu_maxes = [r["cpu_max_c"] for r in records if r["cpu_max_c"] is not None]
        npu_maxes = [r["npu_max_c"] for r in records if r["npu_max_c"] is not None]
        gpu_maxes = [r["gpu_max_c"] for r in records if r["gpu_max_c"] is not None]
        powers = [r["power_w"] for r in records if r["power_w"] is not None]

        if cpu_maxes:
            print(f"  CPU Temp Peak / Avg : {max(cpu_maxes):.1f}°C / {sum(cpu_maxes)/len(cpu_maxes):.1f}°C")
        if npu_maxes:
            print(f"  NPU Temp Peak / Avg : {max(npu_maxes):.1f}°C / {sum(npu_maxes)/len(npu_maxes):.1f}°C")
        if gpu_maxes:
            print(f"  GPU Temp Peak / Avg : {max(gpu_maxes):.1f}°C / {sum(gpu_maxes)/len(gpu_maxes):.1f}°C")
        if powers:
            print(f"  Power Peak / Avg    : {max(powers):.2f} W / {sum(powers)/len(powers):.2f} W")
        if output_csv:
            print(f"  Log File Saved      : {os.path.abspath(output_csv)}")
        print("=" * 66 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lightweight hardware telemetry logger for Snapdragon 8 Elite."
    )
    parser.add_argument("--snapshot", "-s", action="store_true",
                        help="Capture and print single instantaneous snapshot (default if no monitor args)")
    parser.add_argument("--monitor", "-m", action="store_true",
                        help="Run continuous sampling loop")
    parser.add_argument("--interval", "-i", type=float, default=1.0,
                        help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--duration", "-d", type=float, default=None,
                        help="Monitoring duration in seconds (default: unlimited)")
    parser.add_argument("--proc", "-p", type=str, default=None,
                        help="Process name to track memory footprint (e.g., sd_qidk_runner_encoder)")
    parser.add_argument("--output_csv", "-o", type=str, default=None,
                        help="Output CSV file to log sampled records")

    args = parser.parse_args()

    if args.monitor or args.duration or args.output_csv:
        run_continuous_monitor(
            interval=args.interval,
            duration=args.duration,
            proc_name=args.proc,
            output_csv=args.output_csv
        )
    else:
        snap = DeviceTelemetry.capture_snapshot(proc_name=args.proc)
        print_dashboard_snapshot(snap)
