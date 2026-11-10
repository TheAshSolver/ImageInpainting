#!/usr/bin/env python3
"""
scripts/rl_thermal_data_collector.py
================================================================================
🚀 REINFORCEMENT LEARNING SYSTEM TELEMETRY & THERMAL BENCHMARK HARNESS
Strategy B: Thermal Hysteresis Ramp-Up & Ramp-Down
Target Platform: Qualcomm Snapdragon 8 Elite (Hexagon HTP v79 NPU & Adreno 830 GPU)
================================================================================

This module:
  1. Profiles the exact relationship between SoC Silicon Temperature (33°C <-> 60°C)
     and Inference Latency, RAM (RSS), Accelerator Utilization (NPU Usage %, GPU Usage %),
     and Battery Voltage/Power/Energy across MIGAN, LaMa-Dilated, and AOT-GAN.
  2. Implements Strategy B (Thermal Hysteresis Ramp-Up & Ramp-Down):
     - Alternates heating sweeps (33 -> 60°C on NPU) with cooling sweeps (60 -> 33°C on GPU).
     - Eliminates idle cooldown wait times.
     - Captures directional thermal hysteresis.
  3. Outputs telemetry at every step in dual format:
     - Machine-readable raw JSON line.
     - Human-readable formatted live HUD table with real-time progress bar.
     - Buffer display showing the most recently collected data points.
  4. Incrementally checkpoints data to CSV after every sample.
  5. Renders 5 publication-ready multi-panel figures.

Usage:
  # Full Strategy B sweep (504 data points across 6 configurations):
  python scripts/rl_thermal_data_collector.py --samples-per-deg 3 --ceiling 60 --floor 33

  # Fast dry-run / smoke test (1 sample per deg, 33-36°C):
  python scripts/rl_thermal_data_collector.py --samples-per-deg 1 --ceiling 36 --floor 33 --dry-run
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from collections import deque
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==============================================================================
# CONFIGURATION & REPOSITORY PATHS
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_CSV = REPO_ROOT / "Benchmark" / "output" / "rl_system_thermal_benchmark.csv"
DEFAULT_FIGURES_DIR = REPO_ROOT / "Benchmark" / "output" / "rl_thermal_figures"

DEVICE_SANDBOX = "/data/local/tmp/lama"
DEVICE_RUNNER = f"{DEVICE_SANDBOX}/bench_thermal_runner"
HOST_RUNNER = REPO_ROOT / "scripts" / "bench_thermal_runner"

# Strategy B Sequence: Alternating Heating (NPU) and Cooling (GPU)
STRATEGY_B_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": "migan_npu",
        "model": "MIGAN",
        "runtime": "dsp",
        "runtime_label": "NPU",
        "dlc": "migan_htp_v79.dlc",
        "polarity": 1,          # Inverted: 0=hole, 1=keep
        "direction": "up",       # Heat up
        "target_bound": 60,
        "color": "#1f77b4"
    },
    {
        "id": "migan_gpu",
        "model": "MIGAN",
        "runtime": "gpu_float16",
        "runtime_label": "GPU",
        "dlc": "migan.dlc",
        "polarity": 1,
        "direction": "down",     # Cool down
        "target_bound": 33,
        "color": "#aec7e8"
    },
    {
        "id": "lama_npu",
        "model": "LaMa-Dilated",
        "runtime": "dsp",
        "runtime_label": "NPU",
        "dlc": "bench_cache_lama_dilated.dlc",
        "polarity": 0,          # Standard: 1=hole, 0=keep
        "direction": "up",       # Heat up
        "target_bound": 60,
        "color": "#2ca02c"
    },
    {
        "id": "lama_gpu",
        "model": "LaMa-Dilated",
        "runtime": "gpu_float16",
        "runtime_label": "GPU",
        "dlc": "lama_dilated.dlc",
        "polarity": 0,
        "direction": "down",     # Cool down
        "target_bound": 33,
        "color": "#98df8a"
    },
    {
        "id": "aotgan_npu",
        "model": "AOT-GAN",
        "runtime": "dsp",
        "runtime_label": "NPU",
        "dlc": "bench_cache_aotgan.dlc",
        "polarity": 0,
        "direction": "up",       # Heat up
        "target_bound": 60,
        "color": "#d62728"
    },
    {
        "id": "aotgan_gpu",
        "model": "AOT-GAN",
        "runtime": "gpu_float16",
        "runtime_label": "GPU",
        "dlc": "aotgan.dlc",
        "polarity": 0,
        "direction": "down",     # Cool down
        "target_bound": 33,
        "color": "#ff9896"
    }
]


# ==============================================================================
# PROGRESS BAR & FORMATTING UTILITIES
# ==============================================================================
def render_progress_bar(current: int, total: int, width: int = 25) -> str:
    """Generates a high-contrast visual Unicode progress bar."""
    if total <= 0:
        return "[" + "░" * width + "] 0.0%"
    frac = min(1.0, max(0.0, current / total))
    filled = int(round(width * frac))
    pct = frac * 100.0
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:>5.1f}%"


def print_recent_samples_hud(recent_samples: deque) -> None:
    """Prints a structured inspection HUD showing the most recently collected data points."""
    if not recent_samples:
        return
    print("\n   ┌─── MOST RECENT COLLECTED TELEMETRY BUFFER (Last {} Points) ────────────────────────────────────────────────────────┐".format(len(recent_samples)))
    print("   │ #   │ Model        │ Engine │ Dir  │ Temp   │ Latency (ms) │ RSS RAM  │ NPU / GPU Usg │ Power (mW) │ Battery V │ Transition │")
    print("   ├───┼──────────────┼────────┼──────┼────────┼──────────────┼──────────┼───────────────┼────────────┼───────────┼────────────┤")
    for idx, s in enumerate(recent_samples, 1):
        m_name = f"{s['model']:<12}"
        eng = f"{s['runtime']:<6}"
        direction = f"{s.get('direction', 'up'):<4}"
        t_str = f"{s['exact_temp_c']:>5.1f}°C"
        lat_str = f"{s['latency_ms']:>9.2f} ms"
        rss_str = f"{s['ram_rss_mb']:>6.1f} MB"
        util_val = s['npu_usage_pct'] if s['runtime'] == 'NPU' else s['gpu_usage_pct']
        util_tag = "NPU" if s['runtime'] == 'NPU' else "GPU"
        util_str = f"{util_tag}: {util_val:>4.1f}%"
        pwr_str = f"{s['power_mw']:>7.1f} mW"
        v_str = f"{s['battery_v']:>5.3f} V"
        trans_str = f"{s.get('transition_time_s', s.get('heat_time_s', 0.0)):>6.2f} s"
        print(f"   │ {idx:<3} │ {m_name} │ {eng} │ {direction} │ {t_str} │ {lat_str} │ {rss_str} │ {util_str:<13} │ {pwr_str} │ {v_str} │ {trans_str:>10} │")
    print("   └───┴──────────────┴────────┴──────┴────────┴──────────────┴──────────┴───────────────┴────────────┴───────────┴────────────┘\n")


# ==============================================================================
# MODULE 1: DEVICE ENVIRONMENT & TELEMETRY
# ==============================================================================
class DeviceEnvironment:
    """Manages ADB connectivity and binary execution on the Qualcomm DevKit."""

    def __init__(self, adb_path: str = "adb"):
        self.adb = adb_path
        self._verify_connection()

    def _verify_connection(self) -> None:
        p = subprocess.run([self.adb, "devices"], capture_output=True, text=True)
        lines = [l.strip() for l in p.stdout.splitlines() if l.strip()]
        devices = [l.split()[0] for l in lines[1:] if "device" in l]
        if not devices:
            raise RuntimeError("No ADB device detected. Please ensure Qualcomm DevKit is connected.")
        print(f"✅ Connected ADB device: {devices[0]}")

    def push_runner(self) -> None:
        """Pushes compiled resident thermal runner to device."""
        if not HOST_RUNNER.exists():
            raise FileNotFoundError(f"Host binary not found: {HOST_RUNNER}. Compile bench_thermal_runner.cpp first.")
        subprocess.run([self.adb, "push", str(HOST_RUNNER), DEVICE_RUNNER], check=True, capture_output=True)
        subprocess.run([self.adb, "shell", f"chmod +x {DEVICE_RUNNER}"], check=True, capture_output=True)
        print(f"✅ Staged native runner at {DEVICE_RUNNER}")

    def get_zone_temperature(self, runtime: str) -> float:
        """Reads highest temperature (°C) of active compute zone."""
        pattern = "nsphvx-*|nsphmx-*" if runtime == "dsp" else "gpuss-*"
        cmd = (
            f"for t in /sys/class/thermal/thermal_zone*; do "
            f"ty=$(cat $t/type 2>/dev/null); "
            f"case $ty in {pattern}) cat $t/temp 2>/dev/null;; esac; "
            f"done | sort -n | tail -n 1"
        )
        res = subprocess.run([self.adb, "shell", cmd], capture_output=True, text=True)
        val = res.stdout.strip()
        if not val or not val.isdigit():
            raise RuntimeError(
                f"Failed to read temperature from device for runtime '{runtime}'. "
                f"Command returned: '{val}'. Ensure device is connected and sysfs nodes exist."
            )
        return int(val) / 1000.0


# ==============================================================================
# MODULE 2: STRATEGY B THERMAL SWEEP BENCHMARK ORCHESTRATOR
# ==============================================================================
class ThermalSweepOrchestrator:
    """Orchestrates Strategy B thermal sweeps with dual streaming output and progress reporting."""

    def __init__(self, env: DeviceEnvironment, samples_per_deg: int = 3, ceiling_c: int = 60, floor_c: int = 33):
        self.env = env
        self.samples_per_deg = max(1, min(samples_per_deg, 5))
        self.ceiling_c = ceiling_c
        self.floor_c = floor_c
        self.recent_buffer: deque = deque(maxlen=5)

    def run_configuration(
        self,
        cfg: Dict[str, Any],
        config_idx: int,
        total_configs: int,
        overall_collected: int,
        total_expected_points: int,
        output_csv: Path
    ) -> List[Dict[str, Any]]:
        """Executes resident thermal sweep for a single model and runtime configuration."""
        dlc_path = f"{DEVICE_SANDBOX}/{cfg['dlc']}"
        runtime = cfg["runtime"]
        polarity = cfg["polarity"]
        model_name = cfg["model"]
        direction = cfg["direction"]
        target_bound = self.ceiling_c if direction == "up" else self.floor_c

        dir_arrow = "▲ HEATING RAMP-UP" if direction == "up" else "▼ COOLING RAMP-DOWN"
        temp_span_str = f"33°C → {self.ceiling_c}°C" if direction == "up" else f"{self.ceiling_c}°C → {self.floor_c}°C"

        print(f"\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║ 🚀 CONFIG [{config_idx}/{total_configs}]: {model_name:<12} ({cfg['runtime_label']:<3}) │ Mode: {dir_arrow:<20} │ Target: {temp_span_str:<14} │ Samples/°C: {self.samples_per_deg} ║")
        print(f"╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
        print(f"┌──────────┬─────────────────────────────┬──────────────┬────────────┬─────────────┬────────────┬───────────┬─────────────┐")
        print(f"│ Temp     │ Sweep Progress              │ Latency (ms) │ RAM (RSS)  │ Utilization │ Power (mW) │ Energy    │ Transition  │")
        print(f"├──────────┼─────────────────────────────┼──────────────┼────────────┼─────────────┼────────────┼───────────┼─────────────┤")

        cmd = (
            f"cd {DEVICE_SANDBOX} && "
            f"export LD_LIBRARY_PATH={DEVICE_SANDBOX}/lib:{DEVICE_SANDBOX}:/system/lib64 && "
            f"export ADSP_LIBRARY_PATH='{DEVICE_SANDBOX}/dsp/lib;{DEVICE_SANDBOX}/dsp;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && "
            f"./bench_thermal_runner {dlc_path} {runtime} {polarity} {model_name} {direction} {target_bound} {self.samples_per_deg}"
        )

        records: List[Dict[str, Any]] = []
        proc = subprocess.Popen([self.env.adb, "shell", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        collecting = False
        sweep_collected = 0
        expected_per_sweep = (self.ceiling_c - self.floor_c + 1) * self.samples_per_deg

        for line in proc.stdout:
            line_str = line.strip()
            if line_str == "---SWEEP_DATA_START---":
                collecting = True
                continue
            elif line_str == "---SWEEP_DATA_END---":
                break

            if collecting and line_str.startswith("{") and line_str.endswith("}"):
                try:
                    data = json.loads(line_str)
                    records.append(data)
                    sweep_collected += 1
                    current_overall = overall_collected + sweep_collected

                    # Update recent buffer
                    self.recent_buffer.append(data)

                    # 1. Output Raw JSON (Machine Readable)
                    print(f"[RAW_JSON] {line_str}")

                    # 2. Output Human Readable HUD Table Row
                    sweep_bar = render_progress_bar(sweep_collected, expected_per_sweep, width=12)
                    util_val = data['npu_usage_pct'] if data['runtime'] == 'NPU' else data['gpu_usage_pct']
                    util_label = "NPU" if data['runtime'] == 'NPU' else "GPU"
                    util_str = f"{util_label}: {util_val:>4.1f}%"
                    trans_s = data.get('transition_time_s', data.get('heat_time_s', 0.0))
                    trans_str = f"{trans_s:>5.1f} s"

                    print(f"│ {data['exact_temp_c']:>5.1f}°C  │ {sweep_bar} │ {data['latency_ms']:>10.2f} ms │ {data['ram_rss_mb']:>7.1f} MB │ {util_str:>11} │ {data['power_mw']:>8.1f} mW │ {data['energy_mj']:>6.2f} mJ │ {trans_str:>9}   │")

                    # 3. Overall Progress Indicator
                    overall_bar = render_progress_bar(current_overall, total_expected_points, width=20)
                    print(f"[PROGRESS] Overall: {overall_bar} ({current_overall}/{total_expected_points} pts) | Last Latency: {data['latency_ms']:.2f} ms | Temp: {data['exact_temp_c']:.1f}°C")

                    # 4. Show Recent Data Window every 3 samples (1 degree step)
                    if sweep_collected % self.samples_per_deg == 0:
                        print_recent_samples_hud(self.recent_buffer)

                    # 5. Incremental checkpoint save to disk
                    df_row = pd.DataFrame([data])
                    df_row.to_csv(output_csv, mode="a", header=not output_csv.exists(), index=False)

                    sys.stdout.flush()

                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Corrupt JSON telemetry received from device: '{line_str}'") from e

        proc.wait()
        print(f"└──────────┴─────────────────────────────┴──────────────┴────────────┴─────────────┴────────────┴───────────┴─────────────┘")
        if proc.returncode != 0:
            stderr_out = proc.stderr.read()
            raise RuntimeError(f"Native runner failed with exit code {proc.returncode}:\n{stderr_out}")
        return records

    def run_full_suite(self, output_csv: Path, configs_to_run: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """Executes full Strategy B benchmark suite across all configurations."""
        self.env.push_runner()
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        # Clear existing CSV if starting a fresh full run
        if output_csv.exists():
            output_csv.unlink()

        target_configs = configs_to_run or STRATEGY_B_CONFIGS
        deg_span = (self.ceiling_c - self.floor_c + 1)
        total_expected_points = len(target_configs) * deg_span * self.samples_per_deg

        print("\n" + "=" * 90)
        print("🎯 INITIATING QUALCOMM SNAPDRAGON 8 ELITE BENCHMARK SWEEP (STRATEGY B)")
        print(f"   Target Thermal Range   : {self.floor_c}°C ⟷ {self.ceiling_c}°C (1°C Stepping)")
        print(f"   Samples per Degree     : {self.samples_per_deg}")
        print(f"   Total Configurations   : {len(target_configs)} (Alternating NPU Up & GPU Down)")
        print(f"   Total Data Points Goal : {total_expected_points}")
        print(f"   Output Dataset Path    : {output_csv}")
        print("=" * 90)

        all_records: List[Dict[str, Any]] = []

        for idx, cfg in enumerate(target_configs):
            current_count = len(all_records)
            cfg_records = self.run_configuration(
                cfg=cfg,
                config_idx=idx + 1,
                total_configs=len(target_configs),
                overall_collected=current_count,
                total_expected_points=total_expected_points,
                output_csv=output_csv
            )
            all_records.extend(cfg_records)

            print(f"\n📌 Checkpoint: Configuration {idx+1}/{len(target_configs)} complete. Total points on disk: {len(all_records)} / {total_expected_points}")

        df = pd.DataFrame(all_records)
        print("\n" + "=" * 90)
        print(f"🎉 BENCHMARK COMPLETE! Successfully collected {len(df)} empirical telemetry points.")
        print("=" * 90)
        self._print_final_summary_table(df)
        return df

    def _print_final_summary_table(self, df: pd.DataFrame) -> None:
        """Prints a comprehensive end-of-benchmark summary table."""
        print("\n📊 ════════════════════════════════════ FINAL BENCHMARK SUMMARY ════════════════════════════════════")
        print(f"   {'Model':<14} {'Engine':<6} {'Direction':<6} {'Points':<8} {'Latency Mean±Std':<20} {'RAM RSS':<12} {'Mean Power':<12}")
        print("   " + "─" * 86)
        for (m, r, d), grp in df.groupby(["model", "runtime", "direction"]):
            pts = len(grp)
            lat_mean = grp['latency_ms'].mean()
            lat_std = grp['latency_ms'].std()
            ram_mean = grp['ram_rss_mb'].mean()
            pwr_mean = grp['power_mw'].mean()
            lat_str = f"{lat_mean:.2f} ± {lat_std:.2f} ms"
            print(f"   {m:<14} {r:<6} {d:<6} {pts:<8} {lat_str:<20} {ram_mean:>6.1f} MB     {pwr_mean:>7.1f} mW")
        print("   " + "═" * 86 + "\n")


# ==============================================================================
# MODULE 3: VISUAL GRAPH GENERATOR
# ==============================================================================
class BenchmarkVisualizer:
    """Generates comprehensive, publication-quality figures from collected telemetry."""

    def __init__(self, csv_path: Path, output_dir: Path):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(csv_path)
        self._set_plot_style()

    def _set_plot_style(self) -> None:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.titlesize"] = 13
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 10
        plt.rcParams["figure.titlesize"] = 15

    def plot_latency_vs_temperature(self) -> Path:
        """Figure 1: Latency (ms) vs Silicon Temperature (1°C steps) across models and runtimes."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        for (model, runtime), group in self.df.groupby(["model", "runtime"]):
            style = "-" if runtime == "NPU" else "--"
            marker = "o" if runtime == "NPU" else "s"
            deg_summary = group.groupby("temp_deg")["latency_ms"].agg(["mean", "std"]).reset_index()
            ax.plot(deg_summary["temp_deg"], deg_summary["mean"], 
                    label=f"{model} ({runtime})", linestyle=style, marker=marker, linewidth=2, markersize=5)
            if deg_summary["std"].notnull().any() and (deg_summary["std"] > 0).any():
                ax.fill_between(deg_summary["temp_deg"], 
                                deg_summary["mean"] - deg_summary["std"], 
                                deg_summary["mean"] + deg_summary["std"], alpha=0.15)

        ax.set_title("Inference Latency vs. Silicon Temperature (1°C Stepping)", weight="bold", pad=12)
        ax.set_xlabel("Compute Zone Temperature (°C)", labelpad=8)
        ax.set_ylabel("Pure Inference Latency (ms)", labelpad=8)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(title="Model & Engine", loc="best", frameon=True)
        
        out_path = self.output_dir / "01_latency_vs_temperature.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_hardware_utilization(self) -> Path:
        """Figure 2: Hardware Accelerator Utilization (NPU Usage % vs GPU Busy %)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, sharex=True)

        # NPU Usage
        npu_df = self.df[self.df["runtime"] == "NPU"]
        for model, group in npu_df.groupby("model"):
            deg_summary = group.groupby("temp_deg")["npu_usage_pct"].mean().reset_index()
            ax1.plot(deg_summary["temp_deg"], deg_summary["npu_usage_pct"], 
                     label=model, marker="o", linewidth=2, markersize=5)

        ax1.set_title("NPU Usage (% Active Cycles / Total Cycles)", weight="bold")
        ax1.set_xlabel("NPU Temperature (°C)")
        ax1.set_ylabel("NPU Hardware Usage (%)")
        ax1.set_ylim(50, 105)
        ax1.grid(True, linestyle="--", alpha=0.6)
        if not npu_df.empty:
            ax1.legend(loc="lower right")

        # GPU Usage
        gpu_df = self.df[self.df["runtime"] == "GPU"]
        for model, group in gpu_df.groupby("model"):
            deg_summary = group.groupby("temp_deg")["gpu_usage_pct"].mean().reset_index()
            ax2.plot(deg_summary["temp_deg"], deg_summary["gpu_usage_pct"], 
                     label=model, marker="s", linestyle="--", linewidth=2, markersize=5)

        ax2.set_title("GPU Engine Busy Utilization (kgsl)", weight="bold")
        ax2.set_xlabel("GPU Temperature (°C)")
        ax2.set_ylabel("GPU Usage (%)")
        ax2.set_ylim(50, 105)
        ax2.grid(True, linestyle="--", alpha=0.6)
        if not gpu_df.empty:
            ax2.legend(loc="lower right")

        fig.suptitle("Hardware Accelerator Utilization vs. Operating Temperature", weight="bold", y=1.02)
        out_path = self.output_dir / "02_hardware_utilization.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_memory_footprint(self) -> Path:
        """Figure 3: Memory Footprint (RAM VmRSS MB) across temperature."""
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
        
        for (model, runtime), group in self.df.groupby(["model", "runtime"]):
            deg_summary = group.groupby("temp_deg")["ram_rss_mb"].mean().reset_index()
            ax.plot(deg_summary["temp_deg"], deg_summary["ram_rss_mb"], 
                    label=f"{model} ({runtime})", marker="^", linewidth=2, markersize=5)

        ax.set_title("Resident RAM Footprint vs. Temperature (Static Allocation)", weight="bold", pad=12)
        ax.set_xlabel("Silicon Temperature (°C)")
        ax.set_ylabel("Resident Set Size (MB)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(title="Model & Engine", loc="best")

        out_path = self.output_dir / "03_memory_footprint.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_power_and_energy(self) -> Path:
        """Figure 4: Instantaneous Power Draw (mW) and Energy per Inpaint (mJ)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        for (model, runtime), group in self.df.groupby(["model", "runtime"]):
            deg_summary = group.groupby("temp_deg")[["power_mw", "energy_mj"]].mean().reset_index()
            style = "-" if runtime == "NPU" else "--"
            ax1.plot(deg_summary["temp_deg"], deg_summary["power_mw"], 
                     label=f"{model} ({runtime})", linestyle=style, marker="o", markersize=4)
            ax2.plot(deg_summary["temp_deg"], deg_summary["energy_mj"], 
                     label=f"{model} ({runtime})", linestyle=style, marker="s", markersize=4)

        ax1.set_title("Instantaneous Power Draw (mW)", weight="bold")
        ax1.set_xlabel("Silicon Temperature (°C)")
        ax1.set_ylabel("Power (mW)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="best")

        ax2.set_title("Energy Per Inpaint (mJ = Power × Latency)", weight="bold")
        ax2.set_xlabel("Silicon Temperature (°C)")
        ax2.set_ylabel("Energy (mJ)")
        ax2.set_yscale("log")
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="best")

        fig.suptitle("Battery Power Draw and Energy Footprint across Thermal Range", weight="bold", y=1.02)
        out_path = self.output_dir / "04_power_and_energy.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_rl_summary_dashboard(self) -> Path:
        """Figure 5: Master 4-panel dashboard integrating all system variables for RL calibration."""
        fig, axs = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
        
        # Panel 1: Latency Curves
        for (model, runtime), group in self.df.groupby(["model", "runtime"]):
            s = group.groupby("temp_deg")["latency_ms"].mean().reset_index()
            style = "-" if runtime == "NPU" else ":"
            axs[0, 0].plot(s["temp_deg"], s["latency_ms"], label=f"{model} ({runtime})", linestyle=style, marker="o", markersize=3)
        axs[0, 0].set_title("A. Inference Latency (ms)", weight="bold")
        axs[0, 0].set_ylabel("Latency (ms)")
        axs[0, 0].set_yscale("log")
        axs[0, 0].yaxis.set_major_formatter(ticker.ScalarFormatter())
        axs[0, 0].grid(True, linestyle="--", alpha=0.5)
        axs[0, 0].legend(fontsize=8, loc="best")

        # Panel 2: Energy per Inpaint
        for (model, runtime), group in self.df.groupby(["model", "runtime"]):
            s = group.groupby("temp_deg")["energy_mj"].mean().reset_index()
            style = "-" if runtime == "NPU" else ":"
            axs[0, 1].plot(s["temp_deg"], s["energy_mj"], label=f"{model} ({runtime})", linestyle=style, marker="s", markersize=3)
        axs[0, 1].set_title("B. Energy per Inpaint (mJ)", weight="bold")
        axs[0, 1].set_ylabel("Energy (mJ)")
        axs[0, 1].set_yscale("log")
        axs[0, 1].yaxis.set_major_formatter(ticker.ScalarFormatter())
        axs[0, 1].grid(True, linestyle="--", alpha=0.5)
        axs[0, 1].legend(fontsize=8, loc="best")

        # Panel 3: Hardware Utilization
        for (model, runtime), group in self.df.groupby(["model", "runtime"]):
            col = "npu_usage_pct" if runtime == "NPU" else "gpu_usage_pct"
            s = group.groupby("temp_deg")[col].mean().reset_index()
            axs[1, 0].plot(s["temp_deg"], s[col], label=f"{model} ({runtime})", marker="^", markersize=3)
        axs[1, 0].set_title("C. Engine Utilization (%)", weight="bold")
        axs[1, 0].set_xlabel("Operating Temperature (°C)")
        axs[1, 0].set_ylabel("Usage (%)")
        axs[1, 0].set_ylim(60, 105)
        axs[1, 0].grid(True, linestyle="--", alpha=0.5)
        axs[1, 0].legend(fontsize=8, loc="best")

        # Panel 4: RAM (RSS)
        for (model, runtime), group in self.df.groupby(["model", "runtime"]):
            s = group.groupby("temp_deg")["ram_rss_mb"].mean().reset_index()
            axs[1, 1].plot(s["temp_deg"], s["ram_rss_mb"], label=f"{model} ({runtime})", marker="d", markersize=3)
        axs[1, 1].set_title("D. Static Memory Allocation (MB)", weight="bold")
        axs[1, 1].set_xlabel("Operating Temperature (°C)")
        axs[1, 1].set_ylabel("Resident RAM (MB)")
        axs[1, 1].grid(True, linestyle="--", alpha=0.5)
        axs[1, 1].legend(fontsize=8, loc="best")

        fig.suptitle("Qualcomm Snapdragon 8 Elite: RL Inpainting System Parameter Calibration Dashboard", 
                     weight="bold", fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = self.output_dir / "05_rl_system_summary_dashboard.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def render_all(self) -> List[Path]:
        """Renders the complete set of 5 evaluation figures."""
        print(f"📊 Rendering publication figures from {self.csv_path}...")
        paths = [
            self.plot_latency_vs_temperature(),
            self.plot_hardware_utilization(),
            self.plot_memory_footprint(),
            self.plot_power_and_energy(),
            self.plot_rl_summary_dashboard()
        ]
        for p in paths:
            print(f"  ✅ Saved figure: {p}")
        return paths


# ==============================================================================
# CLI DISPATCHER
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy B RL Inpainting System Telemetry & Thermal Benchmark Harness"
    )
    parser.add_argument("--samples-per-deg", type=int, default=3, choices=[1, 2, 3, 4, 5],
                        help="Number of inference repetitions recorded per 1°C step (default: 3)")
    parser.add_argument("--ceiling", type=int, default=60,
                        help="Target thermal ceiling in °C (default: 60°C)")
    parser.add_argument("--floor", type=int, default=33,
                        help="Target thermal floor in °C (default: 33°C)")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV,
                        help=f"Destination CSV for benchmark records (default: {DEFAULT_OUTPUT_CSV})")
    parser.add_argument("--output-figs", type=Path, default=DEFAULT_FIGURES_DIR,
                        help=f"Directory for generated graphs (default: {DEFAULT_FIGURES_DIR})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run short smoke test (1 config, 33-35°C)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip on-device execution; generate graphs directly from existing CSV")

    args = parser.parse_args()

    # Mode 1: Plot only
    if args.plot_only:
        if not args.output_csv.exists():
            sys.exit(f"❌ Error: CSV file does not exist at {args.output_csv}")
        visualizer = BenchmarkVisualizer(args.output_csv, args.output_figs)
        visualizer.render_all()
        return

    # Mode 2: Strategy B Execution
    env = DeviceEnvironment()
    orchestrator = ThermalSweepOrchestrator(
        env=env,
        samples_per_deg=args.samples_per_deg,
        ceiling_c=args.ceiling,
        floor_c=args.floor
    )

    configs_to_run = STRATEGY_B_CONFIGS
    if args.dry_run:
        print("⚠️ DRY RUN MODE: Executing 1 config up to 35°C...")
        configs_to_run = [STRATEGY_B_CONFIGS[0]]
        orchestrator.ceiling_c = 35

    # Run sweep
    orchestrator.run_full_suite(
        output_csv=args.output_csv,
        configs_to_run=configs_to_run
    )

    # Automatically generate visualization charts upon completion
    visualizer = BenchmarkVisualizer(args.output_csv, args.output_figs)
    visualizer.render_all()


if __name__ == "__main__":
    main()
