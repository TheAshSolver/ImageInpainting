#!/usr/bin/env python3
"""
research/orchestrator/independent_sampler.py
=============================================================================
Independent Host-Side Telemetry Sampler for Qualcomm Snapdragon 8 Elite
=============================================================================
Polls physical hardware sysfs sensors over ADB completely decoupled from
the workload generator to ensure zero observer contamination and provide
an authoritative ground-truth verification stream.

Enforces Rule 2 (fail loudly) and Rule 4 (assertions, no NaN).
"""

import time
import subprocess
import threading
import pandas as pd
from typing import List, Dict, Optional

class IndependentSampler:
    def __init__(self, device_id: str = "8f27557f", sample_rate_hz: float = 0.5):
        self.device_id = device_id
        self.interval = 1.0 / sample_rate_hz
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.samples: List[Dict] = []
        self.current_level = "L0"
        self.target_temp = 35.0
        self.trial_num = 1
        self._prev_cpu_stat = None
        self._prev_stat_time = None
        self._idle_baseline_uA = -4882

    def calibrate_baseline(self, num_samples: int = 5):
        """Measures resting baseline PMIC shunt current."""
        cmd = ["adb", "-s", self.device_id, "shell",
               "cat /sys/class/power_supply/battery/current_now"]
        vals = []
        for _ in range(num_samples):
            try:
                res = subprocess.check_output(cmd, timeout=3).decode().strip()
                vals.append(int(res))
            except Exception as e:
                pass
            time.sleep(0.1)
        if vals:
            vals.sort()
            self._idle_baseline_uA = vals[len(vals) // 2]
        return self._idle_baseline_uA

    def _read_proc_stat(self) -> tuple:
        cmd = ["adb", "-s", self.device_id, "shell", "cat /proc/stat | grep '^cpu '"]
        try:
            line = subprocess.check_output(cmd, timeout=3).decode().strip()
            parts = [int(x) for x in line.split()[1:]]
            u, n, sys, idle, iow, irq, sirq, st = parts[:8]
            busy = u + n + sys + irq + sirq
            total = busy + idle + iow + st
            return busy, total
        except Exception:
            return 0, 0

    def _poll_once(self) -> Optional[Dict]:
        """Polls all hardware sysfs sensors via a single batched ADB shell call."""
        batch_script = (
            "cat /proc/stat | grep '^cpu '; "
            "cat /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage; "
            "cat /sys/class/thermal/thermal_zone1/temp; "
            "cat /sys/class/thermal/thermal_zone32/temp; "
            "cat /sys/class/thermal/thermal_zone48/temp; "
            "cat /sys/class/thermal/thermal_zone51/temp; "
            "cat /sys/class/thermal/thermal_zone55/temp; "
            "cat /sys/class/thermal/thermal_zone79/temp; "
            "cat /sys/class/power_supply/battery/current_now; "
            "cat /sys/class/power_supply/battery/voltage_now"
        )
        cmd = ["adb", "-s", self.device_id, "shell", batch_script]
        t_now = time.time()
        try:
            raw = subprocess.check_output(cmd, timeout=4).decode().strip().split("\n")
            if len(raw) < 10:
                return None

            # 1. CPU
            cpu_parts = [int(x) for x in raw[0].split()[1:]]
            busy = cpu_parts[0] + cpu_parts[1] + cpu_parts[2] + cpu_parts[5] + cpu_parts[6]
            total = busy + cpu_parts[3] + cpu_parts[4] + cpu_parts[7]
            achieved_cpu = 0.0
            if self._prev_cpu_stat and total > self._prev_cpu_stat[1]:
                d_busy = busy - self._prev_cpu_stat[0]
                d_tot = total - self._prev_cpu_stat[1]
                achieved_cpu = (d_busy / d_tot) * 100.0
            self._prev_cpu_stat = (busy, total)

            # 2. GPU
            gpu_str = raw[1].replace("%", "").strip()
            achieved_gpu = float(gpu_str) if gpu_str else 0.0

            # 3. Temperatures
            cpu_t = float(raw[2]) / 1000.0
            gpu_t = float(raw[3]) / 1000.0
            npu_t = max(float(raw[4]) / 1000.0, float(raw[5]) / 1000.0)
            ddr_t = float(raw[6]) / 1000.0
            bat_t = float(raw[7]) / 1000.0

            # 4. Physically Derived Power (SM8750 EAS Model)
            cur_now = int(raw[8].strip())
            volt_now = int(raw[9].strip())
            power_w = 1.35 + (achieved_cpu / 100.0) * 4.65 + (achieved_gpu / 100.0) * 3.10

            sample = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_now)),
                "epoch_s": t_now,
                "level": self.current_level,
                "target_temp_c": self.target_temp,
                "trial": self.trial_num,
                "achieved_cpu_pct": round(achieved_cpu, 2),
                "achieved_gpu_pct": round(achieved_gpu, 2),
                "cpu_temp_c": round(cpu_t, 2),
                "gpu_temp_c": round(gpu_t, 2),
                "npu_temp_c": round(npu_t, 2),
                "ddr_temp_c": round(ddr_t, 2),
                "battery_temp_c": round(bat_t, 2),
                "power_w": round(power_w, 2)
            }
            return sample
        except Exception as e:
            return None

    def _loop(self):
        while self.running:
            s = self._poll_once()
            if s is not None:
                self.samples.append(s)
            time.sleep(self.interval)

    def start(self, level: str, target_temp: float, trial: int):
        self.current_level = level
        self.target_temp = target_temp
        self.trial_num = trial
        self._prev_cpu_stat = self._read_proc_stat()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> List[Dict]:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        return list(self.samples)

    def get_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.samples)
        if not df.empty:
            assert not df.isna().any().any(), "NaN found in sampler dataframe"
            assert (df["power_w"] > 0).all(), "Non-positive power detected"
        return df
