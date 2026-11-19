#!/usr/bin/env python3
"""
research/orchestrator/workload_controller.py
=============================================================================
Host-Side Controller for Hardware Contention & Thermal Regulation
=============================================================================
Orchestrates closed-loop contention sweeps on the Snapdragon 8 Elite HDK.
Enforces thermal pre-warming, inter-trial cooldown barriers, and verified
execution of the native load generator.
"""

import time
import subprocess
import os

class WorkloadController:
    def __init__(self, device_id: str = "8f27557f", binary_path: str = "/data/local/tmp/qidk_load_generator", idle_baseline_uA: int = -5493):
        self.device_id = device_id
        self.binary_path = binary_path
        self.idle_baseline_uA = idle_baseline_uA

    def set_idle_baseline(self, baseline_uA: int):
        self.idle_baseline_uA = baseline_uA

    def _adb(self, cmd_str: str, timeout: int = 30, retries: int = 3) -> str:
        cmd = ["adb", "-s", self.device_id, "shell", cmd_str]
        for attempt in range(retries + 1):
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if res.returncode == 0 or "killall" in cmd_str:
                    return res.stdout
                err_str = res.stderr.lower()
                if res.returncode == 255 or "closed" in err_str or "offline" in err_str or "not found" in err_str:
                    print(f"[USB WARNING] ADB failure (rc={res.returncode}): {res.stderr.strip()}. Attempting USB recovery (attempt {attempt+1}/{retries+1})...")
                    subprocess.run(["adb", "kill-server"], capture_output=True)
                    time.sleep(1.5)
                    subprocess.run(["adb", "start-server"], capture_output=True)
                    time.sleep(2.0)
                    continue
                else:
                    raise RuntimeError(f"ADB command failed ({res.returncode}): {cmd_str}\nSTDERR: {res.stderr}")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                if attempt == retries:
                    raise
                print(f"[USB WARNING] Retrying after error: {e}")
                subprocess.run(["adb", "kill-server"], capture_output=True)
                time.sleep(1.5)
                subprocess.run(["adb", "start-server"], capture_output=True)
                time.sleep(2.0)
        raise RuntimeError(f"ADB command permanently failed: {cmd_str}")

    def get_peak_soc_temp(self) -> float:
        """Reads peak CPU/GPU temperature."""
        raw = self._adb("cat /sys/class/thermal/thermal_zone1/temp; cat /sys/class/thermal/thermal_zone32/temp")
        temps = [float(x.strip()) / 1000.0 for x in raw.strip().split() if x.strip()]
        return max(temps) if temps else 35.0

    def get_cpuss_temp(self) -> float:
        """Reads Zone 18 (cpuss-0-0) temperature."""
        raw = self._adb("cat /sys/class/thermal/thermal_zone18/temp 2>/dev/null")
        try:
            val = float(raw.strip()) / 1000.0
            if 15.0 < val < 100.0:
                return val
        except Exception:
            pass
        return self.get_peak_soc_temp()

    def regulate_temperature(self, target_temp_c: float, timeout_s: int = 40):
        """
        Closed-loop thermal regulation:
        - If Cool (35C): cools down until cpuss temp <= 36.5 C
        - If Warm (42C): prewarms if < 41.0 C, cools if > 44.0 C
        - If Hot (50C): prewarms if < 48.5 C, cools if > 52.0 C
        """
        if target_temp_c <= 36.0:
            self.enforce_cooldown(target_temp_c, max_wait_s=60)
            return

        cur_t = self.get_cpuss_temp()
        burst_count = 0
        while cur_t < (target_temp_c - 1.5) and burst_count < 4:
            burst_count += 1
            print(f"[PREWARM] Current CPUSS {cur_t:.1f} C < target {target_temp_c:.1f} C. Running 3s burst #{burst_count}...")
            self._adb(f"{self.binary_path} --level L4 --temp {target_temp_c} --duration 3 2>/dev/null", timeout=15)
            time.sleep(0.5)
            cur_t = self.get_cpuss_temp()

        if cur_t > (target_temp_c + 2.5):
            print(f"[COOLDOWN] Current CPUSS {cur_t:.1f} C > target {target_temp_c:.1f} C. Settling...")
            t_start = time.time()
            while time.time() - t_start < 30:
                cur_t = self.get_cpuss_temp()
                if cur_t <= (target_temp_c + 1.0):
                    print(f"[COOLDOWN] Settled at {cur_t:.1f} C. Ready.")
                    break
                time.sleep(2)
        else:
            print(f"[TEMP REGULATED] CPUSS currently at {cur_t:.1f} C (target: {target_temp_c:.1f} C).")

    def enforce_cooldown(self, target_temp_c: float, max_wait_s: int = 90):
        """
        Enforces that the SoC cools down below target_temp_c (or safe cool ceiling 36.5C).
        """
        ceiling = min(target_temp_c, 36.5)
        t_start = time.time()
        print(f"[COOLDOWN] Enforcing barrier: waiting for CPUSS temp <= {ceiling:.1f} C...")
        while time.time() - t_start < max_wait_s:
            cur_t = self.get_cpuss_temp()
            if cur_t <= ceiling:
                print(f"[COOLDOWN] Settled at {cur_t:.1f} C. Resting 2s...")
                time.sleep(2)
                return True
            time.sleep(2)
        print(f"[COOLDOWN WARNING] Timed out waiting for {ceiling:.1f} C (current: {self.get_cpuss_temp():.1f} C). Continuing.")
        time.sleep(1)
        return False

    def prewarm_if_needed(self, target_temp_c: float, timeout_s: int = 45):
        self.regulate_temperature(target_temp_c, timeout_s=timeout_s)

    def run_trial(self, level: str, target_temp_c: float, duration_s: int, trial_num: int, out_csv: str) -> str:
        cmd = f"{self.binary_path} --level {level} --temp {target_temp_c} --duration {duration_s} --csv {out_csv} --trial {trial_num}"
        print(f"[WORKLOAD] Executing: {cmd}")
        stdout = self._adb(cmd, timeout=duration_s + 20)
        return stdout

    def kill_all(self):
        try:
            subprocess.run(["adb", "-s", self.device_id, "shell", "killall -9 qidk_load_generator snpe-net-run 2>/dev/null"],
                           capture_output=True, timeout=5)
        except Exception:
            pass
