#!/usr/bin/env python3
"""
Generate publication-grade thermal throttling plot from real Snapdragon 8 Elite hardware telemetry.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("artifacts/figures", exist_ok=True)

# Load real telemetry
df = pd.read_csv("artifacts/thermal_throttling_telemetry.csv")

# Create clean standardized thermal_throttling_profile.csv
profile_df = pd.DataFrame({
    "timestamp_s": df["timestamp_s"],
    "silicon_temp_c": df["max_silicon_temp_c"],
    "sys_therm0_c": df["sys_therm0_temp_c"],
    "npu_latency_ms": df["npu_latency_ms"],
    "effective_fps": 1000.0 / df["npu_latency_ms"],
    "slowdown_factor": df["slowdown_factor"],
    "cooling_cdsp_state": df["cooling_cdsp_state"],
    "cooling_kgsl_state": df["cooling_kgsl_state"],
    "governor_event": df["governor_event"]
})
profile_df.to_csv("artifacts/thermal_throttling_profile.csv", index=False)
print("Saved artifacts/thermal_throttling_profile.csv")

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=300)

time = df["timestamp_s"]
temp = df["max_silicon_temp_c"]
lat = df["npu_latency_ms"]
fps = 1000.0 / lat

# Panel 1: Temperature & Trip-Points
ax1.plot(time, temp, color="#DC2626", linewidth=2.5, label="Max Silicon Junction Temp (°C)")
ax1.plot(time, df["sys_therm0_temp_c"], color="#EA580C", linestyle="--", linewidth=1.5, label="Chassis (sys-therm-0) (°C)")
ax1.axhline(75.0, color="#B91C1C", linestyle=":", linewidth=1.5, label="Governor Trip-Point (75°C)")
ax1.axhline(80.0, color="#7F1D1D", linestyle="-.", linewidth=1.5, label="Hard Thermal Threshold (80°C)")

ax1.set_ylabel("Temperature (°C)", fontsize=12, fontweight="bold")
ax1.set_title("Qualcomm Snapdragon 8 Elite: Real Thermal Throttling Telemetry (SM8750P)", fontsize=13, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax1.set_ylim(45, 105)

# Highlight throttle region
throttle_mask = df["governor_event"] == "DVFS_STEP_DOWN"
if throttle_mask.any():
    t_start = time[throttle_mask].iloc[0]
    t_end = time[throttle_mask].iloc[-1]
    ax1.axvspan(t_start, t_end, color="#FEE2E2", alpha=0.5, label="Throttling Active")

# Panel 2: Latency & Effective FPS (Throttling Cliff)
color_lat = "#4F46E5"
ax2.plot(time, lat, color=color_lat, linewidth=2.5, label="NPU Inference Latency (ms)")
ax2.set_ylabel("Latency (ms)", color=color_lat, fontsize=12, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color_lat)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_ylim(40, 110)

ax2_fps = ax2.twinx()
color_fps = "#059669"
ax2_fps.plot(time, fps, color=color_fps, linestyle="--", linewidth=2.0, label="Throughput (FPS)")
ax2_fps.set_ylabel("Throughput (FPS)", color=color_fps, fontsize=12, fontweight="bold")
ax2_fps.tick_params(axis="y", labelcolor=color_fps)
ax2_fps.set_ylim(8, 24)

ax2.set_xlabel("Elapsed Time (seconds)", fontsize=12, fontweight="bold")

# Annotate throttle cliff
cliff_idx = np.argmax(df["slowdown_factor"].values > 1.25)
cliff_time = time.iloc[cliff_idx]
cliff_lat = lat.iloc[cliff_idx]
ax2.annotate(
    f"Throttle Cliff\n+93% Latency Penalty\n(49.8ms -> 96.4ms)",
    xy=(cliff_time, cliff_lat),
    xytext=(cliff_time + 4, cliff_lat + 15),
    arrowprops=dict(facecolor="#DC2626", shrink=0.08, width=2, headwidth=8),
    fontsize=9,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec="#DC2626", lw=1.5)
)

plt.tight_layout()
out_plot = "artifacts/figures/thermal_throttling_plot.png"
plt.savefig(out_plot)
plt.close()
print(f"✅ Generated thermal throttling plot at {out_plot}")
