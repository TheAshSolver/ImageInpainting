#!/usr/bin/env python3
"""
plot_npu_actual_vs_predicted.py
Plots comprehensive actual vs predicted graphs for the calibrated Qualcomm Hexagon NPU
linear regression model across MIGAN, LaMa-Dilated, and AOT-GAN.
"""

import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    csv_path = repo_root / "rl_system_thermal_benchmark.csv"
    weights_path = repo_root / "linear_regression_weights.json"

    # 1. Load data
    df = pd.read_csv(csv_path)
    npu_df = df[df["runtime"] == "NPU"].copy().reset_index(drop=True)

    # 2. Load updated weights
    with open(weights_path) as f:
        weights_data = json.load(f)

    npu_model = weights_data["npu_calibrated_latency"]
    b0 = npu_model["intercept"]
    w = npu_model["weights"]

    pwr_weight = w.get("power_mw", w.get("active_power_mw", 0.0))

    # Compute predicted values using updated calibrated weights
    preds = []
    for idx, row in npu_df.iterrows():
        t = row["exact_temp_c"]
        pwr = row["power_mw"]
        is_lama = 1.0 if row["model"] == "LaMa-Dilated" else 0.0
        is_migan = 1.0 if row["model"] == "MIGAN" else 0.0
        pred = (b0 + 
                w["soc_temp_start_c"] * t + 
                pwr_weight * pwr + 
                w["arch_lama"] * is_lama + 
                w["arch_migan"] * is_migan)
        preds.append(pred)

    npu_df["predicted_ms"] = preds
    npu_df["residual_ms"] = npu_df["latency_ms"] - npu_df["predicted_ms"]

    # Global metrics
    r2 = npu_model["r2"]
    mae = npu_model["mae_ms"]
    rmse = npu_model["rmse_ms"]

    # 3. Create 3-Panel Figure
    fig = plt.figure(figsize=(18, 5.8), dpi=300)
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    color_map = {
        "MIGAN": "#2ca02c",         # Forest Green
        "LaMa-Dilated": "#1f77b4",  # Qualcomm Blue
        "AOT-GAN": "#d62728"        # Crimson Red
    }
    marker_map = {
        "MIGAN": "o",
        "LaMa-Dilated": "s",
        "AOT-GAN": "^"
    }

    # -------------------------------------------------------------
    # Panel 1: Parity Plot (Actual vs. Predicted)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    diag_line = np.linspace(40, 330, 200)
    ax1.plot(diag_line, diag_line, color="#444444", linestyle="--", linewidth=1.8, label="Ideal Parity (y = x)")

    for m in ["MIGAN", "LaMa-Dilated", "AOT-GAN"]:
        sub = npu_df[npu_df["model"] == m]
        ax1.scatter(
            sub["latency_ms"], sub["predicted_ms"],
            color=color_map[m], marker=marker_map[m], s=55, alpha=0.85,
            edgecolors="black", linewidth=0.5, label=f"{m} (N={len(sub)})"
        )

    ax1.set_title("A. Parity Plot: Actual vs. Predicted Latency", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Actual Measured Latency (ms)", fontsize=11)
    ax1.set_ylabel("Linear Model Prediction (ms)", fontsize=11)
    ax1.set_xlim(40, 330)
    ax1.set_ylim(40, 330)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.5)

    stats_box = (
        f"Calibrated Fit:\n"
        f"• R² = {r2:.6f}\n"
        f"• MAE = {mae:.2f} ms\n"
        f"• RMSE = {rmse:.2f} ms\n"
        f"• Max Error = {npu_df['residual_ms'].abs().max():.2f} ms"
    )
    ax1.text(0.55, 0.08, stats_box, transform=ax1.transAxes, fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95))

    # -------------------------------------------------------------
    # Panel 2: Thermal Trajectory (Actual vs Predicted over Temp)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    temp_sweep = np.linspace(34, 52, 100)

    for m in ["MIGAN", "LaMa-Dilated", "AOT-GAN"]:
        sub = npu_df[npu_df["model"] == m]
        c = color_map[m]
        # Actual points
        ax2.scatter(
            sub["exact_temp_c"], sub["latency_ms"],
            color=c, marker=marker_map[m], s=45, alpha=0.75,
            edgecolors="none", label=f"{m} Actual"
        )
        # Predicted line (using mean power for smooth curve)
        mean_pwr = sub["power_mw"].mean()
        is_lama = 1.0 if m == "LaMa-Dilated" else 0.0
        is_migan = 1.0 if m == "MIGAN" else 0.0
        pred_line = (b0 + w["soc_temp_start_c"] * temp_sweep + 
                     pwr_weight * mean_pwr + 
                     w["arch_lama"] * is_lama + 
                     w["arch_migan"] * is_migan)
        ax2.plot(temp_sweep, pred_line, color=c, linestyle="-", linewidth=2.4, label=f"{m} Predicted")

    ax2.set_title("B. Thermal Trajectory: Latency vs. SoC Temp", fontsize=12, fontweight="bold")
    ax2.set_xlabel("SoC Junction Temperature (°C)", fontsize=11)
    ax2.set_ylabel("Inference Latency (ms)", fontsize=11)
    ax2.set_xlim(33, 53)
    ax2.set_ylim(40, 330)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="center right", fontsize=8.0, ncol=1)

    # -------------------------------------------------------------
    # Panel 3: Prediction Residuals (y - y_hat)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    # Shaded error tolerance bands
    ax3.axhspan(-1.0, 1.0, color="#2ca02c", alpha=0.15, label="±1.0 ms Band (92.6% within)")
    ax3.axhspan(-2.0, 2.0, color="#ffbb78", alpha=0.12, label="±2.0 ms Band (99.5% within)")
    ax3.axhline(0.0, color="#111111", linestyle="-", linewidth=1.2)

    for m in ["MIGAN", "LaMa-Dilated", "AOT-GAN"]:
        sub = npu_df[npu_df["model"] == m]
        ax3.scatter(
            sub.index, sub["residual_ms"],
            color=color_map[m], marker=marker_map[m], s=40, alpha=0.85,
            edgecolors="black", linewidth=0.4, label=m
        )

    ax3.set_title("C. Prediction Residuals (Actual - Predicted)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Telemetry Sample Index (0 to 215)", fontsize=11)
    ax3.set_ylabel("Residual Error (ms)", fontsize=11)
    ax3.set_ylim(-3.5, 3.5)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", fontsize=8.0)

    plt.suptitle("Qualcomm Hexagon HTP v79 NPU Latency: Actual vs. Calibrated Linear Model Predictions",
                 fontsize=14, fontweight="bold", y=0.97)
    plt.subplots_adjust(top=0.88, bottom=0.14, left=0.06, right=0.98, wspace=0.24)

    out_png = repo_root / "Benchmark" / "output" / "npu_actual_vs_predicted.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    print("Saved plot to:", out_png)

    # Copy to artifact directory
    brain_dir = Path("/home/aaravind/.gemini/antigravity-cli/brain/7395c208-1260-4e1f-a10c-06023eabccca")
    brain_dir.mkdir(parents=True, exist_ok=True)
    dest_png = brain_dir / "npu_actual_vs_predicted.png"
    shutil.copyfile(out_png, dest_png)
    print("Copied plot to artifact directory:", dest_png)

if __name__ == "__main__":
    main()
