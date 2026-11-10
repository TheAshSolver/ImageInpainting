#!/usr/bin/env python3
"""
plot_npu_latency_fit.py
Plots empirical NPU latency data from rl_system_thermal_benchmark.csv against
both the prior linear regression model parameters and refitted linear curves.
"""

import json
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    csv_path = os.path.join(repo_root, "rl_system_thermal_benchmark.csv")
    weights_path = os.path.join(repo_root, "linear_regression_weights.json")

    # 1. Load data
    df = pd.read_csv(csv_path)
    npu_df = df[df["runtime"] == "NPU"].copy()

    # 2. Load existing linear regression weights
    with open(weights_path) as f:
        weights_json = json.load(f)

    old_pure = weights_json["gan_pure_compute_latency"]
    w = old_pure["weights"]
    b0 = old_pure["intercept"]

    # Calculate old model predictions
    old_preds = []
    for idx, row in npu_df.iterrows():
        t = row["exact_temp_c"]
        dt = 0.0
        pwr = row["power_mw"] / 1000.0
        is_npu = 1.0
        lama = 1.0 if row["model"] == "LaMa-Dilated" else 0.0
        migan = 1.0 if row["model"] == "MIGAN" else 0.0
        p = (b0 + w["soc_temp_start_c"] * t + w["delta_temp_c"] * dt + 
             w["active_power_w"] * pwr + w["is_npu"] * is_npu + 
             w["arch_lama"] * lama + w["arch_migan"] * migan)
        old_preds.append(p)

    npu_df["pred_old"] = old_preds

    # 3. Create plots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    colors = {"MIGAN": "#2ca02c", "LaMa-Dilated": "#1f77b4", "AOT-GAN": "#d62728"}

    # Subplot 1: All 3 models global comparison
    ax1 = axs[0, 0]
    t_range = np.linspace(34, 52, 100)

    for m in ["MIGAN", "LaMa-Dilated", "AOT-GAN"]:
        sub = npu_df[npu_df["model"] == m]
        c = colors[m]
        
        # Scatter of measured points
        ax1.scatter(sub["exact_temp_c"], sub["latency_ms"], color=c, alpha=0.75, 
                    edgecolors="none", s=40, label=f"{m} Measured (N={len(sub)})")
        
        # Fit new linear regression
        poly = np.polyfit(sub["exact_temp_c"], sub["latency_ms"], deg=1)
        ax1.plot(t_range, np.polyval(poly, t_range), color=c, linestyle="-", 
                 linewidth=2.5, label=f"{m} Linear Fit (slope={poly[0]:.3f})")
        
        # Old model curve
        lama_flag = 1.0 if m == "LaMa-Dilated" else 0.0
        migan_flag = 1.0 if m == "MIGAN" else 0.0
        old_curve = (b0 + w["soc_temp_start_c"] * t_range + w["is_npu"] * 1.0 + 
                     w["arch_lama"] * lama_flag + w["arch_migan"] * migan_flag)
        ax1.plot(t_range, old_curve, color=c, linestyle="--", linewidth=1.8, 
                 alpha=0.7, label=f"{m} Old Model (slope=+1.35)")

    ax1.set_title("Hexagon NPU Inpainting Latency vs SoC Temperature", fontsize=13, fontweight="bold")
    ax1.set_xlabel("SoC Junction Temperature (°C)", fontsize=11)
    ax1.set_ylabel("Inference Latency (ms)", fontsize=11)
    ax1.set_xlim(33, 53)
    ax1.set_ylim(40, 340)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=8.0, ncol=2)

    # Subplots 2, 3, 4: Individual zoom-ins per model
    models_meta = [
        ("MIGAN", axs[0, 1], (50, 90)),
        ("LaMa-Dilated", axs[1, 0], (150, 220)),
        ("AOT-GAN", axs[1, 1], (230, 325))
    ]

    for m, ax, ylim in models_meta:
        sub = npu_df[npu_df["model"] == m]
        c = colors[m]
        t = sub["exact_temp_c"]
        lat = sub["latency_ms"]
        
        # Raw points
        ax.scatter(t, lat, color=c, alpha=0.8, s=55, edgecolors="black", 
                   linewidth=0.5, label=f"Measured Telemetry (N={len(sub)})")
        
        # Refitted linear fit
        poly = np.polyfit(t, lat, deg=1)
        poly_slope = poly[0]
        poly_intercept = poly[1]
        mae_new = np.mean(np.abs(lat - np.polyval(poly, t)))
        t_span = np.linspace(t.min() - 1, t.max() + 1, 50)
        ax.plot(t_span, np.polyval(poly, t_span), color="#111111", linestyle="-", linewidth=2.5, 
                label=f"Refitted Linear: y = {poly_intercept:.2f} + {poly_slope:+.4f}·T (MAE={mae_new:.2f}ms)")
        
        # Old model prediction line
        lama_flag = 1.0 if m == "LaMa-Dilated" else 0.0
        migan_flag = 1.0 if m == "MIGAN" else 0.0
        old_intercept_eff = b0 + w["is_npu"] + w["arch_lama"] * lama_flag + w["arch_migan"] * migan_flag
        old_pred_line = old_intercept_eff + w["soc_temp_start_c"] * t_span
        mae_old = np.mean(np.abs(lat - sub["pred_old"]))
        ax.plot(t_span, old_pred_line, color="#e65100", linestyle="--", linewidth=2.2, 
                label=f"Old Model: y = {old_intercept_eff:.1f} + {w['soc_temp_start_c']:.2f}·T (MAE={mae_old:.2f}ms)")
        
        # Annotate stats box
        bias_old = sub["pred_old"].mean() - lat.mean()
        rel_err_new = (mae_new / lat.mean()) * 100
        stats_text = (
            f"Actual Latency: {lat.mean():.2f} ± {lat.std():.2f} ms\n"
            f"Old Model MAE: {mae_old:.2f} ms (Bias: {bias_old:+.2f} ms)\n"
            f"New Fit MAE: {mae_new:.2f} ms (Rel Error: {rel_err_new:.2f}%)"
        )
        ax.text(0.04, 0.06, stats_text, transform=ax.transAxes, fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.95))
        
        ax.set_title(f"{m} (Hexagon HTP v79 NPU)", fontsize=12, fontweight="bold")
        ax.set_xlabel("SoC Junction Temperature (°C)", fontsize=10.5)
        ax.set_ylabel("Inference Latency (ms)", fontsize=10.5)
        ax.set_ylim(ylim)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8.5)

    plt.suptitle("Qualcomm Hexagon NPU Latency: Empirical Data Points vs Linear Regression Models", 
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_path = os.path.join(repo_root, "Benchmark", "output", "npu_latency_linear_regression_fit.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print("Saved plot to:", out_path)

    # Copy to brain artifact directory
    brain_dir = "/home/aaravind/.gemini/antigravity-cli/brain/7395c208-1260-4e1f-a10c-06023eabccca"
    os.makedirs(brain_dir, exist_ok=True)
    dest_path = os.path.join(brain_dir, "npu_latency_linear_regression_fit.png")
    shutil.copyfile(out_path, dest_path)
    print("Copied plot to artifact dir:", dest_path)

if __name__ == "__main__":
    main()
