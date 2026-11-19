#!/usr/bin/env python3
"""
scripts/generate_correlation_analysis.py
================================================================================
🔬 MULTI-VARIABLE CORRELATION & THERMAL HYSTERESIS ANALYSIS
Target Platform: Qualcomm Snapdragon 8 Elite (Hexagon HTP v79 NPU & Adreno 830 GPU)
================================================================================

This module performs statistical correlation analysis across all measured system
parameters from the Strategy B empirical telemetry dataset:
  1. Computes Pearson & Spearman correlation matrices.
  2. Generates correlation heatmaps with annotated r-values and significance asterisks.
  3. Visualizes thermal hysteresis loops (Ramp-Up vs Ramp-Down divergence).
  4. Exports structured correlation tables to CSV for presentation and paper inclusion.

Usage:
  python scripts/generate_correlation_analysis.py
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "Benchmark" / "output" / "rl_system_thermal_benchmark.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Benchmark" / "output" / "rl_correlation_figures"
DEFAULT_CORR_CSV = REPO_ROOT / "Benchmark" / "output" / "rl_correlation_matrix.csv"


def compute_correlations_with_pvalues(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes correlation matrix and corresponding two-tailed p-values."""
    n = len(cols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    pval_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)

    for i in range(n):
        for j in range(n):
            c1, c2 = cols[i], cols[j]
            if i == j:
                corr_matrix.loc[c1, c2] = 1.0
                pval_matrix.loc[c1, c2] = 0.0
                continue
            v1 = df[c1].to_numpy()
            v2 = df[c2].to_numpy()
            mask = ~np.isnan(v1) & ~np.isnan(v2)
            clean1, clean2 = v1[mask], v2[mask]
            if len(clean1) > 2 and float(np.std(clean1)) > 1e-9 and float(np.std(clean2)) > 1e-9:
                r, p = stats.pearsonr(clean1, clean2)
                corr_matrix.loc[c1, c2] = float(r)
                pval_matrix.loc[c1, c2] = float(p)
            else:
                corr_matrix.loc[c1, c2] = 0.0
                pval_matrix.loc[c1, c2] = 1.0

    return corr_matrix, pval_matrix


def plot_heatmap(
    corr: pd.DataFrame,
    pvals: pd.DataFrame,
    labels: List[str],
    title: str,
    output_path: Path
) -> None:
    """Renders formatted correlation heatmap with significance stars."""
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    data = corr.to_numpy()
    im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation Coefficient (r)", rotation=270, labelpad=15, weight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", weight="bold")
    ax.set_yticklabels(labels, weight="bold")

    # Add text annotations with significance
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data[i, j]
            pval = pvals.iloc[i, j]
            star = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
            txt_color = "white" if abs(val) > 0.55 else "black"
            text_str = f"{val:.2f}\n{star}" if star else f"{val:.2f}"
            ax.text(j, i, text_str, ha="center", va="center", color=txt_color, fontsize=8.5, weight="bold")

    ax.set_title(title, weight="bold", pad=15, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_hysteresis_loops(df: pd.DataFrame, output_path: Path) -> None:
    """Visualizes directional hysteresis: Ramp-Up (Heating) vs Ramp-Down (Cooling)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for (m, r), grp in df.groupby(["model", "runtime"]):
        for d in ["up", "down"]:
            sub = grp[grp["direction"] == d]
            if sub.empty:
                continue
            deg_sum = sub.groupby("temp_deg")[["latency_ms", "power_mw"]].mean().reset_index()
            deg_sum = deg_sum.sort_values("temp_deg")
            
            linestyle = "-" if d == "up" else "--"
            marker = "^" if d == "up" else "v"
            label = f"{m} ({r}) {d.upper()}"

            ax1.plot(deg_sum["temp_deg"], deg_sum["latency_ms"], label=label, linestyle=linestyle, marker=marker, markersize=4)
            ax2.plot(deg_sum["temp_deg"], deg_sum["power_mw"], label=label, linestyle=linestyle, marker=marker, markersize=4)

    ax1.set_title("A. Latency Hysteresis: Heating vs. Cooling", weight="bold")
    ax1.set_xlabel("Silicon Temperature (°C)")
    ax1.set_ylabel("Inference Latency (ms)")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="best")

    ax2.set_title("B. Power Draw Hysteresis: Heating vs. Cooling", weight="bold")
    ax2.set_xlabel("Silicon Temperature (°C)")
    ax2.set_ylabel("Power Draw (mW)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="best")

    fig.suptitle("Strategy B Thermal Hysteresis Dynamics on Snapdragon 8 Elite", weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_correlation_pipeline(csv_path: Path, output_dir: Path, output_csv: Path) -> None:
    if not csv_path.exists():
        sys.exit(f"❌ Error: CSV not found at {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df["log_latency"] = np.log(df["latency_ms"])

    features = [
        "exact_temp_c",
        "latency_ms",
        "power_mw",
        "energy_mj",
        "ram_rss_mb",
        "battery_v",
        "npu_usage_pct",
        "gpu_usage_pct"
    ]
    labels = [
        "Temp (°C)",
        "Latency (ms)",
        "Power (mW)",
        "Energy (mJ)",
        "RAM RSS (MB)",
        "Battery (V)",
        "NPU Usage %",
        "GPU Usage %"
    ]

    # 1. Pearson
    p_corr, p_pvals = compute_correlations_with_pvalues(df, features)
    f1 = output_dir / "01_pearson_correlation_heatmap.png"
    plot_heatmap(p_corr, p_pvals, labels, "Pearson Linear Correlation Matrix (*p<0.05, **p<0.01, ***p<0.001)", f1)
    print(f"✅ Generated Pearson heatmap: {f1}")

    # 2. Spearman
    s_corr = df[features].corr(method="spearman")
    s_pvals = pd.DataFrame(np.zeros((len(features), len(features))), index=features, columns=features)
    for i, c1 in enumerate(features):
        for j, c2 in enumerate(features):
            _, p = stats.spearmanr(df[c1], df[c2])
            s_pvals.iloc[i, j] = p
    f2 = output_dir / "02_spearman_correlation_heatmap.png"
    plot_heatmap(s_corr, s_pvals, labels, "Spearman Monotonic Rank Correlation Matrix", f2)
    print(f"✅ Generated Spearman heatmap: {f2}")

    # 3. Hysteresis Loops
    f3 = output_dir / "03_thermal_hysteresis_loops.png"
    plot_hysteresis_loops(df, f3)
    print(f"✅ Generated Thermal Hysteresis chart: {f3}")

    # Export CSV
    p_corr.columns = labels
    p_corr.index = labels
    p_corr.to_csv(output_csv)
    print(f"✅ Exported numerical correlation matrix to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation & Hysteresis Analysis")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CORR_CSV)
    args = parser.parse_args()

    run_correlation_pipeline(args.csv, args.output_dir, args.output_csv)


if __name__ == "__main__":
    main()
