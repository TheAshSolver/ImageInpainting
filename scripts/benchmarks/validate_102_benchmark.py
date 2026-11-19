#!/usr/bin/env python3
"""
Validate Meta-Predictor on Independent 102-Image NPU Benchmark Dataset
Extracts context vectors from Benchmark/input_102/, runs inference using
meta_predictor_bundle.pkl (trained on 6,000 images), compares against
actual measured NPU values from fresh_benchmark_2026-10-04, and generates
parity graphs and statistical validation reports.
"""

import os
import sys
import pickle
import json
import time
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_102_DIR = ROOT_DIR / "Benchmark/input_102"
ACTUAL_CSV_DIR = ROOT_DIR / "Benchmark/output_old/fresh_benchmark_2026-10-04/per_sample_csv"
BUNDLE_PATH = ROOT_DIR / "Benchmark/output/6000_benchmark/meta_predictor_bundle.pkl"
OUTPUT_DIR = ROOT_DIR / "Benchmark/output/6000_benchmark"
BRAIN_DIR = Path("/home/aaravind/.gemini/antigravity-cli/brain/7395c208-1260-4e1f-a10c-06023eabccca")

def extract_image_features(img_np: np.ndarray) -> dict:
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = round(float(laplacian.var()), 2)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = round(float(np.sum(edges > 0)) / float(edges.size), 4)

    mean_luminance = round(float(np.mean(gray)), 2)
    std_contrast = round(float(np.std(gray)), 2)

    ent = 0.0
    for c in range(3):
        hist, _ = np.histogram(img_np[:, :, c], bins=32, range=(0, 256), density=True)
        hist = hist[hist > 0]
        ent += -np.sum(hist * np.log2(hist))
    color_entropy = round(float(ent / 3.0), 3)

    return {
        "img_laplacian_var": laplacian_var,
        "img_canny_edge_density": edge_density,
        "img_mean_luminance": mean_luminance,
        "img_std_contrast": std_contrast,
        "img_color_entropy": color_entropy
    }

def compute_mask_features(mask: np.ndarray) -> dict:
    h, w = mask.shape
    hole_pixels = np.sum(mask > 127)
    total_pixels = h * w
    coverage_pct = round(float(hole_pixels) / float(total_pixels) * 100.0, 3)

    num_labels, labels = cv2.connectedComponents((mask > 127).astype(np.uint8))
    actual_holes = max(1, num_labels - 1)

    y_coords, x_coords = np.where(mask > 127)
    if len(x_coords) > 0:
        centroid_x = round(float(np.mean(x_coords)) / w, 4)
        centroid_y = round(float(np.mean(y_coords)) / h, 4)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        dist_to_border = round(float(min(min_x, w - 1 - max_x, min_y, h - 1 - max_y)) / min(w, h), 4)
        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        aspect_ratio = round(float(bbox_w) / max(1.0, float(bbox_h)), 4)
    else:
        centroid_x, centroid_y = 0.5, 0.5
        dist_to_border = 0.5
        aspect_ratio = 1.0

    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    perimeter_to_area = round(float(perimeter) / max(1.0, np.sqrt(hole_pixels)), 4) if hole_pixels > 0 else 0.0

    return {
        "mask_coverage_pct": coverage_pct,
        "mask_hole_count": actual_holes,
        "mask_centroid_x": centroid_x,
        "mask_centroid_y": centroid_y,
        "mask_dist_to_border": dist_to_border,
        "mask_aspect_ratio": aspect_ratio,
        "mask_perimeter_to_area": perimeter_to_area
    }

def engineer_features(df_features: pd.DataFrame) -> pd.DataFrame:
    df_out = df_features.copy()
    c = np.clip(df_out["mask_coverage_pct"].values / 100.0, 1e-4, 1.0)
    H = np.clip(df_out["mask_hole_count"].values, 1.0, 10.0)
    pa = df_out["mask_perimeter_to_area"].values
    edge = df_out["img_canny_edge_density"].values

    df_out["r_eff"] = np.sqrt(c / H)
    df_out["log_c"] = np.log10(c)
    df_out["c_times_pa"] = c * pa
    df_out["c_times_edge"] = c * edge
    df_out["hole_density"] = H / (c * 100.0)
    return df_out

def main():
    print("=" * 80)
    print("🔬 OUT-OF-SAMPLE VALIDATION ON 102-IMAGE INDEPENDENT BENCHMARK")
    print("=" * 80)

    # 1. Load trained bundle
    if not BUNDLE_PATH.exists():
        print(f"Error: Bundle {BUNDLE_PATH} not found!")
        return 1

    with open(BUNDLE_PATH, "rb") as f:
        bundle = pickle.load(f)
    print(f"✅ Loaded Meta-Predictor Bundle trained on {bundle.get('trained_samples_count', 6000)} samples")

    # 2. Load ground truth actual NPU metrics for 102 images
    actual_dfs = {
        "migan_npu": pd.read_csv(ACTUAL_CSV_DIR / "quality_migan_npu_lpips.csv"),
        "lama_npu": pd.read_csv(ACTUAL_CSV_DIR / "quality_lama_npu_lpips.csv"),
        "aotgan_npu": pd.read_csv(ACTUAL_CSV_DIR / "quality_aotgan_npu_lpips.csv")
    }

    # 3. Extract features for all 102 images
    sample_records = []
    print("Extracting visual & geometric features from 102 input pairs...")
    t0 = time.time()
    for i in range(1, 103):
        stem = f"{i:03d}"
        img_p = INPUT_102_DIR / "ground_truth" / f"{stem}.png"
        mask_p = INPUT_102_DIR / "mask" / f"{stem}.png"

        img = np.asarray(Image.open(img_p).convert("RGB"))
        mask = np.asarray(Image.open(mask_p).convert("L"))

        i_feats = extract_image_features(img)
        m_feats = compute_mask_features(mask)

        rec = {"sample_idx": i, "sample_stem": stem}
        rec.update(i_feats)
        rec.update(m_feats)
        sample_records.append(rec)

    feat_df = pd.DataFrame(sample_records)
    print(f"✅ Feature extraction completed in {time.time() - t0:.2f}s (Avg {((time.time() - t0)/102)*1000:.1f}ms/image)")

    # 4. Engineer features matching bundle spec
    feat_eng = engineer_features(feat_df)
    eng_cols = bundle["engineered_feature_cols"]
    X_input = feat_eng[eng_cols].values

    # 5. Predict quality targets for each model
    models_to_test = ["migan_npu", "lama_npu", "aotgan_npu"]
    targets_to_test = ["ssim", "lpips", "psnr_hole", "psnr_full"]

    results = {}
    validation_table_rows = []

    for m_key in models_to_test:
        actual_df = actual_dfs[m_key]
        results[m_key] = {}

        # Align actual values
        y_act_ssim = actual_df["ssim"].values
        y_act_lpips = actual_df["lpips_vgg"].values
        y_act_psnr_hole = actual_df["psnr_hole"].values
        y_act_psnr_full = actual_df["psnr_full"].values

        # Run model predictions
        pred_ssim = bundle["models"][m_key]["ssim"]["model"].predict(X_input)
        pred_lpips = bundle["models"][m_key]["lpips"]["model"].predict(X_input)
        pred_psnr_hole = bundle["models"][m_key]["psnr_hole"]["model"].predict(X_input)
        pred_psnr_full = bundle["models"][m_key]["psnr_full"]["model"].predict(X_input)

        results[m_key]["ssim"] = {"actual": y_act_ssim, "predicted": pred_ssim}
        results[m_key]["lpips"] = {"actual": y_act_lpips, "predicted": pred_lpips}
        results[m_key]["psnr_hole"] = {"actual": y_act_psnr_hole, "predicted": pred_psnr_hole}
        results[m_key]["psnr_full"] = {"actual": y_act_psnr_full, "predicted": pred_psnr_full}

        for i in range(102):
            validation_table_rows.append({
                "sample_idx": i + 1,
                "model": m_key,
                "actual_ssim": y_act_ssim[i],
                "pred_ssim": pred_ssim[i],
                "actual_lpips": y_act_lpips[i],
                "pred_lpips": pred_lpips[i],
                "actual_psnr_hole": y_act_psnr_hole[i],
                "pred_psnr_hole": pred_psnr_hole[i],
                "actual_psnr_full": y_act_psnr_full[i],
                "pred_psnr_full": pred_psnr_full[i]
            })

    val_df = pd.DataFrame(validation_table_rows)
    val_csv_path = OUTPUT_DIR / "validation_102_predictions_detailed.csv"
    val_df.to_csv(val_csv_path, index=False)
    print(f"✅ Saved detailed sample-by-sample comparisons to: {val_csv_path}")

    # 6. Compute Statistical Validation Metrics
    print("\n" + "=" * 80)
    print("📊 OUT-OF-SAMPLE VALIDATION METRICS (102 INDEPENDENT TEST IMAGES)")
    print("=" * 80)

    stats_summary = {}
    for m_key in models_to_test:
        m_title = m_key.replace("_npu", "").upper()
        stats_summary[m_key] = {}
        print(f"\n--- {m_title} (NPU) ---")
        for t in targets_to_test:
            y_a = results[m_key][t]["actual"]
            y_p = results[m_key][t]["predicted"]
            r2 = float(r2_score(y_a, y_p))
            rmse = float(np.sqrt(mean_squared_error(y_a, y_p)))
            mae = float(mean_absolute_error(y_a, y_p))
            stats_summary[m_key][t] = {"r2": r2, "rmse": rmse, "mae": mae}
            print(f"  • {t:12s} | R²: {r2:6.4f} | RMSE: {rmse:6.4f} | MAE: {mae:6.4f}")

    # 7. Generate Comprehensive 4x3 Multi-Panel Parity Plots
    print("\n" + "-" * 80)
    print("🎨 GENERATING 4x3 PUBLICATION-QUALITY PARITY CHARTS")
    print("-" * 80)

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.patch.set_facecolor("#0b0f19")

    metrics_config = [
        ("ssim", "SSIM (Structural Similarity)", "#10b981", [0.4, 1.0]),
        ("lpips", "LPIPS (Perceptual Loss)", "#f59e0b", [0.0, 0.35]),
        ("psnr_hole", "PSNR Hole (dB)", "#38bdf8", [10.0, 36.0]),
        ("psnr_full", "PSNR Full (dB)", "#818cf8", [15.0, 42.0])
    ]

    for row_idx, (t_key, t_label, dot_color, lims) in enumerate(metrics_config):
        for col_idx, m_key in enumerate(models_to_test):
            m_title = m_key.replace("_npu", "").upper()
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#161e2e")

            y_act = results[m_key][t_key]["actual"]
            y_pred = results[m_key][t_key]["predicted"]
            st = stats_summary[m_key][t_key]

            # Scatter points
            ax.scatter(y_act, y_pred, alpha=0.75, s=42, color=dot_color, edgecolors="#ffffff", linewidths=0.5, label="102 Test Samples")

            # Ideal parity line (y = x)
            min_v = min(lims[0], y_act.min(), y_pred.min())
            max_v = max(lims[1], y_act.max(), y_pred.max())
            ax.plot([min_v, max_v], [min_v, max_v], color="#ef4444", linestyle="--", linewidth=1.8, label="Ideal (y = x)")

            # Best fit regression line
            slope, intercept = np.polyfit(y_act, y_pred, 1)
            x_fit = np.linspace(min_v, max_v, 50)
            ax.plot(x_fit, slope * x_fit + intercept, color="#e2e8f0", linestyle=":", linewidth=1.2, label=f"Fit (Slope: {slope:.2f})")

            # Titles & Labels
            ax.set_title(f"{m_title} — {t_label}\nR² = {st['r2']:.3f} | RMSE = {st['rmse']:.3f} | MAE = {st['mae']:.3f}",
                         color="#f8fafc", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(f"Actual Measured NPU {t_label}", color="#94a3b8", fontsize=10)
            ax.set_ylabel(f"Predicted {t_label}", color="#94a3b8", fontsize=10)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.grid(True, color="#26334d", linestyle=":", alpha=0.6)
            ax.legend(facecolor="#161e2e", edgecolor="#26334d", labelcolor="#f8fafc", loc="upper left", fontsize=8)

            ax.set_xlim(lims)
            ax.set_ylim(lims)

    plt.tight_layout(pad=3.0)
    parity_chart_path = OUTPUT_DIR / "validation_102_actual_vs_predicted_parity.png"
    plt.savefig(parity_chart_path, dpi=200, bbox_inches="tight")
    if BRAIN_DIR.exists():
        shutil.copy(parity_chart_path, BRAIN_DIR / "validation_102_actual_vs_predicted_parity.png")
    plt.close()
    print(f"✅ Generated & Saved: {parity_chart_path}")

    # 8. Generate Residual Distribution Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0b0f19")

    for col_idx, m_key in enumerate(models_to_test):
        m_title = m_key.replace("_npu", "").upper()
        ax = axes[col_idx]
        ax.set_facecolor("#161e2e")

        res_ssim = results[m_key]["ssim"]["predicted"] - results[m_key]["ssim"]["actual"]
        res_lpips = results[m_key]["lpips"]["predicted"] - results[m_key]["lpips"]["actual"]

        ax.hist(res_ssim, bins=20, alpha=0.65, color="#10b981", label=f"SSIM Error (μ={np.mean(res_ssim):+.3f})", edgecolor="#ffffff", linewidth=0.3)
        ax.hist(res_lpips, bins=20, alpha=0.65, color="#f59e0b", label=f"LPIPS Error (μ={np.mean(res_lpips):+.3f})", edgecolor="#ffffff", linewidth=0.3)
        ax.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5)

        ax.set_title(f"{m_title} — Residual Error Distribution\n(Predicted - Actual)", color="#f8fafc", fontsize=12, fontweight="bold")
        ax.set_xlabel("Prediction Error", color="#94a3b8")
        ax.set_ylabel("Frequency (Count)", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color="#26334d", linestyle=":", alpha=0.6)
        ax.legend(facecolor="#161e2e", edgecolor="#26334d", labelcolor="#f8fafc", loc="upper left")

    plt.tight_layout()
    res_chart_path = OUTPUT_DIR / "validation_102_residuals_distribution.png"
    plt.savefig(res_chart_path, dpi=200, bbox_inches="tight")
    if BRAIN_DIR.exists():
        shutil.copy(res_chart_path, BRAIN_DIR / "validation_102_residuals_distribution.png")
    plt.close()
    print(f"✅ Generated & Saved: {res_chart_path}")

    # 9. Save Summary JSON
    summary_json_path = OUTPUT_DIR / "validation_102_metrics_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(stats_summary, f, indent=2)
    print(f"✅ Saved validation summary JSON: {summary_json_path}")

    print("\n" + "=" * 80)
    print("🎉 102-IMAGE OUT-OF-SAMPLE VALIDATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    main()
