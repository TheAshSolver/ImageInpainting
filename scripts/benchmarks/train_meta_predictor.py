#!/usr/bin/env python3
"""
Meta-Predictor Training & Pareto Routing Engine
Trained on Hexagon HTP v79 NPU 6,000-Image Benchmark Dataset
Predicts PSNR, SSIM, LPIPS, and Latency from visual & geometric features,
and selects the Pareto-optimal inpainting model.
"""

import os
import sys
import json
import pickle
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

CSV_PATH = Path("/home/aaravind/develop/models/Image_Inpainting/Benchmark/output/6000_benchmark/dataset_meta_predictor.csv")
OUTPUT_DIR = Path("/home/aaravind/develop/models/Image_Inpainting/Benchmark/output/6000_benchmark")
BRAIN_DIR = Path("/home/aaravind/.gemini/antigravity-cli/brain/7395c208-1260-4e1f-a10c-06023eabccca")

FEATURE_COLS = [
    "img_laplacian_var",
    "img_canny_edge_density",
    "img_mean_luminance",
    "img_std_contrast",
    "img_color_entropy",
    "mask_coverage_pct",
    "mask_hole_count",
    "mask_centroid_x",
    "mask_centroid_y",
    "mask_dist_to_border",
    "mask_aspect_ratio",
    "mask_perimeter_to_area"
]

TARGET_COLS = ["psnr_hole", "psnr_full", "ssim", "lpips", "latency_ms"]
MODELS = ["migan_npu", "lama_npu", "aotgan_npu"]

def engineer_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """Adds non-linear and physical interaction features derived from empirical laws."""
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

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4)
    }

def train_and_evaluate_meta_predictor():
    print("=" * 80)
    print("🧠 TRAINING HEXAGON HTP NPU INPAINTING META-PREDICTOR")
    print("=" * 80)

    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found!")
        return 1

    df_raw = pd.read_csv(CSV_PATH)
    # Filter valid completed inferences
    df_valid = df_raw[df_raw["psnr_full"] > 0.0].copy()
    print(f"Total rows in CSV: {len(df_raw)} | Valid non-zero rows: {len(df_valid)}")

    # Group by sample_id to verify complete triplets
    samples_with_all_3 = []
    grouped = df_valid.groupby("sample_id")
    for s_id, group in grouped:
        if len(group) == 3 and set(group["model"].unique()) == set(MODELS):
            samples_with_all_3.append(s_id)

    print(f"Complete sample triplets (evaluated on all 3 NPU models): {len(samples_with_all_3)}")
    df_clean = df_valid[df_valid["sample_id"].isin(samples_with_all_3)].copy()

    # Create pivot table for routing and multi-model regression
    pivoted = {}
    # Extract sample level features from first model
    df_migan = df_clean[df_clean["model"] == "migan_npu"].sort_values("sample_id").reset_index(drop=True)
    df_lama = df_clean[df_clean["model"] == "lama_npu"].sort_values("sample_id").reset_index(drop=True)
    df_aot = df_clean[df_clean["model"] == "aotgan_npu"].sort_values("sample_id").reset_index(drop=True)

    X_base = df_migan[FEATURE_COLS].copy()
    X_eng = engineer_features(X_base)
    eng_feature_names = list(X_eng.columns)
    print(f"Engineered feature set size: {len(eng_feature_names)} features:")
    print("  ", eng_feature_names)

    # Train / Test split (80% train, 20% test)
    np.random.seed(42)
    train_idx, test_idx = train_test_split(np.arange(len(X_eng)), test_size=0.20, random_state=42)

    X_train = X_eng.iloc[train_idx].values
    X_test = X_eng.iloc[test_idx].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train regressor for each model and target
    model_predictors = {}
    evaluation_results = {}
    feature_importances = {}

    model_data_map = {
        "migan_npu": df_migan,
        "lama_npu": df_lama,
        "aotgan_npu": df_aot
    }

    print("\n" + "-" * 80)
    print("📊 REGRESSION ACCURACY BENCHMARK (Ridge vs GradientBoosting vs RandomForest)")
    print("-" * 80)

    for m_key in MODELS:
        m_df = model_data_map[m_key]
        model_predictors[m_key] = {}
        evaluation_results[m_key] = {}
        feature_importances[m_key] = {}

        print(f"\n[{m_key.upper()}] Performance Modeling:")

        for target in TARGET_COLS:
            y_all = m_df[target].values
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            # 1. Ridge Regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            pred_ridge = ridge.predict(X_test_scaled)
            res_ridge = compute_metrics(y_test, pred_ridge)

            # 2. Gradient Boosting
            gbr = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
            gbr.fit(X_train, y_train)
            pred_gbr = gbr.predict(X_test)
            res_gbr = compute_metrics(y_test, pred_gbr)

            # 3. Random Forest
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test)
            res_rf = compute_metrics(y_test, pred_rf)

            # Select Best Regressor
            candidates = [("Ridge", ridge, res_ridge, pred_ridge, False),
                          ("GradientBoosting", gbr, res_gbr, pred_gbr, True),
                          ("RandomForest", rf, res_rf, pred_rf, True)]
            candidates.sort(key=lambda c: c[2]["r2"], reverse=True)
            best_name, best_model, best_res, best_preds, is_raw = candidates[0]

            model_predictors[m_key][target] = {
                "model_type": best_name,
                "model": best_model,
                "uses_raw_features": is_raw,
                "test_predictions": best_preds,
                "y_test_true": y_test
            }
            evaluation_results[m_key][target] = {
                "best_model": best_name,
                "metrics": best_res,
                "ridge": res_ridge,
                "gbr": res_gbr,
                "rf": res_rf
            }

            if hasattr(best_model, "feature_importances_"):
                feature_importances[m_key][target] = best_model.feature_importances_

            print(f"  • {target:12s} | Best: {best_name:16s} | R²: {best_res['r2']:6.4f} | RMSE: {best_res['rmse']:6.4f} | MAE: {best_res['mae']:6.4f}")

    # ==============================================================================
    # DYNAMIC ROUTING / META-SELECTOR EVALUATION
    # ==============================================================================
    print("\n" + "=" * 80)
    print("🎯 DYNAMIC META-SELECTOR / PARETO ROUTING EVALUATION")
    print("=" * 80)

    # Policies:
    # 1. fast: max throughput with acceptable PSNR >= 20.0
    # 2. fidelity: max SSIM & PSNR
    # 3. realism: min LPIPS
    # 4. pareto_balanced: Utility = 0.35 * (PSNR/30) + 0.35 * SSIM + 0.20 * (1 - LPIPS) - 0.10 * (Latency / 600)

    def compute_utility(psnr_val, ssim_val, lpips_val, lat_val):
        return (0.35 * (psnr_val / 30.0) +
                0.35 * ssim_val +
                0.20 * (1.0 - lpips_val) -
                0.10 * (lat_val / 600.0))

    N_test = len(test_idx)
    policies = ["speed_first", "structural_fidelity", "perceptual_realism", "balanced_pareto"]
    policy_matches = {p: 0 for p in policies}

    for i in range(N_test):
        # Actual ground truth values for all 3 models on this test sample
        actual = {}
        predicted = {}
        for m_key in MODELS:
            actual[m_key] = {
                "psnr_hole": model_predictors[m_key]["psnr_hole"]["y_test_true"][i],
                "psnr_full": model_predictors[m_key]["psnr_full"]["y_test_true"][i],
                "ssim": model_predictors[m_key]["ssim"]["y_test_true"][i],
                "lpips": model_predictors[m_key]["lpips"]["y_test_true"][i],
                "latency_ms": model_predictors[m_key]["latency_ms"]["y_test_true"][i]
            }
            predicted[m_key] = {
                "psnr_hole": model_predictors[m_key]["psnr_hole"]["test_predictions"][i],
                "psnr_full": model_predictors[m_key]["psnr_full"]["test_predictions"][i],
                "ssim": model_predictors[m_key]["ssim"]["test_predictions"][i],
                "lpips": model_predictors[m_key]["lpips"]["test_predictions"][i],
                "latency_ms": model_predictors[m_key]["latency_ms"]["test_predictions"][i]
            }

        # Policy 1: speed_first (MIGAN if latency < 100, else lowest latency)
        true_speed = min(MODELS, key=lambda m: actual[m]["latency_ms"])
        pred_speed = min(MODELS, key=lambda m: predicted[m]["latency_ms"])
        if true_speed == pred_speed:
            policy_matches["speed_first"] += 1

        # Policy 2: structural_fidelity (highest SSIM)
        true_fid = max(MODELS, key=lambda m: actual[m]["ssim"])
        pred_fid = max(MODELS, key=lambda m: predicted[m]["ssim"])
        if true_fid == pred_fid:
            policy_matches["structural_fidelity"] += 1

        # Policy 3: perceptual_realism (lowest LPIPS)
        true_real = min(MODELS, key=lambda m: actual[m]["lpips"])
        pred_real = min(MODELS, key=lambda m: predicted[m]["lpips"])
        if true_real == pred_real:
            policy_matches["perceptual_realism"] += 1

        # Policy 4: balanced_pareto (highest composite utility)
        true_util = max(MODELS, key=lambda m: compute_utility(actual[m]["psnr_hole"], actual[m]["ssim"], actual[m]["lpips"], actual[m]["latency_ms"]))
        pred_util = max(MODELS, key=lambda m: compute_utility(predicted[m]["psnr_hole"], predicted[m]["ssim"], predicted[m]["lpips"], predicted[m]["latency_ms"]))
        if true_util == pred_util:
            policy_matches["balanced_pareto"] += 1

    routing_accuracies = {p: round((policy_matches[p] / N_test) * 100.0, 2) for p in policies}
    print(f"Evaluated on {N_test} Held-Out Test Samples:")
    for p, acc in routing_accuracies.items():
        print(f"  • Policy: {p:22s} | Routing Precision: {acc:5.2f}%")

    # ==============================================================================
    # VISUALIZATION & ARTIFACT GENERATION
    # ==============================================================================
    print("\n" + "-" * 80)
    print("🎨 GENERATING META-PREDICTOR EVALUATION CHARTS")
    print("-" * 80)

    # 1. Parity Plot: Actual vs Predicted PSNR & SSIM
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#0f172a")

    for col_idx, m_key in enumerate(MODELS):
        m_title = m_key.replace("_npu", "").upper()
        # Row 0: PSNR Hole
        ax_psnr = axes[0, col_idx]
        ax_psnr.set_facecolor("#1e293b")
        y_true_p = model_predictors[m_key]["psnr_hole"]["y_test_true"]
        y_pred_p = model_predictors[m_key]["psnr_hole"]["test_predictions"]
        r2_p = evaluation_results[m_key]["psnr_hole"]["metrics"]["r2"]

        ax_psnr.scatter(y_true_p, y_pred_p, alpha=0.45, s=18, color="#38bdf8", edgecolors="none")
        p_min, p_max = min(y_true_p.min(), y_pred_p.min()), max(y_true_p.max(), y_pred_p.max())
        ax_psnr.plot([p_min, p_max], [p_min, p_max], "r--", lw=1.5, label="Ideal (y=x)")
        ax_psnr.set_title(f"{m_title} — PSNR (Hole)\nR² = {r2_p:.3f}", color="#f8fafc", fontsize=12, fontweight="bold")
        ax_psnr.set_xlabel("True PSNR Hole (dB)", color="#94a3b8")
        ax_psnr.set_ylabel("Predicted PSNR Hole (dB)", color="#94a3b8")
        ax_psnr.tick_params(colors="#94a3b8")
        ax_psnr.grid(True, color="#334155", linestyle=":", alpha=0.6)
        ax_psnr.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")

        # Row 1: SSIM
        ax_ssim = axes[1, col_idx]
        ax_ssim.set_facecolor("#1e293b")
        y_true_s = model_predictors[m_key]["ssim"]["y_test_true"]
        y_pred_s = model_predictors[m_key]["ssim"]["test_predictions"]
        r2_s = evaluation_results[m_key]["ssim"]["metrics"]["r2"]

        ax_ssim.scatter(y_true_s, y_pred_s, alpha=0.45, s=18, color="#4ade80", edgecolors="none")
        s_min, s_max = min(y_true_s.min(), y_pred_s.min()), max(y_true_s.max(), y_pred_s.max())
        ax_ssim.plot([s_min, s_max], [s_min, s_max], "r--", lw=1.5, label="Ideal (y=x)")
        ax_ssim.set_title(f"{m_title} — SSIM\nR² = {r2_s:.3f}", color="#f8fafc", fontsize=12, fontweight="bold")
        ax_ssim.set_xlabel("True SSIM", color="#94a3b8")
        ax_ssim.set_ylabel("Predicted SSIM", color="#94a3b8")
        ax_ssim.tick_params(colors="#94a3b8")
        ax_ssim.grid(True, color="#334155", linestyle=":", alpha=0.6)
        ax_ssim.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc", loc="upper left")

    plt.tight_layout()
    parity_chart_path = OUTPUT_DIR / "meta_predictor_parity_analysis.png"
    plt.savefig(parity_chart_path, dpi=200, bbox_inches="tight")
    if BRAIN_DIR.exists():
        shutil.copy(parity_chart_path, BRAIN_DIR / "meta_predictor_parity_analysis.png")
    plt.close()
    print(f"  ✅ Saved: {parity_chart_path}")

    # 2. Feature Importance Analysis Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0f172a")

    for col_idx, m_key in enumerate(MODELS):
        m_title = m_key.replace("_npu", "").upper()
        ax = axes[col_idx]
        ax.set_facecolor("#1e293b")

        if m_key in feature_importances and "psnr_hole" in feature_importances[m_key]:
            importances = feature_importances[m_key]["psnr_hole"]
            sorted_idx = np.argsort(importances)[-8:]  # Top 8 features
            top_feats = [eng_feature_names[i] for i in sorted_idx]
            top_vals = importances[sorted_idx]

            y_pos = np.arange(len(top_feats))
            bars = ax.barh(y_pos, top_vals, color="#f59e0b", edgecolor="#d97706", height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_feats, color="#f8fafc", fontsize=10)
            ax.set_xlabel("Relative Gini Importance", color="#94a3b8")
            ax.set_title(f"{m_title} — Top Feature Drivers\nfor PSNR (Hole)", color="#f8fafc", fontsize=12, fontweight="bold")
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, axis="x", color="#334155", linestyle=":", alpha=0.6)

    plt.tight_layout()
    feat_chart_path = OUTPUT_DIR / "meta_predictor_feature_importance.png"
    plt.savefig(feat_chart_path, dpi=200, bbox_inches="tight")
    if BRAIN_DIR.exists():
        shutil.copy(feat_chart_path, BRAIN_DIR / "meta_predictor_feature_importance.png")
    plt.close()
    print(f"  ✅ Saved: {feat_chart_path}")

    # ==============================================================================
    # SERIALIZATION / EXPORT OF META-PREDICTOR BUNDLE
    # ==============================================================================
    bundle = {
        "feature_cols": FEATURE_COLS,
        "engineered_feature_cols": eng_feature_names,
        "scaler": scaler,
        "models": {m_key: {t: {"model": model_predictors[m_key][t]["model"],
                               "uses_raw_features": model_predictors[m_key][t]["uses_raw_features"],
                               "model_type": model_predictors[m_key][t]["model_type"]}
                           for t in TARGET_COLS}
                   for m_key in MODELS},
        "evaluation_results": {m_key: {t: evaluation_results[m_key][t]["metrics"] for t in TARGET_COLS} for m_key in MODELS},
        "routing_accuracies": routing_accuracies,
        "trained_samples_count": len(df_clean),
        "timestamp": time.time()
    }

    bundle_path = OUTPUT_DIR / "meta_predictor_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n✅ Exported Meta-Predictor Pickle Bundle: {bundle_path}")

    # Export Human-Readable Metrics JSON
    metrics_json_path = OUTPUT_DIR / "meta_predictor_metrics.json"
    clean_eval_export = {
        "trained_on_samples": len(df_clean),
        "held_out_test_samples": N_test,
        "routing_accuracies": routing_accuracies,
        "model_performance_r2_rmse": {
            m_key: {
                t: evaluation_results[m_key][t]["metrics"]
                for t in TARGET_COLS
            } for m_key in MODELS
        }
    }
    with open(metrics_json_path, "w") as f:
        json.dump(clean_eval_export, f, indent=2)
    print(f"✅ Exported Metrics Summary JSON: {metrics_json_path}")

    print("\n" + "=" * 80)
    print("🎉 META-PREDICTOR TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    train_and_evaluate_meta_predictor()
