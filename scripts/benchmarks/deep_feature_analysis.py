#!/usr/bin/env python3
"""
Deep 50-Feature Correlation and Statistical Importance Analysis
================================================================
Merges 50-dimensional image/mask structural indicators with empirical NPU
benchmark quality and performance metrics (PSNR, SSIM, LPIPS, Latency).

Computes:
1. Full 50x3 Pearson (r) and Spearman (rho) correlation matrices.
2. Non-linear Mutual Information (MI) regression scores.
3. Random Forest Feature Importance and Cumulative Variance Explained (R^2).
4. Top-10 predictive feature subspace isolating primary drivers of inpainting degradation.

Outputs:
- artifacts/correlation_matrices_50_features.json
- artifacts/top10_feature_subspace.json
- artifacts/feature_correlation_500.csv
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
BENCHMARK_CSV = REPO_ROOT / "Benchmark" / "output" / "6000_benchmark" / "dataset_meta_predictor.csv"
FEATURES_50_CSV = ARTIFACTS_DIR / "dataset_50_features_6000.csv"

def run_deep_analysis():
    print("=" * 80)
    print("🔬 DEEP 50-FEATURE STATISTICAL CORRELATION & IMPORTANCE ANALYSIS")
    print("=" * 80)

    if not FEATURES_50_CSV.exists():
        print(f"Error: {FEATURES_50_CSV} not found! Run extract_50_features.py first.")
        return 1

    if not BENCHMARK_CSV.exists():
        print(f"Error: {BENCHMARK_CSV} not found! Run run_6000_npu_benchmark.py first.")
        return 1

    print(f"Loading 50 features from: {FEATURES_50_CSV.name}")
    df_feat = pd.read_csv(FEATURES_50_CSV)
    df_feat["sample_id"] = df_feat["sample_id"].astype(str).str.zfill(4)

    print(f"Loading empirical metrics from: {BENCHMARK_CSV.name}")
    df_bench = pd.read_csv(BENCHMARK_CSV)
    df_bench["sample_id"] = df_bench["sample_id"].astype(str).str.zfill(4)

    # Exclude non-feature columns
    non_feat_cols = {"sample_id", "category", "status"}
    feat_cols = [c for c in df_feat.columns if c not in non_feat_cols]
    print(f"Loaded {len(df_feat)} feature records ({len(feat_cols)} mathematical features).")
    print(f"Loaded {len(df_bench)} benchmark evaluation records across models: {df_bench['model'].unique().tolist()}")

    # Merge features with benchmark metrics
    merged = pd.merge(df_bench, df_feat, on="sample_id", how="inner", suffixes=("", "_feat"))
    print(f"Successfully matched {len(merged)} sample-model evaluations.")

    # Save consolidated 500-sample feature correlation dataset
    out_merged_path = ARTIFACTS_DIR / "feature_correlation_500.csv"
    merged.to_csv(out_merged_path, index=False)
    print(f"Exported consolidated dataset -> {out_merged_path.name}")

    models = sorted(merged["model"].unique())
    target_metrics = ["lpips", "psnr_hole", "ssim", "latency_ms"]

    full_correlation_report = {
        "num_samples": len(merged["sample_id"].unique()),
        "models": models,
        "feature_count": len(feat_cols),
        "features": feat_cols,
        "correlations": {},
        "mutual_info": {},
        "random_forest_importance": {}
    }

    feature_global_scores = {f: {"lpips_importance": [], "psnr_importance": [], "r2_contrib": []} for f in feat_cols}

    for m in models:
        m_df = merged[merged["model"] == m].copy()
        if len(m_df) < 10:
            print(f"Skipping model {m}: insufficient samples ({len(m_df)})")
            continue

        print(f"\n--- Analyzing Model: {m} (N={len(m_df)}) ---")
        full_correlation_report["correlations"][m] = {}
        full_correlation_report["mutual_info"][m] = {}
        full_correlation_report["random_forest_importance"][m] = {}

        X = m_df[feat_cols].values
        # Impute or handle NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        for target in target_metrics:
            if target not in m_df.columns:
                continue
            y = m_df[target].values

            # 1. Pearson & Spearman
            pearson_dict = {}
            spearman_dict = {}
            for idx, f_name in enumerate(feat_cols):
                f_vals = X[:, idx]
                if np.std(f_vals) > 1e-8 and np.std(y) > 1e-8:
                    r_val, p_val = stats.pearsonr(f_vals, y)
                    rho_val, rho_p = stats.spearmanr(f_vals, y)
                    pearson_dict[f_name] = {"r": round(float(r_val), 4), "p_val": float(p_val)}
                    spearman_dict[f_name] = {"rho": round(float(rho_val), 4), "p_val": float(rho_p)}
                else:
                    pearson_dict[f_name] = {"r": 0.0, "p_val": 1.0}
                    spearman_dict[f_name] = {"rho": 0.0, "p_val": 1.0}

            full_correlation_report["correlations"][m][f"{target}_pearson"] = pearson_dict
            full_correlation_report["correlations"][m][f"{target}_spearman"] = spearman_dict

            # 2. Mutual Information
            try:
                mi_scores = mutual_info_regression(X, y, random_state=42)
                mi_dict = {f_name: round(float(score), 4) for f_name, score in zip(feat_cols, mi_scores)}
                full_correlation_report["mutual_info"][m][target] = mi_dict
            except Exception as e:
                print(f"MI error on {target}: {e}")

        # 3. Random Forest Regressor for LPIPS and PSNR_hole
        for target in ["lpips", "psnr_hole"]:
            if target not in m_df.columns:
                continue
            y = m_df[target].values
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            r2 = r2_score(y, rf.predict(X))

            importances = rf.feature_importances_
            ranking = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)
            rf_dict = {f: round(float(imp), 4) for f, imp in ranking}

            full_correlation_report["random_forest_importance"][m][target] = {
                "r2_train": round(float(r2), 4),
                "feature_importance": rf_dict
            }

            for f, imp in ranking:
                if target == "lpips":
                    feature_global_scores[f]["lpips_importance"].append(imp)
                else:
                    feature_global_scores[f]["psnr_importance"].append(imp)

    # 4. Isolate Top-10 Predictive Feature Subspace
    print("\n" + "=" * 80)
    print("🏆 ISOLATING TOP-10 PREDICTIVE FEATURE SUBSPACE")
    print("=" * 80)

    aggregate_importance = []
    for f in feat_cols:
        lp_mean = np.mean(feature_global_scores[f]["lpips_importance"]) if feature_global_scores[f]["lpips_importance"] else 0.0
        psnr_mean = np.mean(feature_global_scores[f]["psnr_importance"]) if feature_global_scores[f]["psnr_importance"] else 0.0
        # Combined composite score
        comp_score = 0.6 * lp_mean + 0.4 * psnr_mean
        aggregate_importance.append((f, comp_score, lp_mean, psnr_mean))

    aggregate_importance.sort(key=lambda x: x[1], reverse=True)
    top_10 = aggregate_importance[:10]

    # Calculate cumulative variance explained (R^2 proxy)
    total_comp_score = sum(x[1] for x in aggregate_importance)
    cum_pct = 0.0

    top_10_list = []
    for rank, (f, c_score, lp, ps) in enumerate(top_10, 1):
        pct = (c_score / total_comp_score * 100.0) if total_comp_score > 0 else 0.0
        cum_pct += pct
        top_10_list.append({
            "rank": rank,
            "feature": f,
            "composite_importance": round(float(c_score), 4),
            "lpips_importance": round(float(lp), 4),
            "psnr_importance": round(float(ps), 4),
            "variance_share_pct": round(float(pct), 2),
            "cumulative_variance_pct": round(float(cum_pct), 2)
        })
        print(f"  {rank:2d}. {f:<32s} | Score: {c_score:.4f} | Share: {pct:5.2f}% | Cumulative: {cum_pct:5.2f}%")

    top10_subspace = {
        "analysis_date": pd.Timestamp.now().isoformat(),
        "total_features_evaluated": len(feat_cols),
        "total_variance_explained_by_top10_pct": round(float(cum_pct), 2),
        "top_10_features": top_10_list,
        "selected_feature_names": [x["feature"] for x in top_10_list]
    }

    # Save to artifacts
    corr_json_path = ARTIFACTS_DIR / "correlation_matrices_50_features.json"
    with open(corr_json_path, "w") as f:
        json.dump(full_correlation_report, f, indent=2)
    print(f"\n✅ Saved correlation matrices -> {corr_json_path.name}")

    subspace_json_path = ARTIFACTS_DIR / "top10_feature_subspace.json"
    with open(subspace_json_path, "w") as f:
        json.dump(top10_subspace, f, indent=2)
    print(f"✅ Saved Top-10 feature subspace -> {subspace_json_path.name}")

    return 0

if __name__ == "__main__":
    sys.exit(run_deep_analysis())
