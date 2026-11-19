#!/usr/bin/env python3
"""
linear_regression.py
================================================================================
⚡ ON-DEVICE LATENCY LINEAR REGRESSION ESTIMATOR
================================================================================
Fits, evaluates, and exports Ordinary Least Squares (OLS) Linear Regression
models estimating image inpainting latency as a function of system resources
and hardware thermodynamic parameters on Qualcomm Snapdragon 8 Elite.

System Resource Inputs:
  1. soc_temp_start_c  : Initial SoC junction temperature in °C
  2. delta_temp_c      : Thermal rise (ΔT) during execution in °C
  3. active_power_w    : Active board power consumption in Watts
  4. is_npu            : Binary flag (1 = Hexagon HTP v79 NPU, 0 = Adreno 830 GPU)
  5. Architecture Flags: One-hot encoded architecture indicators (MIGAN, LaMa, AOT-GAN, SD)

Outputs:
  - Formatted regression equation with exact weights (w_i) and intercept (β_0)
  - Model evaluation metrics (R^2, MAE in ms, RMSE in ms)
  - Persistent parameter exports:
      * linear_regression_weights.json
      * Benchmark/output/linear_regression_weights.csv
================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = REPO_ROOT / "rl_system_thermal_benchmark.csv"
LEGACY_DATASET = REPO_ROOT / "Benchmark" / "output_old" / "previous_dataset_benchmark.csv"
OUTPUT_DIR = REPO_ROOT / "Benchmark" / "output"
LOCAL_WEIGHTS_JSON = REPO_ROOT / "linear_regression_weights.json"
OUTPUT_WEIGHTS_JSON = OUTPUT_DIR / "linear_regression_weights.json"
OUTPUT_WEIGHTS_CSV = OUTPUT_DIR / "linear_regression_weights.csv"


def prepare_npu_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Constructs dedicated NPU feature matrix for sub-millisecond precision."""
    data = df.copy()
    if "runtime" in data.columns:
        data = data[data["runtime"] == "NPU"].copy()
    elif "target_core" in data.columns:
        data = data[data["target_core"] == "NPU"].copy()

    # Model naming
    if "model" in data.columns and "model_name" not in data.columns:
        data["model_name"] = data["model"].str.lower().str.replace("-", "_")

    t_col = "exact_temp_c" if "exact_temp_c" in data.columns else "soc_temp_start_c"
    p_col = "power_mw" if "power_mw" in data.columns else "active_power_w"
    lat_col = "latency_ms" if "latency_ms" in data.columns else "npu_pure_latency_ms"

    X = pd.DataFrame({
        "soc_temp_start_c": data[t_col].astype(float),
        "power_mw": data[p_col].astype(float),
        "arch_lama": (data["model_name"].str.contains("lama")).astype(float),
        "arch_migan": (data["model_name"].str.contains("migan")).astype(float)
    })
    y = data[lat_col].astype(float)
    return X, y, list(X.columns)


def prepare_joint_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Constructs joint NPU + GPU feature matrix with interaction terms."""
    data = df.copy()
    if "model" in data.columns and "model_name" not in data.columns:
        data["model_name"] = data["model"].str.lower().str.replace("-", "_")
    if "runtime" in data.columns and "target_core" not in data.columns:
        data["target_core"] = data["runtime"]

    t_col = "exact_temp_c" if "exact_temp_c" in data.columns else "soc_temp_start_c"
    p_col = "power_mw" if "power_mw" in data.columns else "active_power_w"
    lat_col = "latency_ms" if "latency_ms" in data.columns else "npu_pure_latency_ms"

    is_npu = (data["target_core"].str.upper() == "NPU").astype(float)
    is_lama = (data["model_name"].str.contains("lama")).astype(float)
    is_migan = (data["model_name"].str.contains("migan")).astype(float)

    X = pd.DataFrame({
        "soc_temp_start_c": data[t_col].astype(float),
        "power_mw": data[p_col].astype(float),
        "is_npu": is_npu,
        "arch_lama": is_lama,
        "arch_migan": is_migan,
        "is_npu_x_lama": is_npu * is_lama,
        "is_npu_x_migan": is_npu * is_migan
    })
    y = data[lat_col].astype(float)
    return X, y, list(X.columns)


def fit_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Fits Linear Regression model and computes cross-validated metrics."""
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)

    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(reg, X, y, cv=kf, scoring="r2").mean()
    cv_mae = -cross_val_score(reg, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()

    weights_dict = {col: float(coef) for col, coef in zip(X.columns, reg.coef_)}

    return {
        "intercept": float(reg.intercept_),
        "weights": weights_dict,
        "r2": float(r2_score(y, y_pred)),
        "cv_r2_5fold": float(cv_r2),
        "mae_ms": float(mean_absolute_error(y, y_pred)),
        "cv_mae_ms": float(cv_mae),
        "rmse_ms": float(np.sqrt(mean_squared_error(y, y_pred))),
        "num_samples": len(y)
    }


def print_model_summary(title: str, results: Dict[str, Any], baseline_desc: str):
    print("\n" + "=" * 80)
    print(f"⚡ {title.upper()}")
    print("=" * 80)
    print(f" Samples Evaluated    : {results['num_samples']}")
    print(f" Goodness of Fit (R²) : {results['r2']:.6f} (5-fold CV R²: {results['cv_r2_5fold']:.6f})")
    print(f" Mean Absolute Error  : {results['mae_ms']:.4f} ms (5-fold CV MAE: {results['cv_mae_ms']:.4f} ms)")
    print(f" Root Mean Sq. Error  : {results['rmse_ms']:.4f} ms")
    print(f" Reference Baseline   : {baseline_desc}")
    print("-" * 80)
    print(f"{'Parameter / Feature':28s} | {'Weight (Coefficient)':22s} | Physical Interpretation")
    print("-" * 80)
    print(f"{'Intercept (β₀)':28s} | {results['intercept']:+18.4f} ms | Baseline latency at 0°C on GPU")

    interpretations = {
        "soc_temp_start_c": "Latency shift per 1°C increase in starting junction temp",
        "delta_temp_c": "Latency shift per 1°C thermal accumulation (thermal throttling)",
        "active_power_w": "Latency sensitivity per 1 Watt of board power draw",
        "is_npu": "NPU Hardware Acceleration delta (vs Adreno GPU)",
        "arch_migan": "Architecture efficiency delta vs AOT-GAN baseline",
        "arch_lama": "Architecture efficiency delta vs AOT-GAN baseline",
        "arch_sd": "Stable Diffusion 1.5 multi-step UNet overhead vs AOT-GAN"
    }

    for feature, weight in results["weights"].items():
        interp = interpretations.get(feature, "Model architectural factor")
        print(f"{feature:28s} | {weight:+18.4f} ms | {interp}")

    print("-" * 80)
    print(" Mathematical Prediction Equation:")
    eq_parts = [f"{results['intercept']:.2f}"]
    for f, w in results["weights"].items():
        sign = "+" if w >= 0 else "-"
        eq_parts.append(f"{sign} {abs(w):.2f} * {f}")
    print("  Latency_ms = " + " ".join(eq_parts))
    print("=" * 80)


def export_parameters(all_results: Dict[str, Any]):
    """Exports parameters to JSON and CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Export JSON
    with open(OUTPUT_WEIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    with open(LOCAL_WEIGHTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # 2. Export CSV
    rows = []
    for model_key, res in all_results.items():
        if not isinstance(res, dict) or "intercept" not in res:
            continue
        rows.append({
            "model_type": model_key,
            "feature": "intercept",
            "weight": res["intercept"],
            "unit": "ms",
            "r2": res["r2"],
            "mae_ms": res["mae_ms"]
        })
        for f, w in res["weights"].items():
            rows.append({
                "model_type": model_key,
                "feature": f,
                "weight": w,
                "unit": "ms / unit",
                "r2": res["r2"],
                "mae_ms": res["mae_ms"]
            })

    pd.DataFrame(rows).to_csv(OUTPUT_WEIGHTS_CSV, index=False)
    print(f"\n✅ Regression parameters successfully exported to:")
    print(f"  • JSON File (Root)   : {LOCAL_WEIGHTS_JSON}")
    print(f"  • JSON File (Output) : {OUTPUT_WEIGHTS_JSON}")
    print(f"  • CSV Summary File   : {OUTPUT_WEIGHTS_CSV}")


def predict_sample_latency(model_params: Dict[str, Any], arch: str, is_npu: bool,
                           temp_start: float, power_mw: float) -> float:
    """Convenience predictor function evaluating the calibrated linear equation."""
    weights = model_params["weights"]
    lat = model_params["intercept"]
    lat += weights.get("soc_temp_start_c", 0.0) * temp_start
    lat += weights.get("power_mw", weights.get("active_power_mw", 0.0)) * power_mw

    is_lama = 1.0 if "lama" in arch.lower() else 0.0
    is_migan = 1.0 if "migan" in arch.lower() else 0.0

    if "is_npu" in weights:
        # Joint model with accelerator interaction
        npu_flag = 1.0 if is_npu else 0.0
        lat += weights.get("is_npu", 0.0) * npu_flag
        lat += weights.get("arch_lama", 0.0) * is_lama
        lat += weights.get("arch_migan", 0.0) * is_migan
        lat += weights.get("is_npu_x_lama", 0.0) * (npu_flag * is_lama)
        lat += weights.get("is_npu_x_migan", 0.0) * (npu_flag * is_migan)
    else:
        # NPU dedicated model
        lat += weights.get("arch_lama", 0.0) * is_lama
        lat += weights.get("arch_migan", 0.0) * is_migan

    return max(0.0, lat)


def main():
    parser = argparse.ArgumentParser(description="Snapdragon 8 Elite Latency Linear Regression Estimator")
    parser.add_argument("-i", "--input", default=str(DEFAULT_DATASET), help="Input benchmark CSV")
    parser.add_argument("--predict-demo", action="store_true", help="Run interactive prediction demonstrations")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Dataset {input_path} not found.")
        sys.exit(1)

    print(f"Loading empirical benchmark dataset: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Fit Calibrated NPU Dedicated Latency Model (MIGAN, LaMa, AOT-GAN)
    X_npu, y_npu, feat_npu = prepare_npu_features(df)
    res_npu = fit_model(X_npu, y_npu)

    print_model_summary(
        title="Qualcomm Hexagon NPU Dedicated Latency Model",
        results=res_npu,
        baseline_desc="AOT-GAN on Hexagon HTP v79 NPU"
    )

    # 2. Fit Joint Acceleration Model with NPU Interactions (Full System)
    X_joint, y_joint, feat_joint = prepare_joint_features(df)
    res_joint = fit_model(X_joint, y_joint)

    print_model_summary(
        title="Joint Accelerator Model (Hexagon NPU & Adreno GPU with Cross-Core Interactions)",
        results=res_joint,
        baseline_desc="AOT-GAN on Adreno 830 GPU"
    )

    # 3. Compute Per-Model Dedicated NPU Equations
    per_model_fits = {}
    if "runtime" in df.columns:
        npu_sub = df[df["runtime"] == "NPU"].copy()
    else:
        npu_sub = df.copy()

    for m in ["MIGAN", "LaMa-Dilated", "AOT-GAN"]:
        sub = npu_sub[npu_sub["model"] == m] if "model" in npu_sub.columns else pd.DataFrame()
        if not sub.empty:
            t_col = "exact_temp_c" if "exact_temp_c" in sub.columns else "soc_temp_start_c"
            lat_col = "latency_ms" if "latency_ms" in sub.columns else "npu_pure_latency_ms"
            reg_m = LinearRegression().fit(sub[[t_col]], sub[lat_col])
            preds_m = reg_m.predict(sub[[t_col]])
            per_model_fits[m.lower().replace("-", "_")] = {
                "formula": f"Latency = {reg_m.intercept_:.4f} + ({reg_m.coef_[0]:+.5f}) * Temp_C",
                "intercept": float(reg_m.intercept_),
                "temp_slope": float(reg_m.coef_[0]),
                "mae_ms": float(mean_absolute_error(sub[lat_col], preds_m)),
                "rmse_ms": float(np.sqrt(mean_squared_error(sub[lat_col], preds_m))),
                "actual_mean_ms": float(sub[lat_col].mean()),
                "num_samples": len(sub)
            }

    # 4. Export all parameter files
    all_exports = {
        "metadata": {
            "dataset_source": str(input_path),
            "total_samples": len(df),
            "npu_samples": len(X_npu),
            "target_hardware": "Qualcomm Snapdragon 8 Elite (Hexagon HTP v79 NPU & Adreno 830 GPU)",
            "date_fitted": pd.Timestamp.now().isoformat(),
            "status": "calibrated_with_empirical_thermal_telemetry"
        },
        "npu_calibrated_latency": res_npu,
        "npu_individual_models": per_model_fits,
        "full_system_joint_latency": res_joint
    }
    export_parameters(all_exports)

    # 5. On-Device Latency Prediction Demonstration
    print("\n" + "=" * 80)
    print("🔍 ON-DEVICE LATENCY PREDICTION DEMONSTRATION")
    print("=" * 80)
    demo_scenarios = [
        {"desc": "MIGAN on Hexagon NPU @ Cool Ambient (35.0°C, 35 mW)", "arch": "migan", "is_npu": True, "t_start": 35.0, "pwr": 35.0},
        {"desc": "MIGAN on Hexagon NPU @ Peak Operating Temp (50.0°C, 65 mW)", "arch": "migan", "is_npu": True, "t_start": 50.0, "pwr": 65.0},
        {"desc": "LaMa Dilated on Hexagon NPU @ Peak Temp (50.0°C, 75 mW)", "arch": "lama", "is_npu": True, "t_start": 50.0, "pwr": 75.0},
        {"desc": "LaMa Dilated on Adreno GPU @ Peak Temp (50.0°C, 60 mW)", "arch": "lama", "is_npu": False, "t_start": 50.0, "pwr": 60.0},
        {"desc": "AOT-GAN on Hexagon NPU @ Peak Temp (50.0°C, 65 mW)", "arch": "aotgan", "is_npu": True, "t_start": 50.0, "pwr": 65.0},
        {"desc": "AOT-GAN on Adreno GPU @ Peak Temp (50.0°C, 65 mW)", "arch": "aotgan", "is_npu": False, "t_start": 50.0, "pwr": 65.0},
    ]

    for sc in demo_scenarios:
        est = predict_sample_latency(res_joint, sc["arch"], sc["is_npu"], sc["t_start"], sc["pwr"])
        fps = 1000.0 / est if est > 0 else 0.0
        core_str = "Hexagon NPU" if sc["is_npu"] else "Adreno GPU"
        print(f"  • {sc['desc']}")
        print(f"    → Estimated Pure Compute: {est:6.2f} ms ({fps:5.1f} FPS) on {core_str}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
