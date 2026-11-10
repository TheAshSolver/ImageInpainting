#!/usr/bin/env python3
"""
scripts/train_linucb_regression.py
================================================================================
🎯 LINUCB SYSTEM PARAMETER FEATURE EXTRACTION & REGRESSION ANALYSIS
Target Platform: Qualcomm Snapdragon 8 Elite (Hexagon HTP v79 NPU & Adreno 830 GPU)
================================================================================

This module:
  1. Ingests empirical telemetry from Strategy B thermal sweep:
     `Benchmark/output/rl_system_thermal_benchmark.csv`.
  2. Extracts system feature vector for LinUCB contextual bandit:
     x = [1, Temperature (°C), Accelerator Utilization (%), Battery Voltage (V)]
  3. Fits Ordinary Least Squares (OLS) multi-variable linear regressions for each arm:
     - Latency(T, U, V) = β0 + β1*T + β2*U + β3*V
     - Power(T, U, V)   = γ0 + γ1*T + γ2*U + γ3*V
  4. Computes rigorous statistical metrics: R², Adjusted R², RMSE, t-stats, and p-values.
  5. Computes and exports LinUCB prior matrices:
     A_a = X_a^T X_a + λ I  (Regularization λ = 1.0)
     b_a = X_a^T y_a
     θ_a = A_a^{-1} b_a
     Saved to `Benchmark/output/linucb_system_priors.json` for runtime C++/Python inference.

Usage:
  python scripts/train_linucb_regression.py --input-csv Benchmark/output/rl_system_thermal_benchmark.csv
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_CSV = REPO_ROOT / "Benchmark" / "output" / "rl_system_thermal_benchmark.csv"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "Benchmark" / "output" / "linucb_system_priors.json"


def fit_ols(X: np.ndarray, y: np.ndarray, feature_names: list) -> Dict[str, Any]:
    """Fits analytical Ordinary Least Squares regression and computes statistics."""
    N, d = X.shape
    df_resid = N - d
    if df_resid <= 0:
        raise ValueError(f"Insufficient degrees of freedom: N={N}, d={d}")

    # Robust minimum-norm least squares with condition threshold
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-4)

    y_pred = X @ beta
    resid = y - y_pred
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))

    r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
    adj_r2 = 1.0 - ((rss / df_resid) / (tss / (N - 1))) if tss > 0 and (N - 1) > 0 else 0.0
    rmse = float(np.sqrt(rss / N))

    s2 = rss / max(1, df_resid)
    XtX = X.T @ X
    cov_beta = s2 * np.linalg.pinv(XtX, rcond=1e-4)
    se_beta = np.sqrt(np.maximum(0.0, np.diag(cov_beta)))
    t_stats = np.where(se_beta > 1e-9, beta / se_beta, 0.0)
    p_values = 2.0 * stats.t.sf(np.abs(t_stats), df=max(1, df_resid))

    coef_details = {}
    for idx, fname in enumerate(feature_names):
        coef_details[fname] = {
            "beta": float(beta[idx]),
            "std_error": float(se_beta[idx]),
            "t_stat": float(t_stats[idx]),
            "p_value": float(p_values[idx])
        }

    return {
        "beta": beta.tolist(),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(rmse),
        "coefficients": coef_details,
        "n_samples": int(N)
    }


def compute_linucb_prior(X: np.ndarray, y: np.ndarray, reg_lambda: float = 1.0) -> Dict[str, Any]:
    """Computes LinUCB precision matrix A_a, vector b_a, and ridge estimate theta_a."""
    N, d = X.shape
    A = X.T @ X + reg_lambda * np.eye(d)
    b = X.T @ y
    A_inv = np.linalg.inv(A)
    theta = A_inv @ b

    return {
        "regularization_lambda": float(reg_lambda),
        "A": A.tolist(),
        "A_inv": A_inv.tolist(),
        "b": b.tolist(),
        "theta": theta.tolist()
    }


def analyze_dataset(csv_path: Path, output_json: Path, reg_lambda: float = 1.0) -> None:
    if not csv_path.exists():
        sys.exit(f"❌ Error: Telemetry CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\n📂 Loaded {len(df)} empirical records from {csv_path}")

    feature_cols = ["intercept", "temp_c", "utilization_pct", "battery_v"]
    arms = [
        ("MIGAN", "NPU", "migan_npu"),
        ("MIGAN", "GPU", "migan_gpu"),
        ("LaMa-Dilated", "NPU", "lama_npu"),
        ("LaMa-Dilated", "GPU", "lama_gpu"),
        ("AOT-GAN", "NPU", "aotgan_npu"),
        ("AOT-GAN", "GPU", "aotgan_gpu"),
    ]

    priors_payload: Dict[str, Any] = {
        "metadata": {
            "device": "Qualcomm Snapdragon 8 Elite (SM8750P)",
            "npu": "Hexagon HTP v79",
            "gpu": "Adreno 830",
            "feature_names": feature_cols,
            "total_records": len(df)
        },
        "arms": {}
    }

    print("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║ 🔬 LINUCB SYSTEM TELEMETRY MULTI-VARIABLE LINEAR REGRESSION ANALYSIS                                             ║")
    print("╠══════════════════╦════════╦═════════╦══════════╦═══════════╦══════════╦══════════════════════════════════════════╣")
    print("║ Arm (Model/Eng)  │ Target │ N-Pts   │ R² Score │ Adj-R²    │ RMSE     │ OLS Equation                             ║")
    print("╠══════════════════╬════════╬═════════╬══════════╬═══════════╬══════════╬══════════════════════════════════════════╣")

    for model_name, runtime_str, arm_key in arms:
        sub_df = df[(df["model"] == model_name) & (df["runtime"] == runtime_str)]
        if len(sub_df) < 5:
            print(f"║ {arm_key:<16} │ [INSUFFICIENT DATA: {len(sub_df)} points]                                                                    ║")
            continue

        # Extract features
        N = len(sub_df)
        X0 = np.ones((N, 1))
        X1 = sub_df["exact_temp_c"].to_numpy().reshape(-1, 1)
        util_col = "npu_usage_pct" if runtime_str == "NPU" else "gpu_usage_pct"
        X2 = sub_df[util_col].to_numpy().reshape(-1, 1)
        X3 = sub_df["battery_v"].to_numpy().reshape(-1, 1)
        X = np.hstack([X0, X1, X2, X3])

        # Targets: Latency (ms) and Power (mW)
        y_lat = sub_df["latency_ms"].to_numpy()
        y_pwr = sub_df["power_mw"].to_numpy()

        res_lat = fit_ols(X, y_lat, feature_cols)
        res_pwr = fit_ols(X, y_pwr, feature_cols)

        # LinUCB priors
        prior_lat = compute_linucb_prior(X, y_lat, reg_lambda)
        prior_pwr = compute_linucb_prior(X, y_pwr, reg_lambda)

        b_lat = res_lat["beta"]
        eq_lat = f"{b_lat[0]:.1f} + {b_lat[1]:.2f}*T + {b_lat[2]:.2f}*U + {b_lat[3]:.2f}*V"
        print(f"║ {arm_key:<16} │ Lat(ms)│ {N:<7} │ {res_lat['r2']:>8.4f} │ {res_lat['adj_r2']:>9.4f} │ {res_lat['rmse']:>8.2f} │ {eq_lat:<40} ║")

        b_pwr = res_pwr["beta"]
        eq_pwr = f"{b_pwr[0]:.1f} + {b_pwr[1]:.2f}*T + {b_pwr[2]:.2f}*U + {b_pwr[3]:.2f}*V"
        print(f"║ {' ':<16} │ Pwr(mW)│ {N:<7} │ {res_pwr['r2']:>8.4f} │ {res_pwr['adj_r2']:>9.4f} │ {res_pwr['rmse']:>8.2f} │ {eq_pwr:<40} ║")
        print("╟──────────────────┼────────┼─────────┼──────────┼───────────┼──────────┼──────────────────────────────────────────╢")

        priors_payload["arms"][arm_key] = {
            "model": model_name,
            "runtime": runtime_str,
            "latency_regression": res_lat,
            "power_regression": res_pwr,
            "linucb_latency_prior": prior_lat,
            "linucb_power_prior": prior_pwr
        }

    print("╚══════════════════╧════════╧═════════╧══════════╧═══════════╧══════════╧══════════════════════════════════════════╝\n")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(priors_payload, f, indent=2)

    print(f"✅ Exported LinUCB Prior Parameter Matrices to {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit LinUCB System Regression Models")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV,
                        help=f"Path to empirical telemetry CSV (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON,
                        help=f"Path to export LinUCB priors JSON (default: {DEFAULT_OUTPUT_JSON})")
    parser.add_argument("--regularization-lambda", type=float, default=1.0,
                        help="Ridge / LinUCB regularization parameter λ (default: 1.0)")
    args = parser.parse_args()

    analyze_dataset(args.input_csv, args.output_json, args.regularization_lambda)


if __name__ == "__main__":
    main()
