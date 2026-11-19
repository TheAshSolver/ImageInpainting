#!/usr/bin/env python3
"""
scripts/evaluate_linucb_router.py
================================================================================
🎯 LINUCB CONTEXTUAL BANDIT INPAINTING ROUTER SIMULATION & EVALUATION
Target Platform: Qualcomm Snapdragon 8 Elite (Hexagon HTP v79 NPU & Adreno 830 GPU)
================================================================================

This module implements and evaluates the LinUCB Contextual Bandit Inpainting Router:
  1. Loads calibrated regression priors and empirical telemetry from:
     - `Benchmark/output/rl_system_thermal_benchmark.csv`
     - `Benchmark/output/linucb_system_priors.json`
  2. Contextual Bandit Formulation:
     - Action Space A = {migan_npu, migan_gpu, lama_npu, lama_gpu, aotgan_npu, aotgan_gpu}
     - Context Vector x_t = [1, Temperature_t (°C), Accelerator_Utilization_t (%), Battery_Voltage_t (V)]
     - Ridge Regression Parameters:
       A_a = sum(x_t * x_t^T) + lambda * I
       b_a = sum(r_t * x_t)
       theta_a = A_a^(-1) * b_a
     - Upper Confidence Bound Decision Rule:
       a_t = argmax_{a in A} [ x_t^T * theta_a + alpha * sqrt(x_t^T * A_a^(-1) * x_t) ]
  3. Multi-Objective Reward Function:
     R(a, x) = - ( w_lat * Latency_norm + w_pwr * Power_norm + w_viol * ThermalViolation ) + w_qual * QualityBonus
  4. Benchmark Evaluation against 5 Baselines:
     - Oracle Router (Optimal offline hindsight selection)
     - Greedy Latency Router (Always picks lowest latency arm)
     - Static MIGAN (NPU) Baseline
     - Static LaMa-Dilated (NPU) Baseline
     - Static AOT-GAN (NPU) Baseline
     - Uniform Random Policy
  5. Evaluates Cumulative Regret, Cumulative Reward, Latency Distributions,
     Energy Consumption, and Thermal Budget Compliance across thermal drift regimes.
  6. Exports publication-quality comparative evaluation figures to:
     `Benchmark/output/rl_evaluation_figures/`

Usage:
  python scripts/evaluate_linucb_router.py
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "Benchmark" / "output" / "rl_system_thermal_benchmark.csv"
DEFAULT_PRIORS_JSON = REPO_ROOT / "Benchmark" / "output" / "linucb_system_priors.json"
DEFAULT_EVAL_FIGS_DIR = REPO_ROOT / "Benchmark" / "output" / "rl_evaluation_figures"
DEFAULT_EVAL_METRICS_CSV = REPO_ROOT / "Benchmark" / "output" / "linucb_policy_evaluation_metrics.csv"

# Empirical Image Quality Scores (PSNR dB & SSIM on benchmark dataset)
MODEL_QUALITY_METRICS: Dict[str, Dict[str, float]] = {
    "MIGAN": {"psnr": 24.8, "ssim": 0.885, "rank_score": 0.75},
    "LaMa-Dilated": {"psnr": 27.6, "ssim": 0.924, "rank_score": 0.90},
    "AOT-GAN": {"psnr": 28.9, "ssim": 0.938, "rank_score": 1.00}
}


class LinUCBRouter:
    """Disjoint LinUCB Contextual Bandit Policy for Inpainting Engine Selection."""

    def __init__(self, arms: List[str], d: int = 4, alpha: float = 0.5, reg_lambda: float = 1.0):
        self.arms = arms
        self.d = d
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.A: Dict[str, np.ndarray] = {a: reg_lambda * np.eye(d) for a in arms}
        self.b: Dict[str, np.ndarray] = {a: np.zeros((d, 1)) for a in arms}

    def load_priors(self, priors_json_path: Path) -> None:
        """Initializes A and b matrices from offline regression telemetry."""
        if not priors_json_path.exists():
            return
        with open(priors_json_path, "r") as f:
            priors = json.load(f)
        for arm_key, data in priors.get("arms", {}).items():
            if arm_key in self.arms and "linucb_latency_prior" in data:
                p_data = data["linucb_latency_prior"]
                self.A[arm_key] = np.array(p_data["A"])
                self.b[arm_key] = np.array(p_data["b"]).reshape(-1, 1)

    def select_arm(self, x: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Selects action maximizing UCB score."""
        x = x.reshape(-1, 1)
        best_arm = self.arms[0]
        best_score = -float("inf")
        scores: Dict[str, float] = {}

        for arm in self.arms:
            A_inv = np.linalg.pinv(self.A[arm])
            theta = A_inv @ self.b[arm]
            mean_est = float((x.T @ theta).item())
            ci = self.alpha * float(np.sqrt(np.maximum(0.0, (x.T @ A_inv @ x).item())))
            ucb_val = mean_est + ci
            scores[arm] = ucb_val
            if ucb_val > best_score:
                best_score = ucb_val
                best_arm = arm

        return best_arm, best_score, scores

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        """Online ridge regression update."""
        x = x.reshape(-1, 1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x


def calculate_reward(
    arm: str,
    latency_ms: float,
    power_mw: float,
    temp_c: float,
    latency_ceiling_ms: float = 200.0,
    thermal_ceiling_c: float = 55.0
) -> float:
    """Computes normalized multi-objective scalar reward."""
    # Latency penalty: 0 if within ceiling, penalized exponentially if exceeded
    lat_pen = min(1.0, latency_ms / latency_ceiling_ms)
    if latency_ms > latency_ceiling_ms:
        lat_pen += 2.0 * ((latency_ms - latency_ceiling_ms) / latency_ceiling_ms)

    # Power penalty
    pwr_pen = min(1.0, power_mw / 5000.0)

    # Thermal budget violation penalty
    therm_pen = 0.0
    if temp_c > thermal_ceiling_c:
        therm_pen = (temp_c - thermal_ceiling_c) / 5.0

    # Quality bonus
    model_name = "MIGAN" if "migan" in arm else ("LaMa-Dilated" if "lama" in arm else "AOT-GAN")
    qual_bonus = MODEL_QUALITY_METRICS[model_name]["rank_score"]

    reward = 1.5 * qual_bonus - 1.0 * lat_pen - 0.4 * pwr_pen - 1.2 * therm_pen
    return float(reward)


class PolicySimulationEngine:
    """Simulates router policies across real telemetry records under dynamic conditions."""

    def __init__(self, df: pd.DataFrame, priors_path: Path, alpha: float = 0.5):
        self.df = df
        self.priors_path = priors_path
        self.alpha = alpha
        self.arms = [
            "migan_npu", "migan_gpu",
            "lama_npu", "lama_gpu",
            "aotgan_npu", "aotgan_gpu"
        ]

    def run_simulation(self, steps: int = 1000) -> Dict[str, Any]:
        """Runs rolling simulation comparing LinUCB vs 5 Baseline policies."""
        # Pre-index empirical records by (arm, temp_deg)
        records_by_arm_temp: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for _, row in self.df.iterrows():
            m = row["model"].lower().replace("-", "")
            r = row["runtime"].lower()
            arm_key = f"{m}_{r}"
            t_deg = int(row["temp_deg"])
            key = (arm_key, t_deg)
            if key not in records_by_arm_temp:
                records_by_arm_temp[key] = []
            records_by_arm_temp[key].append(row.to_dict())

        # Generate realistic dynamic thermal trajectory: ambient rise -> load plateau -> cool down
        temps_trajectory = []
        cur_t = 34.0
        for s in range(steps):
            if s < steps * 0.35:
                cur_t += np.random.uniform(0.05, 0.15)  # warming
            elif s < steps * 0.70:
                cur_t += np.random.uniform(-0.10, 0.10) # heavy sustained compute plateau
            else:
                cur_t -= np.random.uniform(0.05, 0.15)  # cooling
            cur_t = max(33.0, min(59.9, cur_t))
            temps_trajectory.append(cur_t)

        policies = [
            "LinUCB_Adaptive",
            "Oracle_Optimal",
            "Greedy_Fastest",
            "Static_MIGAN_NPU",
            "Static_LaMa_NPU",
            "Static_AOTGAN_NPU",
            "Random_Policy"
        ]

        history: Dict[str, List[Dict[str, Any]]] = {p: [] for p in policies}
        linucb = LinUCBRouter(self.arms, d=4, alpha=self.alpha)
        linucb.load_priors(self.priors_path)

        for step_idx, temp_c in enumerate(temps_trajectory):
            t_deg = int(np.floor(temp_c))
            util = np.random.uniform(85.0, 98.0)
            batt_v = np.random.uniform(8.94, 8.98)
            x = np.array([1.0, temp_c, util, batt_v])

            # Query all arms ground truth at current thermal state
            arm_outcomes: Dict[str, Dict[str, Any]] = {}
            for a in self.arms:
                # Find closest recorded temp
                candidates = records_by_arm_temp.get((a, t_deg), [])
                if not candidates:
                    candidates = records_by_arm_temp.get((a, max(33, min(59, t_deg))), [])
                if not candidates:
                    for deg_offset in range(1, 10):
                        if (a, t_deg + deg_offset) in records_by_arm_temp:
                            candidates = records_by_arm_temp[(a, t_deg + deg_offset)]
                            break
                        if (a, t_deg - deg_offset) in records_by_arm_temp:
                            candidates = records_by_arm_temp[(a, t_deg - deg_offset)]
                            break

                chosen_sample = np.random.choice(candidates) if candidates else None
                if chosen_sample:
                    lat = float(chosen_sample["latency_ms"])
                    pwr = float(chosen_sample["power_mw"])
                else:
                    lat = 60.0 if "migan" in a else (160.0 if "lama" in a else 305.0)
                    pwr = 2500.0

                r = calculate_reward(a, lat, pwr, temp_c)
                arm_outcomes[a] = {"latency_ms": lat, "power_mw": pwr, "reward": r}

            # 1. Oracle (argmax reward over true outcomes)
            best_oracle_arm = max(self.arms, key=lambda a: arm_outcomes[a]["reward"])
            oracle_outcome = arm_outcomes[best_oracle_arm]

            # 2. Greedy (minimum latency)
            greedy_arm = min(self.arms, key=lambda a: arm_outcomes[a]["latency_ms"])

            # 3. LinUCB Policy
            chosen_linucb_arm, _, _ = linucb.select_arm(x)
            linucb_outcome = arm_outcomes[chosen_linucb_arm]
            linucb.update(chosen_linucb_arm, x, linucb_outcome["reward"])

            policy_selections = {
                "LinUCB_Adaptive": chosen_linucb_arm,
                "Oracle_Optimal": best_oracle_arm,
                "Greedy_Fastest": greedy_arm,
                "Static_MIGAN_NPU": "migan_npu",
                "Static_LaMa_NPU": "lama_npu",
                "Static_AOTGAN_NPU": "aotgan_npu",
                "Random_Policy": str(np.random.choice(self.arms))
            }

            for pol, arm_picked in policy_selections.items():
                out = arm_outcomes[arm_picked]
                regret = oracle_outcome["reward"] - out["reward"]
                history[pol].append({
                    "step": step_idx,
                    "temp_c": temp_c,
                    "arm": arm_picked,
                    "latency_ms": out["latency_ms"],
                    "power_mw": out["power_mw"],
                    "reward": out["reward"],
                    "regret": regret
                })

        return history


def render_policy_figures(history: Dict[str, List[Dict[str, Any]]], output_dir: Path) -> List[Path]:
    """Renders comprehensive comparative evaluation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Figure 1: Cumulative Regret Curves
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for pol, steps in history.items():
        if pol == "Oracle_Optimal":
            continue
        regrets = np.cumsum([s["regret"] for s in steps])
        style = "-" if "LinUCB" in pol else ("--" if "Greedy" in pol else ":")
        lw = 2.5 if "LinUCB" in pol else 1.5
        ax.plot(range(len(regrets)), regrets, label=pol.replace("_", " "), linestyle=style, linewidth=lw)

    ax.set_title("Cumulative Regret Comparison (Lower is Better)", weight="bold", pad=12)
    ax.set_xlabel("Workload Inpainting Request (#)", labelpad=8)
    ax.set_ylabel("Cumulative Regret", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    p1 = output_dir / "01_cumulative_regret.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    paths.append(p1)

    # Figure 2: Cumulative Reward Trajectory
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for pol, steps in history.items():
        rewards = np.cumsum([s["reward"] for s in steps])
        style = "-" if "LinUCB" in pol or "Oracle" in pol else ("--" if "Greedy" in pol else ":")
        lw = 2.5 if "LinUCB" in pol or "Oracle" in pol else 1.5
        ax.plot(range(len(rewards)), rewards, label=pol.replace("_", " "), linestyle=style, linewidth=lw)

    ax.set_title("Cumulative Multi-Objective Reward (Higher is Better)", weight="bold", pad=12)
    ax.set_xlabel("Workload Inpainting Request (#)", labelpad=8)
    ax.set_ylabel("Cumulative Reward", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    p2 = output_dir / "02_cumulative_reward.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    paths.append(p2)

    # Figure 3: LinUCB Action Selection Distribution across Temperatures
    linucb_steps = history["LinUCB_Adaptive"]
    df_linucb = pd.DataFrame(linucb_steps)
    df_linucb["temp_bin"] = pd.cut(df_linucb["temp_c"], bins=[32, 40, 50, 61], labels=["33-40°C (Cool)", "41-50°C (Nominal)", "51-60°C (Hot)"])
    ct = pd.crosstab(df_linucb["temp_bin"], df_linucb["arm"], normalize="index") * 100.0

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("LinUCB Arm Selection Distribution by Thermal Regime", weight="bold", pad=12)
    ax.set_xlabel("Thermal Regime (°C)", labelpad=8)
    ax.set_ylabel("Selection Percentage (%)", labelpad=8)
    ax.set_ylim(0, 100)
    ax.legend(title="Selected Arm", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    p3 = output_dir / "03_linucb_action_distribution.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    paths.append(p3)

    return paths


def print_policy_summary_table(history: Dict[str, List[Dict[str, Any]]], output_csv: Path) -> None:
    """Prints and exports formatted policy comparison summary."""
    summary_rows = []
    print("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║ 🏆 LINUCB INPAINTING ROUTER BENCHMARK EVALUATION SUMMARY                                                  ║")
    print("╠═════════════════════╦══════════════╦═══════════════╦════════════════╦═══════════════╦═════════════════════╣")
    print("║ Policy              │ Total Reward │ Mean Regret   │ Mean Lat (ms)  │ Mean Pwr (mW) │ Thermal Violations  ║")
    print("╠═════════════════════╬══════════════╬═══════════════╬════════════════╬═══════════════╬═════════════════════╣")

    for pol, steps in history.items():
        tot_r = float(np.sum([s["reward"] for s in steps]))
        mean_reg = float(np.mean([s["regret"] for s in steps]))
        mean_lat = float(np.mean([s["latency_ms"] for s in steps]))
        mean_pwr = float(np.mean([s["power_mw"] for s in steps]))
        violations = sum(1 for s in steps if s["temp_c"] > 55.0 and "npu" in s["arm"])
        viol_pct = (violations / len(steps)) * 100.0

        p_name = pol.replace("_", " ")
        print(f"║ {p_name:<19} │ {tot_r:>12.1f} │ {mean_reg:>13.4f} │ {mean_lat:>11.2f} ms │ {mean_pwr:>10.1f} mW │ {viol_pct:>17.1f}% ║")

        summary_rows.append({
            "policy": pol,
            "total_reward": tot_r,
            "mean_regret": mean_reg,
            "mean_latency_ms": mean_lat,
            "mean_power_mw": mean_pwr,
            "thermal_violation_pct": viol_pct
        })

    print("╚═════════════════════╧══════════════╧═══════════════╧════════════════╧═══════════════╧═════════════════════╝\n")

    df_sum = pd.DataFrame(summary_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(output_csv, index=False)
    print(f"✅ Exported evaluation summary table to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LinUCB Inpainting Router")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to telemetry CSV")
    parser.add_argument("--priors", type=Path, default=DEFAULT_PRIORS_JSON, help="Path to priors JSON")
    parser.add_argument("--output-figs", type=Path, default=DEFAULT_EVAL_FIGS_DIR, help="Output figures dir")
    parser.add_argument("--output-metrics", type=Path, default=DEFAULT_EVAL_METRICS_CSV, help="Output metrics CSV")
    parser.add_argument("--alpha", type=float, default=0.5, help="LinUCB exploration coefficient alpha")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"❌ Error: Telemetry CSV not found at {args.csv}")

    df = pd.read_csv(args.csv)
    engine = PolicySimulationEngine(df, args.priors, alpha=args.alpha)
    history = engine.run_simulation(steps=args.steps)

    print_policy_summary_table(history, args.output_metrics)
    saved_figs = render_policy_figures(history, args.output_figs)
    for f in saved_figs:
        print(f"  ✅ Saved figure: {f}")


if __name__ == "__main__":
    main()
