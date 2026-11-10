# Project Master Roadmap & Academic Rigor Action Plan (TODO)

> **Document Purpose**: Complete handover context, identified empirical discrepancies, root-cause code diagnostics, and step-by-step resolution roadmap for the Qualcomm Snapdragon 8 Elite (`SM8750P`) Neural Inpainting & Model Routing project.

---

## 1. Project Context & Current Architecture

### 1.1 Executive System Overview
* **Target Hardware**: Qualcomm Snapdragon 8 Elite Mobile Platform (`SM8750P` / `sun`, Serial `8f27557f`, HTP v79 NPU & Adreno 830 GPU).
* **Models Supported**:
  1. **MIGAN** (`512x512` static DLC, HTP v79 FastRPC resident daemon).
  2. **LaMa-Dilated** (`512x512` static DLC, large receptive field).
  3. **AOT-GAN** (`512x512` static DLC, gated attention for dense textures).
  4. **Classical Telea & Navier-Stokes** (ARM64 CPU fallback).
  5. **Stable Diffusion 1.5** (UNet + VAE on Hexagon NPU).
* **Target Application**: Android application [`qidk-inpaint-app-v5`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5) with custom touch selection, GrabCut refinement, resident daemon communication over Unix domain sockets/FastRPC, and a dynamic model router.

### 1.2 The 6,000-Image Empirical Benchmark Pipeline
* **Images**: 6,000 canonical 512×512 natural images in [`Benchmark/6000_images/compiled/images/`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/6000_images/compiled/images/).
* **Masks**: 6,000 jagged, multi-modal synthetic masks in [`Benchmark/6000_images/compiled/masks/`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/6000_images/compiled/masks/) partitioned into 1–5 hole counts and 5%–30% coverage bins.
* **Feature Matrices**:
  * [`artifacts/dataset_50_features_6000.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/artifacts/dataset_50_features_6000.csv): 50 computer vision features (color, Sobel edges, Gabor texture, morphology).
  * [`artifacts/dataset_100_features_6000.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/artifacts/dataset_100_features_6000.csv): 100 features including wavelets and GLCM texture features.
  * [`artifacts/dataset_complete_metrics_6000.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/artifacts/dataset_complete_metrics_6000.csv): Full quality evaluation across engines (PSNR, Hole PSNR, SSIM, LPIPS-VGG, CMMD).

---

## 2. Comprehensive Discrepancy & Peer-Review Audit

A rigorous peer review of the Reinforcement Learning / Contextual Bandit figures and slides revealed 7 major discrepancies that contradict the project documentation and would fail an academic committee evaluation:

| # | Discrepancy / Claim | Empirical Reality from Plots & Data | Severity |
|---|---|---|---|
| **1** | **"Sub-linear $O(\sqrt{T})$ Regret"** | In [`01_cumulative_regret.png`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/rl_evaluation_figures/01_cumulative_regret.png) and [`linucb_policy_evaluation_metrics.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/linucb_policy_evaluation_metrics.csv), LinUCB ends at **cumulative regret 2,945**, identical to Uniform Random (2,943). The curve steepens after step 600 instead of flattening. | **Critical** |
| **2** | **"Policy Converges in < 200 Rounds"** | In [`02_cumulative_reward.png`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/rl_evaluation_figures/02_cumulative_reward.png), LinUCB finishes near $-2,611$, tracking Random. Oracle, Greedy, and Static MIGAN overlap at $+333.6$. The router did not learn an effective policy. | **Critical** |
| **3** | **"Shifts to Low-Power Models Under Thermal Load"** | In [`03_linucb_action_distribution.png`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/rl_evaluation_figures/03_linucb_action_distribution.png), LinUCB still picks `migan_gpu` (11%) and `lama_npu` (34%) under the Hot regime, while picking slow GPU arms 40% of the time in Cool. Only 6 arms exist (no Telea, no SD). | **High** |
| **4** | **"34°C–75°C Sweep & 0% Thermal Violations"** | [`rl_system_thermal_benchmark.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/rl_system_thermal_benchmark.csv) stops at **51.9°C**. The 75°C ceiling was never reached. Latency across temperature is flat (no throttling observed), giving $T_{\text{SoC}}$ zero predictive signal. | **High** |
| **5** | **"Thermal Hysteresis Loops"** | In [`03_thermal_hysteresis_loops.png`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/rl_thermal_figures/03_memory_footprint.png), every "UP" series is an NPU run and every "DOWN" series is a GPU run. It compares backends, not heating vs cooling on the same hardware block. | **Medium** |
| **6** | **Data-Quality & Axis Clipping Artifacts** | PMIC power swings 0–82 mW without pattern ($r = -0.03$ with latency). The utilization plot hard-clips NPU and GPU axes at 50%, obscuring real values. GPU is artificially pinned at ~99%. | **Medium** |
| **7** | **Slide Claims vs. Physical Data** | Energy in [`SLIDES_REINFORCEMENT_LEARNING_SELECTOR.md`](file:///home/tarsh/Desktop/college/ImageInpainting/todays_presentation/SLIDES_REINFORCEMENT_LEARNING_SELECTOR.md) says $0.089\text{ J}$ & $0.40\text{ J}$, but telemetry shows $2.88\text{ mJ}$ ($25\text{--}100\times$ mismatch). Citation 4 (*"Tan, Ding, Deng et al., TECS 2023"*) is unverified/hallucinated. Dual update thresholds inconsistently state 72°C vs 75°C. | **High** |

---

## 3. Root-Cause Code Diagnostics (The Smoking Guns)

### 3.1 Bug A: Inverted Prior (Latency Maximization)
* **Location**: [`scripts/train_linucb_regression.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/train_linucb_regression.py) & [`scripts/evaluate_linucb_router.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/evaluate_linucb_router.py)
* **Mechanism**: 
  1. `train_linucb_regression.py` computed $b_a = X_a^T y_a$ where $y_a = \textbf{Latency in ms}$ ($b_a$ contains positive values in thousands).
  2. `evaluate_linucb_router.py` loaded this prior directly into UCB arm selection:
     ```python
     ucb_val = (x.T @ theta) + ci
     best_arm = argmax(ucb_val)
     ```
  3. Because UCB maximizes `ucb_val`, the router actively selected the **highest latency arms** (AOT-GAN GPU ~ 600 ms).
  4. During simulation updates:
     ```python
     self.b[arm] += reward * x  # reward is negative scalar ~ -1.5
     ```
     Small negative scalar rewards were added to vectors initialized with thousands of positive latency units, completely corrupting the ridge regression weights.

### 3.2 Bug B: Context Vector Lacks Visual Context
* **Location**: `PolicySimulationEngine.run_simulation` in [`scripts/evaluate_linucb_router.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/evaluate_linucb_router.py)
* **Mechanism**: 
  ```python
  x = np.array([1.0, temp_c, util, batt_v])
  ```
  The context vector contained only hardware telemetry and **zero image or mask properties** (no mask area, no edge density, no texture score). Since hardware latency was invariant across 34°C–52°C, the router had no visual context to trigger model trade-offs.

### 3.3 Bug C: Trivial Global Optimum Under Latency-Dominated Reward
* **Mechanism**: The scalar reward formula penalized latency heavily while quality rank bonuses were small:
  * $\text{MIGAN NPU}$: $R \approx +0.83$
  * $\text{LaMa NPU}$: $R \approx +0.55$
  * $\text{AOT-GAN NPU}$: $R \approx -0.56$
  Because context had no mask size to reward LaMa or AOT-GAN for complex inpainting, MIGAN NPU was statically the highest-reward arm on every single request. Oracle, Greedy, and Static MIGAN were identical by definition.

---

## 4. Step-by-Step Action Plan (Next Session Tasks)

```
                                NEXT SESSION WORKFLOW
+-----------------------------------------------------------------------------------+
| STEP 1: Debug LinUCB Simulation (Feature Context + Proper Inverted Prior/Reward)   |
|                                        ↓                                          |
| STEP 2: Re-run Simulation & Regenerate RL Evaluation Figures                      |
|                                        ↓                                          |
| STEP 3: Reconcile Empirical Telemetry Data (Thermal Ceiling 52°C, 6 Arms, mJ)      |
|                                        ↓                                          |
| STEP 4: Rewrite Presentation Slides (SLIDES_REINFORCEMENT_LEARNING_SELECTOR.md)   |
|                                        ↓                                          |
| STEP 5: Verification & Commit                                                     |
+-----------------------------------------------------------------------------------+
```

### Phase 1: Fix LinUCB Simulation & Prior Generation
- [ ] **Task 1.1: Re-engineer the Context Vector $\mathbf{x}_t$**
  - Update `scripts/evaluate_linucb_router.py` to sample real image features from [`artifacts/dataset_complete_metrics_6000.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/artifacts/dataset_complete_metrics_6000.csv).
  - Define $\mathbf{x}_t = [1.0, \text{MaskAreaRatio}_t, \text{EdgeEnergy}_t, T_{\text{SoC}, t}, \text{BatteryLevel}_t]^T$.
- [ ] **Task 1.2: Formulate Normalized Multi-Objective Utility Reward**
  - Implement a true Pareto reward function:
    $$R(a, \mathbf{x}) = w_q \cdot \text{Quality}(a, \mathbf{x}) - w_l \cdot \frac{\text{Latency}(a)}{\text{TargetLatency}} - w_t \cdot \mathbb{I}(T > T_{\text{budget}})$$
  - Ensure large masks ($>15\%$) or high-frequency textures yield superior reward for LaMa/AOT-GAN, giving the contextual bandit a real trade-off to learn.
- [ ] **Task 1.3: Fix Prior Initialization in [`scripts/train_linucb_regression.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/train_linucb_regression.py)**
  - Initialize $A_a = \lambda I$ and fit $b_a$ on empirical **reward / utility**, NOT raw latency in milliseconds.
  - Or initialize uninformative prior ($b_a = \mathbf{0}$) and demonstrate true online learning from step 0 to step 1,000.
- [ ] **Task 1.4: Fix Plotting Scripts & Axis Clipping**
  - Remove hardcoded `ylim(0, 50)` from utilization plots.
  - Regenerate figures into [`Benchmark/output/rl_evaluation_figures/`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/rl_evaluation_figures/):
    * `01_cumulative_regret.png`: Must exhibit clear concave sub-linear growth ($O(\sqrt{T})$), significantly beating Random and Static baselines.
    * `02_cumulative_reward.png`: Must show positive convergence tracking the Oracle.
    * `03_linucb_action_distribution.png`: Must demonstrate context-dependent arm switching (MIGAN on small strokes, LaMa on wide masks, AOT-GAN on textures).

### Phase 2: Telemetry & Empirical Dataset Realism
- [ ] **Task 2.1: Correct Thermal Operating Range Statements**
  - Update all documentation and slide text to state the true empirical range: **$34.4^\circ\text{C}$ to $51.9^\circ\text{C}$**.
  - Remove claims of "stress tested to 75°C with 0% violations"; state that under passive cooling, the SoC stabilized below 52°C, well within the 75°C critical threshold.
- [ ] **Task 2.2: Standardize Action Space Definition**
  - Explicitly document the **6 evaluated hardware arms**:
    $$\mathcal{A}_{\text{eval}} = \{ \text{MIGAN}_{\text{NPU}}, \text{MIGAN}_{\text{GPU}}, \text{LaMa}_{\text{NPU}}, \text{LaMa}_{\text{GPU}}, \text{AOT}_{\text{NPU}}, \text{AOT}_{\text{GPU}} \}$$
  - Clarify that Classical Telea (ARM64 CPU) and Stable Diffusion 1.5 serve as static fallback / high-compute specialty modes, not part of the high-frequency 6-arm online loop.
- [ ] **Task 2.3: Clarify or Fix Hysteresis Claims**
  - Either remove the claim of thermal hysteresis, or run a true continuous thermal heating/cooling loop on a single designated engine (`migan_npu`).

### Phase 3: Slide & Documentation Overhaul
- [ ] **Task 3.1: Revise [`todays_presentation/SLIDES_REINFORCEMENT_LEARNING_SELECTOR.md`](file:///home/tarsh/Desktop/college/ImageInpainting/todays_presentation/SLIDES_REINFORCEMENT_LEARNING_SELECTOR.md)**
  - **Slide 1**: Retain the strong, mathematically sound critique of Deep RL (PPO) on mobile (sample inefficiency, high battery drain, non-MDP nature).
  - **Slide 2 & 3**: Align context vector $\mathbf{x}$ and action space $\mathcal{A}$ with actual code.
  - **Slide 4**: Fix energy numbers ($2.88\text{ mJ}$ to $6.0\text{ mJ}$ measured vs erroneous $0.089\text{ J}$).
  - **Slide 5**: Present the newly generated, mathematically honest sub-linear regret plot.
  - **Slide 6**: Ensure comparison table values match real benchmark measurements.
  - **Slide 7**: Update the 60-second committee defense script with accurate numbers.
  - **Slide 8**: Replace Citation 4 (*Tan et al.*) with real, verifiable citations:
    * Li, Chu, Langford, Schapire. *"A Contextual-Bandit Approach to Personalized News Article Recommendation"*, WWW 2010.
    * Badanidiyuru, Kleinberg, Slivkins. *"Bandits with Knapsacks"*, FOCS 2013 / JACM 2018.
    * Agarwal et al. *"Taming the Monster: A Fast and Robust Algorithm for Contextual Bandits"*, ICML 2014.
- [ ] **Task 3.2: Reconcile Threshold Consistency**
  - Standardize thermal thresholds across all slides to 75.0°C.
- [ ] **Task 3.3: Update Web Presentation & Dashboard**
  - Update [`Benchmark/output/presentation_slides.html`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/presentation_slides.html) and [`docs/index.html`](file:///home/tarsh/Desktop/college/ImageInpainting/docs/index.html) to reflect the corrected figures and tables.

---

## 5. Master File & Asset Index

### Key Scripts
* [`scripts/train_linucb_regression.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/train_linucb_regression.py): Fits regression priors from telemetry.
* [`scripts/evaluate_linucb_router.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/evaluate_linucb_router.py): Simulates bandit policies and plots evaluation curves.
* [`scripts/rl_thermal_data_collector.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/rl_thermal_data_collector.py): Collects thermal/power feedback on hardware.

### Key Data Files
* [`artifacts/dataset_complete_metrics_6000.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/artifacts/dataset_complete_metrics_6000.csv): Complete 6k image evaluation metrics.
* [`Benchmark/output/rl_system_thermal_benchmark.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/rl_system_thermal_benchmark.csv): Real hardware thermal sweep telemetry.
* [`Benchmark/output/linucb_system_priors.json`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/linucb_system_priors.json): Calibrated regression matrices.
* [`Benchmark/output/linucb_policy_evaluation_metrics.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/linucb_policy_evaluation_metrics.csv): Baseline policy comparative metrics.

### Key Slide & Documentation Files
* [`todays_presentation/SLIDES_REINFORCEMENT_LEARNING_SELECTOR.md`](file:///home/tarsh/Desktop/college/ImageInpainting/todays_presentation/SLIDES_REINFORCEMENT_LEARNING_SELECTOR.md): Markdown presentation deck.
* [`Benchmark/output/presentation_slides.html`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/presentation_slides.html): Standalone HTML slide deck.
* [`docs/index.html`](file:///home/tarsh/Desktop/college/ImageInpainting/docs/index.html): Web dashboard (Tab 6 LinUCB Router).
* [`qidk-inpaint-app-v5/app/src/main/assets/production_ccb_router.json`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/assets/production_ccb_router.json): Deployed production prior.
