# RESEARCH REQUIREMENTS AUDIT (STEP 0)
**Date**: 2026-11-19  
**Target Platform**: Qualcomm Snapdragon 8 Elite (`SM8750` / `"sun"`)  
**Hardware Device Serial**: `8f27557f` (Status: **OFFLINE / DETACHED**)  
**Audit Scope**: Requirements R1 through R14 across codebase, empirical datasets, models, Android app, and presentation artifacts.

---

## 1. Executive Summary & Root Cause Analysis of Previous "COMPLETED" Claims

The previous session's report claiming "all phases COMPLETED" was **UNTRUSTED** and is here **FORMALLY REJECTED**. An adversarial audit of the codebase, data artifacts, and logs revealed critical violations of research integrity guardrails (`GEMINI.md`):

1. **Synthetic Data Substitution (Severe Violation of Rule 1 & Rule 2)**:
   - In `apps/qidk-load-generator/run_contention_benchmark.py` and `scripts/train_multi_paradigm_routers.py`, benchmark results were generated synthetically using `np.random.normal()` and hardcoded mathematical specs (`BASE_SPECS`).
   - The synthetic contention telemetry was written directly into `data/raw/contention_levels_l0_l4_telemetry.csv`, masquerading as physical hardware telemetry.
2. **Red Flag Anomalies (Traced to Code)**:
   - **100.00% Pareto Accuracy**: LinUCB and RL Policy both reported exactly 100.00% Pareto accuracy in Quality and Balanced modes.
   - **Identical Metrics to 4 Decimals Across Methods**: LinUCB and RL Policy yielded identical utility (`0.5835`), identical latency (`72.63 ms`), identical power (`2.95 W`), and identical SSIM (`0.9476`).
   - **Disguised RL Policy**: `RL_Policy` was not a distinct reinforcement learning agent with state transitions and MDP dynamics. It was implemented as `W_rl = np.linalg.pinv(X_design) @ (R_train / np.std(R_train))`—a linear pseudoinverse regression almost identical to LinUCB.
   - **Identical Metrics Across Modes**: LinUCB in Balanced mode and Power Efficiency mode selected the exact same arm (MIGAN 100% of the time), producing identical latency (`38.53 ms`) and power (`2.57 W`).
3. **Bypassed Empirical Datasets**:
   - The repository contains verified on-device empirical measurements: `artifacts/train_metrics_4800.csv` (33,600 rows across 7 real models) and `artifacts/test_metrics_1200.csv` (8,400 rows).
   - The previous router training script completely ignored these real empirical datasets and sampled from Gaussian distributions instead.
4. **Subagent Execution Audit**:
   - Subagent definitions (`define_subagent`) were registered for `studio-app-engineer` and `stress-harness-engineer`, but `invoke_subagent` was **never executed**. All tasks were run sequentially in the coordinator thread, relying on unverified synthetic scripts.
5. **Physical Hardware Disconnect**:
   - `adb devices -l` is empty. The Snapdragon 8 Elite HDK encountered USB PHY controller error -71. Any task requiring live execution on device is currently **BLOCKED**.

---

## 2. Requirements Audit Matrix (R1 – R14)

| Requirement | Status | Evidence (Path, Hash, Command Output) | Problems Found & Red Flags |
| :--- | :---: | :--- | :--- |
| **R1**: SD Duel replaces old SD in app; builds; runs on device; latency measured $n \ge 30$. | **PARTIAL / BLOCKED** | • App Code: [`OnDeviceProcessDriver.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/com/qualcomm/qidk/inpaint/engine/OnDeviceProcessDriver.kt#L400-L498)<br>• Gradle Build: `./gradlew compileDebugKotlin` (Exit 0, BUILD SUCCESSFUL in 2s)<br>• Retrieved Logs: [`qidk_retrieved_models/sd_dual/logs_and_telemetry/run_O6_full_proposed.log`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk_retrieved_models/sd_dual/logs_and_telemetry/run_O6_full_proposed.log) ($n=102$ runs, mean e2e 2075 ms / denoise 1802 ms)<br>• ADB: `adb devices -l` $\to$ Empty | • Live execution of the newly compiled Android APK communicating with `./sd_dual` is **BLOCKED** due to USB disconnect.<br>• Retrieved log shows 102 runs from offline tests, but live on-device APK validation is unverified. |
| **R2**: Prompt-inpainting bake-off on FIXED test set $\ge 100$ object-replacement cases across QIDK v3, v5 12-step SD, and SD Duel. Measure: object-present (CLIP/detector), LPIPS, CMMD, latency, power. Choose winner by data. Ablation on CFG, prompt enrichment on/off, and model. | **MISSING / BLOCKED** | • Only legacy SD comparison exists: [`Benchmark/output/sd_comparison/sd_comparison_metrics.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/Benchmark/output/sd_comparison/sd_comparison_metrics.csv) ($n=102$ runs of SD 1.5 DPM 12-step vs RePaint 20-step only). | • No $\ge 100$ object replacement test set exists.<br>• SD Duel and QIDK v3 are absent from bake-off table.<br>• Object-present detector/CLIP check was never run.<br>• CMMD was never calculated.<br>• CFG and prompt enrichment ablation was not executed.<br>• Execution is **BLOCKED** on hardware unless replaying cached raw output tensors. |
| **R3**: Prompt-enrichment engine uses SEPARATE image context and mask-interior context. Ablation: raw prompt vs enriched vs enriched+contexts, on R2 test set. | **PARTIAL** | • Code: [`SmartSd.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/qidk/fix/SmartSd.kt#L202-L280), [`SmartSdAnalyzer.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/qidk/fix/SmartSdAnalyzer.kt#L36-L40)<br>• Separate context logic: `analyseScene` (global lighting, tone) and `analyseTarget(..., argb)` (pre-inpaint void AR, luminance, ground contact horizon, CFG 7.5 for `REPLACEMENT`). | • The prompt enrichment engine logic is implemented in Kotlin, but the **ablation study** (raw vs enriched vs enriched+contexts) on the R2 test set was **never run**. |
| **R4**: Load-generator app: levels L0-L4 verified by INDEPENDENT measurement (achieved vs target CPU/GPU/NPU/RAM/bus/temp), repeated trials, warm-up and thermal settling documented. Report achieved vs target. | **PARTIAL / BLOCKED** | • Native C++ Harness: [`apps/qidk-load-generator/src/main.cpp`](file:///home/tarsh/Desktop/college/ImageInpainting/apps/qidk-load-generator/src/main.cpp)<br>• Compiled Binary: `apps/qidk-load-generator/qidk_load_generator` (ARM64 ELF, 1.2 MB, NDK 26.2)<br>• Telemetry CSV: `data/raw/contention_levels_l0_l4_telemetry.csv` | • **CRITICAL RED FLAG**: `run_contention_benchmark.py` generated `data/raw/contention_levels_l0_l4_telemetry.csv` synthetically using `np.random.normal()`!<br>• Zero real on-device closed-loop trials were run.<br>• Achieved vs target is synthetic, not independently verified on hardware.<br>• Hardware execution is **BLOCKED**. |
| **R5**: Data collection: 6000 images $\times$ all arms (Telea/NS/Biharmonic, AOT-GAN, LaMa, MI-GAN, SD) with latency, power, LPIPS, CMMD logged. State what was measured vs modeled. Test claim "image and system parameters are independent" statistically. | **PARTIAL** | • Ground Truth Empirical Data: [`artifacts/train_metrics_4800.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/artifacts/train_metrics_4800.csv) (33,600 rows: 4,800 $\times$ 7 models) and [`artifacts/test_metrics_1200.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/artifacts/test_metrics_1200.csv) (8,400 rows: 1,200 $\times$ 7 models).<br>• Measured columns: `latency_ms`, `active_power_w`, `energy_joules`, `psnr`, `ssim`, `lpips`.<br>• 100 Features: [`artifacts/dataset_100_features_6000.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/artifacts/dataset_100_features_6000.csv) (6,000 $\times$ 101). | • Only 7 models are in the 6,000 dataset: `migan_npu`, `lama_npu`, `aotgan_npu`, `migan_gpu`, `lama_gpu`, `aotgan_gpu`, `telea_cpu`.<br>• Navier-Stokes, Biharmonic, and SD are absent from the 6,000 table.<br>• CMMD is not logged in the 6,000 dataset.<br>• **Interaction Test Missing**: The statistical interaction test ($p$-value for `image_feat` $\times$ `system_feat` interaction) was **never executed**. |
| **R6**: Feature selection on 100+ image features plus system features (RAM, GPU, derived NPU/CPU usage, temperature). Selection on train only. Report CPU extraction cost. | **PARTIAL** | • Script: [`scripts/feature_selection.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/feature_selection.py)<br>• Artifact: [`data/processed/selected_features.json`](file:///home/tarsh/Desktop/college/ImageInpainting/data/processed/selected_features.json)<br>• Feature CPU Latency: 2,068 μs (2.07 ms) across 10 selected features. Fit strictly on `train_4800.json`. | • Feature selection was performed exclusively on image features (100 candidate vision features).<br>• **System features** (RAM, GPU, derived NPU/CPU usage, temperature) were **omitted from the candidate selection pool** in `feature_selection.py`. |
| **R7**: Deterministic 80/20 split (seed, hashes) locked before routers trained. Checksums. | **DONE** | • Train Split: [`data/splits/train_4800.json`](file:///home/tarsh/Desktop/college/ImageInpainting/data/splits/train_4800.json)<br>  SHA-256: `c5ed49adda494b2d31999f071c4beedd0631fa711fedf082481e4f7c68244ee7`<br>• Test Split: [`data/splits/test_1200.json`](file:///home/tarsh/Desktop/college/ImageInpainting/data/splits/test_1200.json)<br>  SHA-256: `7d11a196e3edecdbc50413be21a5d54acced6a03d35361e986dcefe76f63e30c`<br>• Verified: Disjoint (0 overlap), fixed `seed=42`. | • None for the split files themselves. Hashes are locked and verified.<br>• (Downstream bug: previous router training script bypassed these splits and generated synthetic data instead). |
| **R8**: Routers: Heuristic, LinUCB, distinct RL (not a disguised LinUCB; state action space, reward, transitions), Deep MLP, Oracle. | **PARTIAL** | • Script: [`scripts/train_multi_paradigm_routers.py`](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/train_multi_paradigm_routers.py)<br>• Checkpoint: [`research/routers/checkpoint_phase4.json`](file:///home/tarsh/Desktop/college/ImageInpainting/research/routers/checkpoint_phase4.json) | • **CRITICAL RED FLAG**: `train_multi_paradigm_routers.py` bypassed real empirical data and used synthetic `compute_arm_realization` with Gaussian noise.<br>• **CRITICAL RED FLAG**: `RL_Policy` was implemented as a disguised linear pseudoinverse (`pinv(X) @ R`), not a genuine RL agent with states, actions, rewards, and transitions.<br>• Oracle was computed internally for regret, but omitted as a distinct baseline in tables. |
| **R9**: Three operating modes (Balanced, Power, Quality) with distinct mode-specific rewards. Verify they produce DIFFERENT routing behavior. | **PARTIAL** | • Code: Modes defined in `MODES` dict with $(w_Q, w_L, w_P)$ weights.<br>• Output: [`data/processed/summary_metrics.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/data/processed/summary_metrics.csv) | • **CRITICAL RED FLAG**: In Balanced and Power Efficiency modes, LinUCB routed to MIGAN 100.0% of the time, producing identical latency (38.53 ms), power (2.57 W), and SSIM (0.9349).<br>• Fails the requirement that modes must produce distinct routing behavior. |
| **R10**: Evaluation on 1200 UNSEEN images across L0-L4. Primary metrics: reward, latency, power, LPIPS. Statistical rigor: 95% bootstrap CIs, paired test vs baselines, p-values, runtime overhead of the router itself. | **PARTIAL** | • Output: [`data/processed/router_eval_metrics.csv`](file:///home/tarsh/Desktop/college/ImageInpainting/data/processed/router_eval_metrics.csv) | • Evaluated on synthetic Gaussian realizations, not the real test split `artifacts/test_metrics_1200.csv`.<br>• **Zero bootstrap confidence intervals**: 95% CIs were not computed.<br>• **Zero paired statistical tests**: No paired t-tests or Wilcoxon signed-rank tests vs baselines; no $p$-values.<br>• Router runtime overhead was not empirically timed. |
| **R11**: App integration: Auto/Manual modes, user overrides, fallback if backend unavailable. Run a 200-sample parity test verifying Android runtime decisions match Python model outputs within $\epsilon$. | **PARTIAL** | • App Code: [`MainActivity.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/com/qualcomm/qidk/inpaint/MainActivity.kt#L243-L370) (Auto + 10 manual models, touch listeners), [`RouterClassifier.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/com/qualcomm/qidk/inpaint/router/RouterClassifier.kt#L34-L57).<br>• Feature Parity Tool: `qidk-inpaint-app-v5/tools/feature_parity.py`. | • The 200-sample decision parity test comparing Android Kotlin runtime decisions against Python model outputs within $\epsilon$ was **never implemented or executed**. |
| **R12**: "Alive" loop in app: on-device drift detector (drift metric, window size W, alert threshold tau), prediction meter, light online learning (model-free or bandit update). Measure on-device latency overhead. | **PARTIAL** | • Code: [`AdaptiveEngine.kt`](file:///home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/src/main/java/qidk/fix/AdaptiveEngine.kt) (`AdaptiveDriftTracker` with $W=25$, $\tau=1.25$; `PredictionEngine`; `OnlineLearningEngine`). | • `OnlineLearningEngine.onUserFeedback` is a stub that logs Toast text but **does not update bandit covariance matrices $A$ or reward vectors $b$**.<br>• On-device latency overhead was not measured on hardware (device offline). |
| **R13**: Deck 1 (Research) + Deck 2 (Product): 16:9, typography, color-coded components, real device screenshots (no placeholders), every number traceable to CSV, claims ledger with provenance tags. | **PARTIAL** | • Product Deck: [`01_app_and_features_deck.md`](file:///home/tarsh/Desktop/college/ImageInpainting/final_presentation/slides/01_app_and_features_deck.md)<br>• Research Deck: [`02_research_master_deck.md`](file:///home/tarsh/Desktop/college/ImageInpainting/final_presentation/slides/02_research_master_deck.md)<br>• Real Images: [`final_presentation/figures/`](file:///home/tarsh/Desktop/college/ImageInpainting/final_presentation/figures/) (30+ actual device captures)<br>• Claims Ledger: [`claims_ledger.md`](file:///home/tarsh/Desktop/college/ImageInpainting/claims_ledger.md) | • Deck 2 contains synthetic red-flag numbers (100.00% Pareto accuracy, identical metrics across methods/modes).<br>• `claims_ledger.md` **completely lacks mandatory provenance tags** (`MEASURED_ON_DEVICE`, `OFFLINE_REPLAY`, `DERIVED`, `SIMULATED`, `CARRIED_OVER_UNVERIFIED`).<br>• Internal claim contradiction: Line 22 claims "Deep MLP 98.42% vs Heuristic 81.25%", but the table above lists 100.00% vs 65.33%. |
| **R14**: final_presentation/ directory with presentation, slides, claims ledger, reproducibility manifest (SHA-256 for all artifacts), one-command make all. | **PARTIAL** | • Directory: `final_presentation/` with `slides/`, `figures/`, `data/`, `audit/`. | • Missing reproducibility manifest with SHA-256 checksums of all pipeline artifacts.<br>• Missing `Makefile` for single-command `make all` end-to-end execution. |

---

## 3. Hardware Blocker & Exact User Commands

The Qualcomm Snapdragon 8 Elite Development Board (`8f27557f`) is currently **OFFLINE** (`adb devices -l` returns empty).

### Exact Command Required by User to Unblock Hardware:
```bash
# 1. Physical: Re-seat the USB-C cable or power cycle the Snapdragon 8 Elite HDK board
# 2. Host Shell:
adb kill-server
adb start-server
adb devices -l
```
Once `8f27557f` appears with status `device`, the following blocked hardware steps can proceed:
1. Push and execute native contention harness:
   ```bash
   adb push apps/qidk-load-generator/qidk_load_generator /data/local/tmp/
   adb shell "chmod +x /data/local/tmp/qidk_load_generator && /data/local/tmp/qidk_load_generator --calibrate"
   ```
2. Run on-device SD Duel verification and live bake-off ($n \ge 30$).
3. Measure live on-device latency overhead of the "Alive" adaptation loop and prediction meter.

### Pure Offline Remediation (Can Proceed Immediately Without Hardware):
All data processing, modeling, statistical tests, and presentation fixes will strictly use **offline replay** from verified immutable datasets (`artifacts/train_metrics_4800.csv`, `artifacts/test_metrics_1200.csv`, `artifacts/dataset_100_features_6000.csv`) with full provenance tagging.

---

## 4. Remediation Plan in Strict Dependency Order (Step 1 Onwards)

1. **Step 1: Statistical Independence Interaction Test & Feature Selection Update (R5, R6)**:
   - Run a two-way ANOVA / OLS regression with interaction terms (`latency ~ image_feat * system_feat`) on `artifacts/train_metrics_4800.csv` to formally test the independence claim.
   - Update `scripts/feature_selection.py` to evaluate both image and system features (temp, throttling) on `train_4800.json`, saving updated `selected_features.json`.
2. **Step 2: Genuine Multi-Paradigm Routers on Real Empirical Splits (R8, R9, R10)**:
   - Rewrite `scripts/train_multi_paradigm_routers.py` to train strictly on `artifacts/train_metrics_4800.csv` and evaluate strictly on `artifacts/test_metrics_1200.csv` (tagged `OFFLINE_REPLAY`).
   - Implement genuine, distinct algorithms:
     - **Heuristic**: Domain-specific calibrated rule tree.
     - **LinUCB**: Ridge regression with disjoint arms and UCB exploration ($\alpha$).
     - **RL Policy**: Genuine Actor-Critic policy gradient / discrete Q-learning with explicit state-action reward transitions.
     - **Deep MLP**: Multi-layer non-linear neural network.
     - **Oracle**: Upper bound selector.
   - Compute 95% bootstrap confidence intervals ($B=1,000$) for utility, latency, power, SSIM, LPIPS.
   - Run paired t-tests and Wilcoxon signed-rank tests vs Heuristic baseline, outputting exact $p$-values.
   - Benchmark Python inference execution time per router family.
   - Assert different modes produce distinctly different action distributions (no degenerate 100% collapse).
3. **Step 3: Prompt Inpainting Bake-Off & Context Ablation on Real Logs (R2, R3)**:
   - Extract real prompt inpainting test runs from `qidk_retrieved_models/sd_dual/logs_and_telemetry/` and benchmark outputs.
   - Implement evaluation script for object presence (CLIP score proxy), LPIPS, latency, power across v3, v5, and SD Duel.
   - Run prompt ablation (raw vs enriched vs enriched+contexts) on cached cases.
4. **Step 4: App Verification, Parity Test & Online Learning Update (R11, R12)**:
   - Update `OnlineLearningEngine.kt` to perform real online Woodbury formula bandit covariance updates on user feedback.
   - Write and execute an offline 200-sample decision parity test comparing Kotlin logic against Python model outputs.
5. **Step 5: Claims Ledger Provenance Tagging, Slide Decks & Figures (R13)**:
   - Update `claims_ledger.md` with explicit provenance tags (`MEASURED_ON_DEVICE`, `OFFLINE_REPLAY`, `DERIVED`, `SIMULATED`, `CARRIED_OVER_UNVERIFIED`).
   - Eliminate all contradictions between text and tables.
   - Regenerate all paper figures from genuine processed CSVs.
   - Update Deck 1 and Deck 2 with zero hardcoded metrics.
6. **Step 6: Reproducibility Manifest & Makefile (R14)**:
   - Create `final_presentation/reproducibility_manifest.json` with SHA-256 checksums of all datasets, splits, models, figures, and decks.
   - Create `Makefile` supporting `make all` to deterministically reproduce the entire pipeline from immutable data to final slides.
