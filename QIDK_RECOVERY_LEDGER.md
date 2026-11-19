# QIDK RESILIENCE & EXECUTION RECOVERY LEDGER
**Hardware Serial**: `8f27557f` (`SUNP-HDK_SN:D975B15E`, Platform `"sun"` / `SM8750P`)  
**Target OS**: Android 15 (API 35, Kernel 6.6)  
**Hardware Status**: **ONLINE / VERIFIED** (`transport_id:3`, battery 100%, ambient ~32°C)  
**Safety Protocol**: Rate-limit ADB transfers $\le 12\text{ MB/s}$ to prevent USB PHY resets; verify atomic checkpoint after every batch.

---

## 1. Master Phase Progress Matrix

| Phase | Description | Lead Subagent / Track | Status | Atomic Checkpoint File | Gate Verification Rule |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **Phase 1** | App Upgrade: SD Duel Integration, Dual-Context Prompt Engine & Deck 1 | `agent-studio-app` | **COMPLETED** | `qidk-inpaint-app-v5/checkpoint_phase1.json` | SD Duel functional on NPU; Deck 1 compiled in `final_presentation/` |
| **Phase 2** | Controlled Multi-Workload Contention App (`L0` to `L4`) | `agent-stress-app` | **COMPLETED** | `apps/qidk-load-generator/checkpoint_phase2.json` | 738 physical on-device samples across 15 trials (L0-L4) verified |
| **Phase 3** | Factorized 6k Profiling, Feature Selection & Locked 80/20 Splits | `agent-data-profiler` | **COMPLETED** | `data/splits/checkpoint_phase3.json` | `train_4800.json` & `test_1200.json` locked with SHA-256; zero data leakage |
| **Phase 4** | Multi-Paradigm Router Training (3 Modes $\times$ 5 Families) | `agent-router-ml` | **COMPLETED** | `research/routers/checkpoint_phase4.json` | Genuine RL Q-Network (Bellman TD updates); evaluation on `test_1200` with 95% CIs |
| **Phase 5** | In-App Control Surface + "Alive" Real-Time Adaptation Loop | `agent-studio-app` | **COMPLETED** | `qidk-inpaint-app-v5/checkpoint_phase5.json` | Sherman-Morrison rank-1 online update + Drift Tracker ($W=25$) + Prediction Meter (23.8 μs total overhead) |
| **Phase 6** | Publication-Ready Plotting, Deck 2 & Adversarial Audit | `agent-paper-auditor` | **COMPLETED** | `final_presentation/audit/checkpoint_phase6.json` | 100% vector PDF/SVG + 300 DPI PNG; 24-point audit passed with 0 discrepancies |

---

## 2. Hardware Connection Ledger

### Event: 2026-11-19 01:32:48
- **Status**: Disconnected (USB PHY reset error -71).
- **Resolution**: Port reset and re-enumeration on host USB 1-13 (`8f27557f device`).

### Event: 2026-11-19 08:05:00
- **Status**: On-Device Storage Reclaimed & App SD Duel / Smart SD Architecture Repaired.
- **Physical Verification & Fixes**:
  1. **Storage Cleanup**: Identified 17.2 GB in `/data/local/tmp` and 10.0 GB in `/data/media/0`. Reclaimed **5.4 GB** of redundant files (old output directories in `sd_dual/outputs_*`, unused experimental branches `models_k0`/`models_k1`, and obsolete 1.6 GB UNet binary). `/data` partition recovered from 98% (1.2 GB available) to 86% (**6.6 GB available**).
  2. **SD Duel vs Smart SD Separation**:
     - *Root cause 1*: `sd_dual` was designed for ultra-fast unconditioned inpainting (628 ms - 1.5s) using Taylor Order-1 cache extrapolation with fixed $\text{CFG}=1.0$ and $B=1$. It does not perform Classifier-Free Guidance.
     - *Root cause 2*: In `OnDeviceProcessDriver.kt`, `hasDualBin` presence caused all SD models (including Smart SD and Standard SD) to be hijacked by `sd_dual`.
     - *Root cause 3*: `SmartSd.kt` previously bloated the prompt with 60 words of boilerplate ("ultra realistic, ambient occlusion..."), exhausting CLIP's 77-token budget.
     - *Root cause 4*: `MainActivity.kt` checked `if (targetModel == "SD")`, leaving `plan = null` when `SMART_SD` was selected.
  3. **Repairs Applied**:
     - Separated `SD_DUEL` (routes to `/data/local/tmp/sd_dual/sd_dual`, 1.5s NPU) and `SMART_SD` (routes to `/data/local/tmp/sd_runtime/sd_qidk_runner_inpaint`, CFG 7.5 dual-pass NPU guidance).
     - Front-loaded concise prompt templates in `SmartSd.kt`.
     - Unit tests verified (70/70 passing).
     - Built and installed updated `app-debug.apk` on `8f27557f`.
     - Verified physical execution of both `sd_dual` (628 ms warm) and `Smart SD` (19.2s, 47k pixel delta with CFG 7.5 prompt steering) on Hexagon NPU.
