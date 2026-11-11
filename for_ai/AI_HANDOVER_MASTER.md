# AI Handover Master Documentation
**Project**: Edge-Native Neural Inpainting & Super-Resolution Framework on Snapdragon 8 Elite  
**Target Hardware**: Qualcomm Snapdragon 8 Elite Mobile Platform (`SM8750P` / `sun`, Serial: `8f27557f`, Hexagon HTP v79 NPU, Adreno 830 GPU, Oryon CPU)  
**Workspace**: `/home/tarsh/Desktop/college/ImageInpainting`  
**Author**: Antigravity Autonomous Coding Agent  
**Intended Recipient**: Incoming AI Agent (Claude, GPT-4, etc.) / Research Engineering Team  

---

## 1. Executive Summary & Hardware Architecture

This repository hosts a production-grade, edge-native image inpainting, generative synthesis, super-resolution, and reinforcement learning hardware routing engine running on Android 15. The core pipeline is deployed on Qualcomm Snapdragon 8 Elite hardware.

```mermaid
flowchart TD
    User["User / UI Layer\n(Android Kotlin / Jetpack / Material 3)"] --> Controller["Inpaint / SR Controller\n(ViewModel & Bitmap Viewports)"]
    Controller --> Router["Reinforcement Learning Router\n(LinUCB Contextual Bandit - 5D Context)"]
    Router -->|Selects Hardware Arm| Engine["Hardware Execution Engines"]
    
    subgraph Snapdragon 8 Elite Mobile Platform
        Engine -->|NPU / HTP v79| FastRPC["Resident FastRPC Daemon\n(/vendor/bin/inpainting_daemon)"]
        Engine -->|GPU / Adreno 830| OpenCL["QNN OpenCL / Direct ML Accelerator"]
        Engine -->|CPU / Oryon| OpenCV["OpenCV C++ (Telea Fast Marching)"]
        Engine -->|Native C++ CLI| SDRoot["RealSD Runner (sd_qidk_runner_inpaint_real)"]
        
        FastRPC -->|HTP Shared Memory / tmpfs| Models["QNN DLC Models (512x512 Float32/UInt8)\nMIGAN | LaMa | AOT-GAN"]
        SDRoot -->|Quantized UNet/VAE/CLIP| SDModel["Stable Diffusion 1.5 Inpainting"]
    end
    
    FastRPC --> Output["Processed Tensor / Bitmap"]
    OpenCL --> Output
    OpenCV --> Output
    SDRoot --> Output
    Output --> UI["Synchronized Dual-Viewport UI\n(Zoom/Pan/Comparison Slider)"]
```

### Hardware Specifications
- **SoC**: Qualcomm SM8750P (Snapdragon 8 Elite / "sun").
- **NPU**: Qualcomm Hexagon Tensor Processor (HTP) Architecture v79. Sustains up to 75–80 TOPS INT4/INT8 and native FP16/FP32 quantized execution via Qualcomm Neural Network (QNN) SDK.
- **GPU**: Adreno 830 with OpenCL 3.0 Full Profile and native FP16 execution units.
- **CPU**: Qualcomm Oryon CPU cluster (2x Prime cores @ 4.32 GHz, 6x Performance cores @ 3.53 GHz).
- **Communication Protocol**: FastRPC (Fast Remote Procedure Call) over shared physical contiguous memory (`/dev/ion` / dmabuf memory pools) interfacing with Hexagon DSP via `/vendor/bin/inpainting_daemon`.
- **Target Device Serial**: `8f27557f` (accessible via `adb -s 8f27557f`).

---

## 2. Directory Structure of `for_ai/`

```
for_ai/
├── AI_HANDOVER_MASTER.md       <-- THIS FILE: Master reference & architectural handover
├── app/                        <-- Android application source code (Kotlin, Java, C++, Layouts)
│   ├── src/                    <-- All source packages (engine, ui, telemetry, router)
│   ├── build.gradle.kts        <-- Module build configuration
│   └── AndroidManifest.xml     <-- App permissions & hardware feature flags
├── scripts/                    <-- Production benchmark & RL router scripts
│   ├── evaluate_linucb_router.py      <-- 1,000-request LinUCB evaluation & simulation
│   ├── train_linucb_regression.py     <-- OLS regression prior fitter for 5D context
│   ├── rl_thermal_data_collector.py   <-- Telemetry collector & thermal curve plotting
│   ├── benchmark_inpaint_models.py    <-- 6k dataset benchmark harness (LPIPS/PSNR/SSIM)
│   └── run_native_sd_pipeline.py      <-- Native ADB C++ runner wrapper for Stable Diffusion
├── tools/                      <-- Developer tools & desktop viewers
│   └── desktop_sr_viewer.py    <-- PyQt5 synchronized multi-model Super-Resolution viewer
├── benchmark_data/             <-- Empirical benchmark outputs, JSON priors, evaluation metrics
│   ├── linucb_system_priors.json              <-- Extracted A_inv and b priors for 6 hardware arms
│   ├── linucb_policy_evaluation_metrics.csv   <-- 1,000-request empirical metrics comparison
│   ├── rl_evaluation_figures/                 <-- Publication-ready RL evaluation plots (PNG)
│   └── rl_thermal_figures/                    <-- Publication-ready thermal telemetry plots (PNG)
├── upscale_example/            <-- Sample 001 comparative benchmark assets & report
│   ├── UPSCALE_SAMPLE_001_ANALYSIS.md         <-- Deep physical analysis of all models & scales
│   └── images/                                <-- Original & upscaled output PNGs (Tarsh, MIGAN, LaMa, AOT)
└── documentation/              <-- Academic presentation slides, designs, and task trackers
    ├── TODO.md                                <-- Completed engineering tasks and solutions
    ├── APP_DESIGN_REPORT.md                   <-- Exhaustive UI/UX console design specification
    ├── SLIDES_REINFORCEMENT_LEARNING_SELECTOR.md <-- 8-slide presentation script & speaker notes
    └── index.html                             <-- Interactive web dashboard for benchmark metrics
```

---

## 3. Comprehensive Model Catalog

The framework integrates 5 inpainting engines and 5 super-resolution pipelines:

| Model Backbone | Architecture / Method | Native Input Resolution | Primary Hardware Target | FastRPC Daemon ID | Key Strengths & Failure Modes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MIGAN** | Edge-guided multi-stage GAN with spatial modulation | $512 \times 512 \times 3$ (Float32, $[0, 1]$) | HTP v79 NPU (`migan_npu`) / Adreno 830 (`migan_gpu`) | `migan` | Ultra-fast ($27.2\text{ ms}$ on NPU), sharp boundary reconstruction. Fails on textures if input is normalized to $[-1, 1]$. |
| **LaMa** | Fast Fourier Convolutions (FFC) with global receptive field | $512 \times 512 \times 3$ (Float32, $[0, 1]$) | HTP v79 NPU (`lama_npu`) / Adreno 830 (`lama_gpu`) | `lama` | Excellent structural and periodic texture completion. Fails if single-pixel lattices are fed without pre-interpolation. |
| **AOT-GAN** | Aggregated Contextual Transformations with dilated blocks | $512 \times 512 \times 3$ (Float32, $[0, 1]$) | HTP v79 NPU (`aot_npu`) / Adreno 830 (`aot_gpu`) | `aotgan` | Robust general-purpose inpainting ($58.4\text{ ms}$ on NPU). Resistant to normalization errors. |
| **Stable Diffusion 1.5** | Latent Diffusion with quantized UNet/VAE/CLIP | $512 \times 512 \times 3$ (Latents $64 \times 64 \times 4$) | HTP v79 NPU (`sd_qidk_runner_inpaint_real`) | Native CLI | Exceptional semantic synthesis and photorealism. Slow ($1,200\text{--}2,400\text{ ms}$); positional CLI args required. |
| **Telea Inpainting** | Fast Marching Method (FMM) | Arbitrary (CPU memory bound) | Oryon CPU (`cpu_telea`) | In-process C++ | Zero model size, ultra-low energy ($1.1\text{ mJ}$), high speed ($12.5\text{ ms}$ for small holes). Blurs large areas. |
| **Tarsh SR (Fast)** | Guided Bilinear + Unsharp Masking + Sub-pixel Anchor | $2\times, 4\times, 8\times$ Scaling | Oryon CPU / GPU Compute | In-process Kotlin | Zero inference latency ($10\text{--}40\text{ ms}$), stable, sharp edges. No hallucinated textures. |
| **GAN SR (MIGAN/LaMa/AOT)** | Sub-pixel interleaving + Tile-based Inpainting Synthesis | $2\times, 4\times, 8\times$ Scaling | HTP v79 NPU via Daemon | `migan` / `lama` / `aotgan` | Deep high-frequency synthesis. Requires guided interpolation prior to inpainting. |
| **SD Latent SR** | Latent img2img reverse diffusion ($\sigma = 0.35$) | $2\times$ Scaling ($512 \to 1024$) | HTP v79 NPU | Native CLI | Superb photorealistic details. Must avoid independent quadrant tiling. |

---

## 4. Reinforcement Learning / Contextual Bandit Engine

To dynamically route between the 6 neural/classical arms (`migan_npu`, `migan_gpu`, `lama_npu`, `lama_gpu`, `aot_npu`, `aot_gpu`), the system implements a **LinUCB Contextual Bandit with Disjoint Linear Models** under thermal, energy, and visual constraints.

### 4.1. Mathematical Formulation

#### 5-Dimensional Context Vector $\mathbf{x}_t \in \mathbb{R}^5$
The context vector encodes both visual complexity and mobile device hardware telemetry:
$$\mathbf{x}_t = \begin{bmatrix} 1.0 \\ \frac{\text{MaskAreaRatio}}{0.30} \\ \frac{\text{EdgeEnergy}}{0.05} \\ \frac{T - 30.0}{30.0} \\ \frac{V - 8.0}{2.0} \end{bmatrix}$$
where:
- $\text{MaskAreaRatio} \in [0, 1]$: Fraction of pixels masked for inpainting.
- $\text{EdgeEnergy} \in [0, 1]$: Mean Canny gradient intensity of the boundary ring around the mask.
- $T \in [30^\circ\text{C}, 60^\circ\text{C}]$: Instantaneous SoC junction temperature.
- $V \in [8.0\text{V}, 10.0\text{V}]$: Battery rail voltage.

#### Multi-Objective Pareto Utility Reward Function
The scalar reward $r_t(a)$ balances execution latency, active power consumption, synthesized visual quality, and thermal safety:
$$r_t(a) = 1.0 - w_{\text{lat}} \cdot \frac{L_a}{600.0} - w_{\text{pow}} \cdot \frac{P_a}{80.0} + w_{\text{qual}} \cdot Q_a - \lambda_{\theta} \cdot \max(0, T - 75.0)$$
with production weights:
- $w_{\text{lat}} = 0.40$ (Latency penalty, normalized against $600\text{ ms}$ max ceiling).
- $w_{\text{pow}} = 0.20$ (Power penalty, normalized against $80\text{ mW}$ active draw).
- $w_{\text{qual}} = 0.40$ (Quality reward, derived from PSNR/LPIPS/SSIM).
- $\lambda_{\theta} = 10.0$ (Severe barrier penalty for exceeding the $75^\circ\text{C}$ thermal throttling threshold).

#### LinUCB Arm Selection Rule
For each candidate arm $a \in \mathcal{A}$:
$$\hat{\theta}_a = \mathbf{A}_a^{-1} \mathbf{b}_a$$
$$\text{UCB}_a(t) = \hat{\theta}_a^T \mathbf{x}_t + \alpha \sqrt{\mathbf{x}_t^T \mathbf{A}_a^{-1} \mathbf{x}_t}$$
$$a_t = \arg\max_{a \in \mathcal{A}} \text{UCB}_a(t)$$
where $\alpha = 0.20$ controls exploration confidence, $\mathbf{A}_a \in \mathbb{R}^{5 \times 5}$ tracks covariance, and $\mathbf{b}_a \in \mathbb{R}^5$ tracks cumulative rewards.

### 4.2. Empirical Evaluation Across 1,000 Requests

From `Benchmark/output/linucb_policy_evaluation_metrics.csv`:

| Policy | Cumulative Reward | Cumulative Regret | Mean Regret / Req | Mean Latency (ms) | Mean Power (mW) | Thermal Violations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oracle Optimal** | **+1711.7** | **0.0** | **0.0000** | $104.2\text{ ms}$ | $39.5\text{ mW}$ | 0.0% |
| **LinUCB Adaptive ($\alpha=0.2$)** | **+1709.6** | **2.1** | **0.0021** | $104.1\text{ ms}$ | $39.6\text{ mW}$ | **0.1%** |
| **Static LaMa NPU** | +1533.6 | 178.1 | 0.1781 | $112.5\text{ ms}$ | $41.3\text{ mW}$ | 0.0% |
| **Greedy / Static MIGAN NPU** | +1505.2 | 206.5 | 0.2065 | $27.2\text{ ms}$ | $38.4\text{ mW}$ | 0.0% |
| **Static AOT-GAN NPU** | +1442.8 | 268.9 | 0.2689 | $58.4\text{ ms}$ | $39.1\text{ mW}$ | 0.0% |
| **Static AOT-GAN GPU** | +1289.4 | 422.3 | 0.4223 | $239.5\text{ ms}$ | $40.8\text{ mW}$ | 0.0% |
| **Uniform Random** | +971.7 | 740.0 | 0.7400 | $223.5\text{ ms}$ | $40.5\text{ mW}$ | 13.8% |

**Key Takeaways**:
1. LinUCB achieves a sub-linear $O(\sqrt{T})$ cumulative regret curve, plateauing at only **2.1 total regret** over 1,000 requests.
2. LinUCB surpasses the naive Greedy/Static MIGAN policy by $+204.4$ reward points because it correctly promotes LaMa when high edge complexity demands FFC structural synthesis, and balances thermal safety under continuous execution.

---

## 5. Major Bugs, Root Causes, and Proven Fixes

This section documents the 6 critical bugs encountered in the system, complete with physical root causes and code snippets.

### Bug 1: MIGAN & LaMa Produce Glitched / Pure White Outputs in Super-Resolution

#### Root Cause
In `NeuralSrEngine.kt:531`, the preprocessing logic had:
```kotlin
// INCORRECT (BUGGY):
val isMinusOneToOne = (activeModel == "migan" || activeModel == "lama")
```
Hexagon HTP v79 QNN DLC execution expects inputs in the **$[0.0, 1.0]$** range. Passing $[-1.0, 1.0]$ caused:
- **MIGAN**: Clamped negative numbers to $0$, dropping mean pixel intensity on synthesized subpixels to $11.8$, creating a visible dark checkerboard.
- **LaMa**: Fast Fourier Convolutions (FFC) amplified negative values into spectral overflow, producing washed out white saturation ($193.5$ mean intensity).
- **AOT-GAN**: Had `isMinusOneToOne = false`, receiving $[0.0, 1.0]$, which is why it was the only model functioning correctly.

#### Proven Fix
In `NeuralSrEngine.kt`:
```kotlin
// CORRECT (FIXED):
val isMinusOneToOne = false // All QNN DLC backbones on HTP v79 expect [0.0, 1.0] Float32 tensors
```

---

### Bug 2: Touch Zoom / Pan Freezing and Stalling in Image Viewports

#### Root Cause
In `SingleImageViewport.kt:180-220`:
1. `isScaling` was set to `true` on `ACTION_POINTER_DOWN`, but `ACTION_POINTER_UP` was never handled. When the user lifted one finger to start panning, `isScaling` remained `true`, completely locking out translation.
2. In `onScale(detector: ScaleGestureDetector)`:
```kotlin
// INCORRECT (BUGGY):
val fx = detector.focusX
val fy = detector.focusY
val bx = mapper.viewToBitmapX(fx)
val by = mapper.viewToBitmapY(fy)
// mapper.scale was already updated by detector.scaleFactor, causing:
val newPanX = fx - mapper.bitmapToViewX(bx) // Exactly equals 0! Pan snapped to (0,0)
```

#### Proven Fix
In `SingleImageViewport.kt`:
1. Add `MotionEvent.ACTION_POINTER_UP` handling:
```kotlin
MotionEvent.ACTION_POINTER_UP -> {
    if (event.pointerCount <= 2) {
        isScaling = false
        val remainingIndex = if (event.actionIndex == 0) 1 else 0
        lastTouchX = event.getX(remainingIndex)
        lastTouchY = event.getY(remainingIndex)
    }
}
```
2. Correct the focal zoom math:
```kotlin
val factor = detector.scaleFactor
val fx = detector.focusX
val fy = detector.focusY
mapper.zoomTo(mapper.scale * factor, fx, fy)
```

---

### Bug 3: Missing Telemetry & Breakdown Card on Super-Resolution Tab

#### Root Cause
`TelemetryBarView.kt` was initially implemented as a persistent top bar rather than an end-of-run telemetry card. In `SuperResolutionFragment.kt`, metrics were not captured from the native daemon across the 4 distinct execution phases.

#### Proven Fix
Implement phase instrumentation in `NeuralSrEngine.kt`:
```kotlin
val t0 = System.nanoTime()
val tPre = measureTimeMillis { prepareTiles(src) }
val tInf = measureTimeMillis { runFastRpcDaemonInference() }
val tPost = measureTimeMillis { stitchAndBlend(tiles) }
val tTotal = (System.nanoTime() - t0) / 1_000_000.0

val telemetry = SrTelemetryResult(
    prepMs = tPre,
    inferenceMs = tInf,
    stitchMs = tPost,
    totalWallMs = tTotal,
    startTempC = pmic.getSoCTemp(),
    endTempC = pmic.getSoCTemp(),
    activePowerMw = pmic.getActivePower(),
    totalEnergyMilliJoules = (tTotal / 1000.0) * pmic.getActivePower()
)
```

---

### Bug 4: Stable Diffusion Native C++ CLI Argument Parsing Failure

#### Root Cause
The native ELF executable on device `/vendor/bin/sd_qidk_runner_inpaint_real` was built with positional arguments:
- `argv[1]`: Positive Prompt string.
- `argv[2]`: Denoising Step Count (integer, e.g., `20`).
- The negative prompt is hardcoded in the binary's `.rodata` section at offset `0x1cf13`.
When Python or Android executed `sd_qidk_runner_inpaint_real --prompt "sunglasses" --steps 20`, the binary interpreted `argv[1]` as literally `"--prompt"`, completely ignoring `"sunglasses"`.

#### Proven Fix
In `run_native_sd_pipeline.py` and `InpaintEngine.kt`:
```bash
# Correct invocation syntax:
/data/local/tmp/sd_qidk_runner_inpaint_real "a high quality photo of sunglasses on a face" 20
```

---

### Bug 5: LinUCB Inverted Prior Bug Causing Worst-Policy Selection

#### Root Cause
In `scripts/train_linucb_regression.py`, the linear model was fitting:
$$\mathbf{b}_a = \mathbf{X}_a^T \mathbf{y}_a \quad \text{where } \mathbf{y}_a = \text{latency (in ms, e.g. 60–600)}$$
Because LinUCB selects the arm that **maximizes** $\hat{\theta}_a^T \mathbf{x}_t$, higher positive coefficients meant the bandit actively chose the slowest, most power-hungry arms.

#### Proven Fix
Fit $\mathbf{b}_a$ directly on the normalized multi-objective Pareto reward:
```python
# scripts/train_linucb_regression.py:
reward = 1.0 - 0.40 * (lat / 600.0) - 0.20 * (pow_mw / 80.0) + 0.40 * quality - thermal_penalty
b_a = np.dot(X_a.T, rewards)
```

---

### Bug 6: Tarsh 8x Super-Resolution Out-Of-Memory (OOM) Crash

#### Root Cause
Upscaling a $512 \times 512$ image by $8\times$ yields $4096 \times 4096 \times 4\text{ bytes} = 67.1\text{ MB}$ per uncompressed `Bitmap`. Allocating 3 intermediate buffers (source, guided bicubic, and output unsharp mask) concurrently in Android ART heap exceeded the app's default per-process memory limit (`256 MB`), causing `java.lang.OutOfMemoryError`.

#### Proven Fix
1. Enable `android:largeHeap="true"` in `AndroidManifest.xml`.
2. Slicing into $1024 \times 1024$ tile strides with $64\text{ px}$ overlap blending, processing and recycling native memory iteratively:
```kotlin
val options = BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 }
bitmap.recycle() // Recycle intermediate source tiles immediately
```

---

## 6. How to Run Benchmarks & Reproduce Results

### 1. LinUCB Contextual Bandit Simulation & Metrics
```bash
cd /home/tarsh/Desktop/college/ImageInpainting
# Fit priors from telemetry and 6k dataset
python3 scripts/train_linucb_regression.py

# Simulate 1,000 requests across 7 policies and render regret/reward curves
python3 scripts/evaluate_linucb_router.py
```
Outputs:
- Prior definitions: `Benchmark/output/linucb_system_priors.json`
- Metrics table: `Benchmark/output/linucb_policy_evaluation_metrics.csv`
- Evaluation curves: `Benchmark/output/rl_evaluation_figures/`

### 2. Thermal Telemetry Collector
```bash
# Collect live thermal telemetry from Snapdragon 8 Elite device:
python3 scripts/rl_thermal_data_collector.py --serial 8f27557f --duration 300

# Re-plot existing thermal data without re-running device tests:
python3 scripts/rl_thermal_data_collector.py --plot-only
```
Outputs:
- Figures: `Benchmark/output/rl_thermal_figures/`

### 3. Desktop Synchronized Multi-Model Super-Resolution Viewer
```bash
# Launch interactive PyQt5 viewer comparing all models at 2x, 4x, 8x:
python3 tools/desktop_sr_viewer.py
```
- Features: Dual image canvas, synchronized pan & zoom across scales, split-screen comparison slider, dark/light theme toggle.

### 4. Compiling and Installing the Android APK
```bash
cd /home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5
./gradlew assembleDebug
adb -s 8f27557f install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## 7. App UI/UX Design System Tokens

From `APP_DESIGN_REPORT.md`:

| Semantic Token | Dark Mode Hex | Light Mode Hex | Usage |
| :--- | :--- | :--- | :--- |
| `background` | `#0B0F19` | `#F8FAFC` | App root window canvas |
| `surface_primary` | `#161F30` | `#FFFFFF` | Model cards, viewport frames |
| `surface_elevated` | `#1E293B` | `#F1F5F9` | Dropdowns, dialogs, sliders |
| `accent_teal` | `#00F2FE` | `#0284C7` | NPU arm highlights, primary buttons |
| `accent_blue` | `#4FACFE` | `#2563EB` | Active toggles, focus rings |
| `accent_orange` | `#FF8A00` | `#EA580C` | GPU arm badge, thermal warnings |
| `accent_danger` | `#EF4444` | `#DC2626` | Thermal throttling violations ($>75^\circ\text{C}$) |
| `text_primary` | `#F8FAFC` | `#0F172A` | Card headers, primary labels |
| `text_secondary` | `#94A3B8` | `#64748B` | Subtext, latency readouts |

---

## 8. Guidance for Incoming AI / Next Engineer

When continuing work on this codebase:
1. **Always preserve 5D Context Normalization**: If you adjust features, remember that $\mathbf{x}_t$ must stay within range $[-1, 1]$ or $[0, 1]$ to avoid numerical explosion in $\mathbf{A}_a^{-1}$.
2. **Never change QNN DLC tensor inputs to $[-1, 1]$**: The Hexagon HTP v79 kernels for MIGAN, LaMa, and AOT-GAN are compiled with unsigned UINT8 / Float32 $[0.0, 1.0]$ ranges.
3. **Respect Android UI Threading**: All FastRPC and JNI inpainting calls must execute within `Dispatchers.Default` or a dedicated background executor thread. Never block the UI thread during bitmap encoding.
4. **FastRPC Daemon Memory Management**: Always clean up `/data/local/tmp/` input/output tensors after execution to prevent filling device storage.
