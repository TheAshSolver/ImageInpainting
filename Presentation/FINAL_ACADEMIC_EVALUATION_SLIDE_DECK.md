# Neural Image Inpainting on Qualcomm Snapdragon 8 Elite (Hexagon HTP v79)
## Comprehensive Academic Slide Deck & Technical Evaluation Defense (Latest Measured Data)
**Course / Module**: Embedded Systems Workshop (ESW), IIIT Hyderabad  
**Faculty Guide**: Dr. Deepak Gangadharan  
**Teaching Assistant / Mentor**: Vedant Tejas  
**Team Members**: Ashwin Aravind, Tarsh Singh, Raghavendra Navneet Nawander, Aryan Kulkarni  
**Target Hardware Platform**: Qualcomm Snapdragon 8 Elite Development Kit (SM8750P / Platform `sun` / APQ8750 / Device `8f27557f`)  
**Compute Acceleration Engines**: Qualcomm Hexagon NPU (HTP v79 Architecture, FastRPC), Qualcomm Adreno 830 GPU, Qualcomm Oryon CPU  
**Operating Environment**: Android 15 (API Level 35) / Linux Kernel 7.0+  
**Benchmark Provenance**: Latest Measured On-Device Benchmark in `Benchmark/output/fresh_benchmark/20261008_035419/` and `Benchmark/output/`  
**Generated PowerPoint File**: [`Presentation/Neural_Image_Inpainting_Snapdragon_8_Elite_Evaluation.pptx`](file:///home/aaravind/develop/models/Image_Inpainting/Presentation/Neural_Image_Inpainting_Snapdragon_8_Elite_Evaluation.pptx)

---

## Executive Summary: Latest Empirical Benchmark Overview

### 1. Fresh Measured On-Device Benchmark (`Benchmark/output/fresh_benchmark/20261008_035419/BENCHMARK_RESULTS.md`)
*Measured on target hardware device `8f27557f` on 2026-10-08 across 102 benchmark samples (`Benchmark/input_102`):*

| Model | Runtime Core | Per Image (Steady) | Throughput (FPS) | Cold Launch | Full PSNR (dB) | Hole PSNR (dB) | SSIM | LPIPS (VGG) | Rep 1→3 Throttling Slowdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIGAN** | **Hexagon NPU** | **50.2 ms** | **19.91 img/s** | **0.32 s** | **23.01 dB** | **17.30 dB** | **0.7112** | **0.2412** | **1.1 %** (Safe) |
| **LaMa-Dilated** | **Hexagon NPU** | **101.4 ms** | **9.87 img/s** | **7.87 s** (0.47s*) | **23.54 dB** | **18.18 dB** | **0.7381** | **0.2355** | **1.7 %** (Safe) |
| **AOT-GAN** | **Hexagon NPU** | **163.9 ms** | **6.10 img/s** | **7.93 s** (0.57s*) | **22.96 dB** | **16.79 dB** | **0.7301** | **0.2380** | **-0.1 %** (Zero) |
| **MIGAN** | Adreno GPU | 239.6 ms | 4.17 img/s | 1.19 s | 23.42 dB | 18.12 dB | 0.7270 | 0.2270 | 0.2 % |
| **LaMa-Dilated** | Adreno GPU | 548.9 ms | 1.82 img/s | 1.26 s | 23.53 dB | 18.17 dB | 0.7380 | 0.2355 | **8.2 %** (Throttled, 90.3°C) |
| **AOT-GAN** | Adreno GPU | 640.2 ms | 1.56 img/s | 1.25 s | 22.95 dB | 16.73 dB | 0.7300 | 0.2380 | **13.5 %** (Severe, 92.2°C) |
| **Stable Diff 1.5** | **Hexagon NPU** | **13.3 s** (DPM) | **0.08 img/s** | **25.2 s wall** | **18.87 dB** | **13.46 dB** | **0.6048** | **0.2815** | Thermal peak (+30°C in Euler) |

*\*Cold launch slashed to 0.47s (LaMa) and 0.57s (AOT-GAN) via precompiled SNPE HTP init cache (`--init-cache`).*

### 2. Hexagon NPU vs. Adreno GPU Acceleration & Thermal Throttling Audit
* **Speedup Ratios**:
  * MIGAN: $50.2\text{ ms}$ (NPU) vs. $239.6\text{ ms}$ (GPU) $\rightarrow$ **4.77x NPU speedup**
  * LaMa-Dilated: $101.4\text{ ms}$ (NPU) vs. $548.9\text{ ms}$ (GPU) $\rightarrow$ **5.41x NPU speedup**
  * AOT-GAN: $163.9\text{ ms}$ (NPU) vs. $640.2\text{ ms}$ (GPU) $\rightarrow$ **3.91x NPU speedup**
* **Thermal Throttling Divide**:
  * **Hexagon NPU**: Continuous 102-sample inference produces negligible batch slowdown ($-0.1\%$ to $1.7\%$). Junction temperatures remain strictly within safe operational limits ($44.2^\circ\text{C}$ to $60.7^\circ\text{C}$).
  * **Adreno GPU**: GPU execution triggers severe thermal throttling (**8.2%** slowdown on LaMa, **13.5%** on AOT-GAN). GPU junction temperatures hit critical peaks of **$90.3^\circ\text{C}$** and **$92.2^\circ\text{C}$**, proving that fanless mobile devices cannot sustain heavy shader-based inpainting.

### 3. Mask Severity Stratification (Latest Measured Data)
* **Tier 1 (Light: <15% mask area, n=68)**:
  * LaMa NPU: Hole PSNR $18.99\text{ dB}$ | SSIM $0.7456$ | LPIPS $0.2006$
  * MIGAN NPU: Hole PSNR $18.35\text{ dB}$ | SSIM $0.7343$ | LPIPS $0.2034$
  * AOT-GAN NPU: Hole PSNR $17.36\text{ dB}$ | SSIM $0.7427$ | LPIPS $0.2036$
* **Tier 2 (Medium: 15%-30% mask area, n=25)**:
  * LaMa NPU: Hole PSNR $16.28\text{ dB}$ | SSIM $0.7219$ | LPIPS $0.2877$
  * AOT-GAN NPU: Hole PSNR $15.82\text{ dB}$ | SSIM $0.7098$ | LPIPS $0.2889$
  * MIGAN NPU: Hole PSNR $15.15\text{ dB}$ | SSIM $0.6788$ | LPIPS $0.2947$
* **Tier 3 (Heavy: >30% mask area, n=9)**:
  * **LaMa NPU: Hole PSNR $17.27\text{ dB}$ | SSIM $0.7263$ | LPIPS $0.3537$** (Decisive structural leader)
  * MIGAN NPU: Hole PSNR $15.41\text{ dB}$ | SSIM $0.6265$ (drops $-0.107$) | LPIPS $0.3788$
  * AOT-GAN NPU: Hole PSNR $15.19\text{ dB}$ | SSIM $0.6917$ | LPIPS $0.3558$

---

## Detailed Slide-by-Slide Technical Specification

### Slide 1: Title Slide
* **Category / Tracker**: `EMBEDDED SYSTEMS WORKSHOP (ESW) · FINAL TECHNICAL EVALUATION`
* **Title**: Neural Image Inpainting on Qualcomm Snapdragon 8 Elite
* **Subtitle**: On-Device Evaluation, Hexagon HTP v79 NPU Acceleration & Contextual Bandit Routing
* **Visuals & Layout**:
  * Qualcomm Dark Slate theme (`#0B0F19`) with top Qualcomm Blue accent bar (`#3B82F6`).
  * Platform tags: Qualcomm Snapdragon 8 Elite Development Kit (SM8750P / APQ8750 / Platform `sun` / Device `8f27557f`), Hexagon HTP v79 NPU, Adreno 830 GPU, Android 15.
  * Institutional & Team metadata: Dr. Deepak Gangadharan (Supervisor), Vedant Tejas (Mentor/TA), Ashwin Aravind, Tarsh Singh, Raghavendra Navneet Nawander, Aryan Kulkarni.
* **Speaker Script**: Sets the project context, outlining the engineering achievements in porting 4 neural inpainting architectures natively to Qualcomm mobile hardware and evaluating them with live hardware telemetry.

---

### Slide 2: Team Members & System Engineering Ownership
* **Category / Tracker**: `PROJECT GOVERNANCE & RESPONSIBILITIES`
* **Title**: Team Members & System Engineering Ownership
* **Subtitle**: End-to-end module ownership spanning low-level CDSP drivers to mobile UI
* **Engineering Ownership Grid**:
  * **Ashwin Aravind**: On-Device FastRPC CDSP runtime architecture, isolated Android 15 library paths, native C++ SD runner (`sd_qidk_runner_inpaint`), zero-copy DMA-BUF memory buffers.
  * **Tarsh Singh**: ONNX/DLC graph optimization, eliminating data-dependent Range nodes in MI-GAN, Qualcomm AIMET W8A16 quantization, SNPE init-cache compilation (0.47s cold launch).
  * **Raghavendra Navneet Nawander**: Multi-modal vision router (`src/router.py`), sub-50ms CPU GrabCut auto-masking engine, Constrained Contextual Bandit (LinUCB) RL formulation.
  * **Aryan Kulkarni**: Native Android 15 QIDK application (`qidk-inpaint-app`), Gradio Web Studio (`src/app_gui.py`), monotonic clock benchmark suite, 1 Hz PMIC & thermal logging daemon.
* **Speaker Script**: Highlights individual systems engineering contributions across CDSP firmware, compiler modifications, auto-masking, and application deployment.

---

### Slide 3: The Problem Statement — Pitfalls of Cloud-Based Inpainting
* **Category / Tracker**: `THE EDGE IMPERATIVE & MOTIVATION`
* **Title**: The Problem Statement: Pitfalls of Cloud-Based Inpainting
* **Subtitle**: Why commercial cloud offloading (Google Magic Eraser, Adobe Firefly) fails on mobile devices
* **Visual Layout**:
  * 3 Threat Alert Cards:
    1. *Strict Network Reliance*: Zero offline functionality in airplane mode, rural areas, or subways. Multi-megabyte photo uploads consume mobile bandwidth.
    2. *Severe Latency Overhead*: 1.5s - 4.0s round-trip delay destroys interactive touch-to-erase editing.
    3. *Privacy & Compliance Risks*: Sending unencrypted personal photos (faces, documents, geotags) across public networks exposes users to data retention and regulatory liabilities (GDPR/HIPAA).
  * Pipeline Latency Comparison: Cloud offloading (~2,120 ms) vs. On-device Hexagon HTP v79 NPU (< 220 ms total active execution).

---

### Slide 4: Proposed Solution — Edge AI on Qualcomm Snapdragon 8 Elite
* **Category / Tracker**: `SYSTEM ARCHITECTURE & HARDWARE PLATFORM`
* **Title**: Proposed Solution: Edge AI on Qualcomm Snapdragon 8 Elite
* **Subtitle**: Hardware-accelerated neural inpainting executing natively on Hexagon HTP v79 silicon
* **Hardware & Tooling Specs**:
  * *Hexagon HTP v79 NPU*: Dual 128-byte vector engines optimized for INT8/INT16 matrix tensor operations at peak TOPS/Watt.
  * *FastRPC CDSP Subsystem*: Direct zero-copy communication via `/dev/fastrpc-cdsp` between Host CPU and CDSP.
  * *Adreno 830 GPU & Oryon CPU*: Secondary accelerators and host CPU for OpenCV GrabCut segmentation.
  * *Tooling Stack*: Qualcomm AI Hub (QAI Hub), Qualcomm AI Runtime SDK (QAIRT / QNN / SNPE v2.49), Qualcomm AIMET quantization toolkit.

---

### Slide 5: Evaluated AI Model Architectures Overview
* **Category / Tracker**: `NEURAL INPAINTING TAXONOMY`
* **Title**: Evaluated AI Model Architectures Overview
* **Subtitle**: Evaluating four distinct architectural paradigms spanning feed-forward GANs to latent diffusion
* **Taxonomy Table & Paradigms**:
  * *MIGAN*: Multi-scale depthwise separable GAN with gating ($1 \times 3 \times 512 \times 512$ NHWC inverted tensor).
  * *AOT-GAN*: Aggregated Contextual Transformations with stacked dilated bottlenecks ($r=1,2,4,8$).
  * *LaMa-Dilated*: Large Mask Inpainting with Fast Fourier Convolutions (FFC).
  * *Stable Diffusion 1.5*: 12-step DPM-Solver++ on 16-channel U-Net in latent space ($1 \times 4 \times 64 \times 64$).
  * *Trade-Off*: Feed-forward GANs (< 200 ms, ~1 J) vs. Iterative Latent Diffusion (> 13 s, 35-135 J).

---

### Slide 6: Model Deep Dive 1 — MI-GAN (Mobile Inpainting GAN)
* **Category / Tracker**: `MODEL DEEP DIVE · MOBILE EFFICIENCY CHAMPION`
* **Title**: MI-GAN: Multi-Scale Mobile Inpainting GAN
* **Subtitle**: Ultra-fast depthwise separable convolutions optimized for facial structure and mobile energy
* **Measured Stats**: **50.2 ms steady latency** (19.91 FPS) | **0.32 s cold launch** | **1.1% batch slowdown** | **0.62 J energy** | **+5.3°C NSP rise** (peak 44.2°C).
* **Compiler Bugs & Fixes**:
  * *Dynamic Range Op Elimination*: Original MIGAN used runtime `get_masked_bbox`, generating data-dependent ONNX `Range` nodes (`/Cast_4_output_0`). Qualcomm QNN converter aborted compilation. Re-authored with static 512×512 core.
  * *Boundary Transpose Elimination*: Declared native NHWC I/O layout directly inside ONNX, eliminating boundary `Transpose` nodes that caused DSP crashes.
  * *Mask Polarity Inversion*: Requires inverted mask polarity ($0.0 = \text{hole to fill}$, $1.0 = \text{keep background}$). Passing standard polarity produced blinding white block artifacts.

---

### Slide 7: Model Deep Dive 2 — AOT-GAN (Contextual Transformations)
* **Category / Tracker**: `MODEL DEEP DIVE · STRUCTURAL FIDELITY CHAMPION`
* **Title**: AOT-GAN: Aggregated Contextual Transformations GAN
* **Subtitle**: Stacked dilated convolution bottlenecks for fine-grained textures, repetitive patterns, and crisp edges
* **Measured Stats**: **163.9 ms steady latency** (6.10 FPS) | **28.34 dB Global PSNR** (200-pair) | **0.1058 LPIPS** | **-0.1% throttle slowdown** (zero NPU throttling).
* **Architecture & Hardware Profile**:
  * *AOT Block Splitting*: Splits intermediate features into 4 parallel sub-spaces with dilation rates $r \in \{1, 2, 4, 8\}$, simultaneously capturing fine textures and wide scene context.
  * *NPU vs. GPU Contrast*: On NPU, throttling is -0.1% with junction temps staying at 60.7°C. On GPU, AOT-GAN triggers a **13.5% throttling slowdown** with GPU junction temperatures reaching **92.2°C**!

---

### Slide 8: Model Deep Dive 3 — LaMa-Dilated (Large Mask Inpainting)
* **Category / Tracker**: `MODEL DEEP DIVE · LARGE VOID RESILIENCE CHAMPION`
* **Title**: LaMa-Dilated: Large Mask Inpainting with Fast Fourier Convolutions
* **Subtitle**: Global receptive fields in the frequency domain providing robust synthesis across massive corruptions
* **Measured Stats**: **101.4 ms steady latency** (9.87 FPS) | **0.47 s cold launch** (w/ init cache) | **17.27 dB Heavy Hole PSNR** | **5.41x NPU vs. GPU speedup**.
* **FFC Theory & Cold-Start Solution**:
  * *2D Real FFT Transformation*: Frequency-domain spectral convolutions provide an infinite receptive field covering the entire 512×512 canvas from the first layer.
  * *Cold-Start Solution*: Precompiled SNPE HTP init cache (`--init-cache`) reduced DSP initialization from 7.87s to 0.47s (17x speedup).
  * *Heavy Occlusion Dominance*: On heavy masks (>30% area), LaMa achieves 17.27 dB Hole PSNR, outperforming MIGAN (15.41 dB) and AOT-GAN (15.19 dB).

---

### Slide 9: Model Deep Dive 4 — Stable Diffusion 1.5 Inpainting on Hexagon NPU
* **Category / Tracker**: `MODEL DEEP DIVE · GENERATIVE DIFFUSION ON EDGE SILICON`
* **Title**: Stable Diffusion 1.5 Inpainting on Hexagon NPU (HTP v79)
* **Subtitle**: Full C++ native QIDK execution engine running 4 serialized QNN graphs on mobile silicon
* **Measured Stats**: **13.3 s DPM pipeline** | **16-Channel SIMD alignment** | **W8A16 AIMET quantization** | **+30.0°C Euler surge**.
* **Hardware Adaptations**:
  * *16-Channel Alignment*: Expanded 9-channel inpainting U-Net input to 16 channels (+7 zero-padded channels) to satisfy Hexagon SIMD 128-byte memory alignment.
  * *PermuteLayerNorm*: Replaced NCHW LayerNorm with native NHWC `PermuteLayerNorm` kernels.
  * *Scheduler Upgrade*: Transitioned from 20-step Euler RePaint (50.8s, 135 J) to 12-step DPM-Solver++ (13.3s pipeline, 25.2s wall-clock, 35 J).
  * *Empty Prompt Contract*: Text prompt must be empty `""` to prevent semantic hallucination into the hole.

---

### Slide 10: Benchmarking Methodology & Evaluation Protocol
* **Category / Tracker**: `EXPERIMENTAL METHODOLOGY & RIGOR`
* **Title**: Benchmarking Methodology & Measurement Protocol
* **Subtitle**: Standardized academic evaluation across 102 fresh on-device pairs and 200 master validation pairs
* **Protocol Details**:
  * *Datasets*: 102 standardized on-device samples (`Benchmark/input_102`) and 200-pair extended validation suite (Places365 + CelebA-HQ + NVIDIA PConv masks).
  * *Hardware Telemetry*: Monotonic clocks (`/proc/uptime`), continuous 1 Hz PMIC & thermal logging (`thermal_logger.sh`), and Qprof power profiling.
  * *Mandatory 25s Cooldown Barrier*: Guaranteed every model sweep started strictly at an idle SoC temperature $T \le 45^\circ\text{C}$.
  * *Multi-Fidelity Metrics*: Global PSNR, Hole-Only PSNR, SSIM, LPIPS (VGG-16), and stratified mask tier analysis.

---

### Slide 11: Benchmark Results 1 — Quality vs. Latency Pareto Frontier
* **Category / Tracker**: `EMPIRICAL BENCHMARK · MASTER AUDIT`
* **Title**: Benchmark Results 1: Quality vs. Latency Pareto Frontier
* **Subtitle**: Latest measured on-device benchmarks on Qualcomm Snapdragon 8 Elite (SM8750P / HTP v79)
* **Embedded Figure**: [`Benchmark/output/fresh_benchmark/20261008_035419/figures/02_latency_vs_quality_pareto.png`](file:///home/aaravind/develop/models/Image_Inpainting/Benchmark/output/fresh_benchmark/20261008_035419/figures/02_latency_vs_quality_pareto.png)
* **Pareto Frontier Points**:
  * *Point 1 (Mobile Interactivity)*: MIGAN (50.2 ms steady, 19.91 FPS, 0.32s cold start, 0.62 J energy).
  * *Point 2 (Overall Quality & Heavy Holes)*: LaMa-Dilated (101.4 ms steady, 9.87 FPS, 18.18 dB Hole PSNR, 0.7381 SSIM).
  * *Point 3 (Structural Textures)*: AOT-GAN (163.9 ms steady, 6.10 FPS, -0.1% throttling slowdown).
  * *Generative Outlier*: Stable Diffusion 1.5 (13.3s DPM pipeline, 35 J energy).

---

### Slide 12: Benchmark Results 2 — Global vs. Hole-Only Fidelity Gap
* **Category / Tracker**: `PERCEPTUAL METRIC ANALYSIS · METHODOLOGICAL INSIGHT`
* **Title**: Benchmark Results 2: The Global vs. Hole-Only Fidelity Gap
* **Subtitle**: Why standard global metrics inflate reported quality and how Hole-Only evaluation exposes true generative performance
* **Embedded Figure**: [`Benchmark/output/fresh_benchmark/20261008_035419/figures/01_metric_distributions_boxplots.png`](file:///home/aaravind/develop/models/Image_Inpainting/Benchmark/output/fresh_benchmark/20261008_035419/figures/01_metric_distributions_boxplots.png)
* **Inflation Gap Data**:
  * MIGAN: 23.01 dB Global vs. 17.30 dB Hole-Only $\rightarrow$ **+5.71 dB inflation**
  * LaMa-Dilated: 23.54 dB Global vs. 18.18 dB Hole-Only $\rightarrow$ **+5.36 dB inflation**
  * AOT-GAN: 22.96 dB Global vs. 16.79 dB Hole-Only $\rightarrow$ **+6.17 dB inflation**
  * Stable Diffusion: 18.87 dB Global vs. 13.46 dB Hole-Only $\rightarrow$ **+5.41 dB inflation**
* **Takeaway**: Hole-Only PSNR isolates genuine pixel synthesis inside the hole, demonstrating LaMa's generative leadership (18.18 dB). Outside pixels are preserved via alpha compositing: $I_{out} = (M \odot I_{model}) + ((1 - M) \odot I_{in})$.

---

### Slide 13: Benchmark Results 3 — Mask Stress Degradation Curves
* **Category / Tracker**: `ROBUSTNESS AUDIT · OCCLUSION SEVERITY STRESS TEST`
* **Title**: Benchmark Results 3: Mask Coverage Stress Stratification
* **Subtitle**: Measured on-device performance across Light (<15%), Medium (15-30%), and Heavy (>30%) occlusions
* **Embedded Figure**: [`Benchmark/output/fresh_benchmark/20261008_035419/figures/07_quality_by_mask_tier.png`](file:///home/aaravind/develop/models/Image_Inpainting/Benchmark/output/fresh_benchmark/20261008_035419/figures/07_quality_by_mask_tier.png)
* **Stratified Analysis**:
  * *Light (<15%, n=68)*: LaMa (18.99 dB Hole PSNR, 0.7456 SSIM), MIGAN (18.35 dB / 0.7343), AOT-GAN (17.36 dB / 0.7427).
  * *Medium (15-30%, n=25)*: LaMa (16.28 dB / 0.7219), AOT-GAN (15.82 dB / 0.7098), MIGAN (15.15 dB / 0.6788).
  * *Heavy (>30%, n=9)*: LaMa achieves 17.27 dB Hole PSNR and 0.7263 SSIM (SSIM drops by only 0.019), whereas MIGAN collapses to 15.41 dB and 0.6265 SSIM (SSIM drops by 0.107).

---

### Slide 14: Benchmark Results 4 — Thermal Saturation & Power Dynamics
* **Category / Tracker**: `EMBEDDED SYSTEMS TELEMETRY · THERMAL & POWER AUDIT`
* **Title**: Benchmark Results 4: Thermal Saturation & Power Dynamics
* **Subtitle**: Live 1 Hz PMIC and junction temperature logging revealing mobile thermal throttling limits
* **Embedded Figure**: [`Benchmark/output/fresh_benchmark/20261008_035419/figures/04_telemetry_latency_coldstart_thermals.png`](file:///home/aaravind/develop/models/Image_Inpainting/Benchmark/output/fresh_benchmark/20261008_035419/figures/04_telemetry_latency_coldstart_thermals.png)
* **Thermal Telemetry Audit**:
  * *NPU Thermal Stability*: MIGAN peaks at 44.2°C, LaMa at 59.2°C, and AOT-GAN at 60.7°C, well below the 72°C throttle limit with < 1.7% slowdown.
  * *GPU Thermal Throttling*: Running LaMa and AOT-GAN on GPU pushes junction temperatures to **90.3°C** and **92.2°C**, causing 8.2% and 13.5% throttling slowdowns.
  * *Stable Diffusion*: Pushes SoC to 74.9°C (+30.0°C rise) during 20 Euler steps.

---

### Slide 15: Benchmark Results 5 — NPU (Hexagon HTP) vs. Adreno GPU Acceleration
* **Category / Tracker**: `HARDWARE ACCELERATION COMPARISON`
* **Title**: Benchmark Results 5: Hexagon NPU vs. Adreno GPU Acceleration
* **Subtitle**: Measured on-device speedup and thermal throttling advantages of Hexagon HTP v79 over mobile GPU shaders
* **Embedded Figure**: [`Benchmark/output/presentation_figures/06_npu_vs_gpu_efficiency_and_thermal_throttling.png`](file:///home/aaravind/develop/models/Image_Inpainting/Benchmark/output/presentation_figures/06_npu_vs_gpu_efficiency_and_thermal_throttling.png)
* **Speedup & Throttling Metrics**:
  * MIGAN: 50.2 ms vs. 239.6 ms $\rightarrow$ **4.77x speedup** (1.1% vs 0.2% slowdown)
  * LaMa-Dilated: 101.4 ms vs. 548.9 ms $\rightarrow$ **5.41x speedup** (1.7% vs **8.2% slowdown**)
  * AOT-GAN: 163.9 ms vs. 640.2 ms $\rightarrow$ **3.91x speedup** (-0.1% vs **13.5% slowdown**)

---

### Slide 16: The Application — Unified Mobile & Web GUI Ecosystem
* **Category / Tracker**: `END-TO-END DEPLOYMENT · MOBILE & WEB APPLICATION`
* **Title**: The Application: Unified Mobile & Web GUI Ecosystem
* **Subtitle**: Dual-surface deployment integrating sub-50ms CPU GrabCut auto-masking and live NPU execution
* **Application Architecture**:
  * *Native Android 15 App*: Kotlin UI + C++ JNI bridge, tmpfs RAM preloading of DLC models.
  * *Gradio Web Studio*: Interactive brush canvas, 102 benchmark sample loader, live ADB telemetry HUD.
  * *Sub-50ms CPU GrabCut Auto-Masking*: Downsamples ROI to $\le 112\text{ px}$, runs OpenCV GrabCut with Gaussian Mixture Models, upsamples and morphological closing in **24.60 ms average latency** (Min: 22.88 ms, Max: 40.21 ms).

---

### Slide 17: Current Heuristic Auto-Router Architecture
* **Category / Tracker**: `INTELLIGENT MODEL ORCHESTRATION`
* **Title**: Current Heuristic Auto-Router Architecture
* **Subtitle**: Multi-modal feature extraction and audited decision tree routing requests to optimal NPU models
* **Decision Rules**:
  * *Faces detected & area $\le 25\%$* $\rightarrow$ **MIGAN** (facial geometry, 50.2 ms steady compute, lowest energy).
  * *Area $> 25\%$ OR fragments $\ge 3$* $\rightarrow$ **LaMa-Dilated** (global Fourier context, 17.27 dB heavy Hole PSNR).
  * *Edge density $> 0.08$ OR texture $> 500$ OR OCR text* $\rightarrow$ **AOT-GAN** (sharp structural lines).
  * *Smooth baseline default* $\rightarrow$ **MIGAN** (19.91 FPS throughput, minimal thermals).

---

### Slide 18: What Next — Stable Diffusion Optimizations
* **Category / Tracker**: `FUTURE ROADMAP · ON-DEVICE DIFFUSION ACCELERATION`
* **Title**: What Next: Stable Diffusion Optimizations on Edge Silicon
* **Subtitle**: Engineering pathways to bring latent diffusion inference from 13.3s down to sub-3 second interactivity
* **Roadmap Pathways**:
  1. *INT4 Weight Quantization*: Compress 842 MB U-Net container to $< 450$ MB, relieving DRAM memory bus bandwidth pressure.
  2. *Consistency Distillation (LCM / SD-Turbo)*: Reduce denoising from 12 steps down to 2-4 passes, targeting sub-3s edge diffusion.
  3. *HTP Init & Empty-Prompt Token Caching*: Pre-serialize KV attention caches for empty prompt tokens and compile SNPE init caches to eliminate cold-start pauses.

---

### Slide 19: What Next — Adaptive Routing via Reinforcement Learning
* **Category / Tracker**: `FUTURE ROADMAP · ADAPTIVE EDGE INTELLIGENCE`
* **Title**: What Next: Adaptive Routing via Reinforcement Learning
* **Subtitle**: Overcoming the limitations of static rule-based heuristics through online reinforcement learning
* **Paradigm Shift**:
  * Static cutoffs ($\alpha_{mask} > 0.25$, $\rho_{edge} > 0.08$) create sharp decision cliffs and are blind to battery percentage ($B_{level}$) and current SoC temperature ($T_{SoC}$).
  * Formulate model dispatch as an Online Constrained Contextual Bandit, dynamically balancing perceptual quality against real-time battery and thermal constraints.

---

### Slide 20: Mathematical Formulation — Constrained Contextual Bandits
* **Category / Tracker**: `REINFORCEMENT LEARNING MATHEMATICS`
* **Title**: Mathematical Formulation: Constrained Contextual Bandits
* **Subtitle**: Rigorous formulation of context space, action space, and dual ridge regression estimators
* **Mathematical Specification**:
  * Context Vector: $x_t = [\alpha_{mask}, \rho_{edge}, \sigma^2, N_{faces}, B_{level}, T_{SoC}]^\top \in \mathbb{R}^6$.
  * Action Space: $\mathcal{A} = \{\text{MIGAN}, \text{AOT-GAN}, \text{LaMa}, \text{SD 1.5}\}$.
  * Dual Ridge Estimators per Arm:
    * Quality: $\hat{Q}_a(x) = \theta_{Q,a}^\top x$
    * Latency: $\hat{L}_a(x) = \theta_{L,a}^\top x$
    * Thermal Rise: $\Delta\hat{T}_a(x) = \theta_{T,a}^\top x$
    * Closed-Form Update: $\theta_a = (D_a^\top D_a + \lambda I_d)^{-1} D_a^\top y_a$.

---

### Slide 21: RL Solution — Linear Upper Confidence Bound (LinUCB)
* **Category / Tracker**: `REINFORCEMENT LEARNING ALGORITHM`
* **Title**: RL Solution: Linear Upper Confidence Bound (LinUCB)
* **Subtitle**: Constrained multi-objective optimization balancing exploration, exploitation, and thermal safety
* **Objective & Selection Rule**:
  * Constrained Reward: $R(x, a) = \hat{Q}_a(x) - \lambda \hat{L}_a(x) - \gamma \hat{C}_a(x)$ subject to $\hat{L}_a(x) \le L_{max}$ and $T_{SoC} + \Delta\hat{T}_a(x) \le 72^\circ\text{C}$.
  * LinUCB Rule: $a^* = \arg\max_{a \in \mathcal{A}_{feasible}} \left( \hat{R}_a(x_t) + \alpha \sqrt{x_t^\top A_a^{-1} x_t} \right)$.
  * Behavior: Automatically falls back to cool-running MIGAN (50.2 ms, 44.2°C) when battery $< 20\%$ or device temperature $> 60^\circ\text{C}$; unlocks AOT-GAN or Stable Diffusion only when thermal headroom permits.

---

### Slide 22: Conclusion & Summary
* **Category / Tracker**: `PROJECT RETROSPECTIVE & SUMMARY`
* **Title**: Conclusion & Major Engineering Milestones
* **Subtitle**: Summary of accomplishments across hardware compilation, empirical benchmarks, and edge intelligence
* **Milestones**:
  * 4 Neural Inpainting Models compiled and running natively on Snapdragon 8 Elite Hexagon NPU.
  * Verified 3.91x - 5.41x speedups on Hexagon HTP v79 over Adreno GPU shaders without GPU thermal saturation (where GPU reached 92.2°C).
  * Real-time sub-50ms GrabCut auto-masking (24.60 ms).
  * Fully deployed on Android 15 and Gradio Web Studio.
  * Comprehensive LinUCB Contextual Bandit roadmap for thermally safe, battery-aware edge AI.

---

## Deliverables & Associated Repository Paths
* **Generated Presentation PPTX**: [`Presentation/Neural_Image_Inpainting_Snapdragon_8_Elite_Evaluation.pptx`](file:///home/aaravind/develop/models/Image_Inpainting/Presentation/Neural_Image_Inpainting_Snapdragon_8_Elite_Evaluation.pptx)
* **Slide Deck Markdown Reference**: [`Presentation/FINAL_ACADEMIC_EVALUATION_SLIDE_DECK.md`](file:///home/aaravind/develop/models/Image_Inpainting/Presentation/FINAL_ACADEMIC_EVALUATION_SLIDE_DECK.md)
* **Fresh On-Device Benchmark Report**: [`Benchmark/output/fresh_benchmark/20261008_035419/BENCHMARK_RESULTS.md`](file:///home/aaravind/develop/models/Image_Inpainting/Benchmark/output/fresh_benchmark/20261008_035419/BENCHMARK_RESULTS.md)
* **Fresh Benchmark Summary JSON**: [`Benchmark/output/fresh_benchmark/20261008_035419/summary.json`](file:///home/aaravind/develop/models/Image_Inpainting/Benchmark/output/fresh_benchmark/20261008_035419/summary.json)
* **Automated Deck Generator**: [`scripts/generate_evaluation_deck.py`](file:///home/aaravind/develop/models/Image_Inpainting/scripts/generate_evaluation_deck.py)
