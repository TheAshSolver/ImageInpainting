# Neural Image Inpainting on Qualcomm Snapdragon 8 Elite (Hexagon HTP v79)
## Comprehensive Technical Documentation, Architecture Audit & Setup Guide

**Target Hardware**: Qualcomm Snapdragon 8 Elite Development Kit (SM8750P / Platform `sun`)  
**Acceleration Engine**: Qualcomm Hexagon NPU (HTP v79 Architecture) via FastRPC & QNN/SNPE Runtime  
**Project Module**: Embedded Systems Workshop — Image Inpainting Evaluation  

---

## 1. Problem Addressed

Image inpainting aims to synthesize visually plausible and semantically coherent pixels within corrupted, occluded, or missing regions of an image. Deploying state-of-the-art neural inpainting models to battery-powered mobile and edge platforms presents several critical engineering challenges:

1. **Computational Complexity vs. Mobile Thermal Envelopes**:
   Modern generative architectures (such as Latent Diffusion Models) demand tens of billions of FLOPs across iterative denoising passes, resulting in extreme battery depletion and rapid thermal saturation on mobile SoCs.
2. **Feed-Forward GANs vs. Iterative Diffusion Models**:
   While feed-forward convolutional GANs (e.g., MIGAN, AOT-GAN, LaMa) execute in a single deterministic pass ($<400\,\text{ms}$), generative diffusion models (Stable Diffusion 1.5 RePaint) perform multi-step stochastic sampling ($\sim 50\,\text{s}$). Quantifying the precise trade-off between **sub-second edge interactivity** and **generative hallucination capacity** is essential for production edge engineering.
3. **NPU Hardware Acceleration Constraints**:
   Qualcomm Hexagon Tensor Processors (HTPs) require static tensor compilation, fixed buffer dimensions ($1 \times 3 \times 512 \times 512$ float32 / uint8), int8/fp16 weight quantization, and strict FastRPC user-space-to-DSP daemon library paths. Raw unquantized dynamic shapes are rejected by the HTP runtime.
4. **Intelligent Model Routing**:
   Different inpainting tasks exhibit starkly distinct characteristics: facial portraits require structural symmetry, large voids require global receptive fields, and fine textures require high-frequency synthesis. A heuristic decision router dynamically dispatches input images to the optimal architecture based on real-time computer vision feature extraction.

This project delivers an end-to-end evaluation suite, an interactive Web UI, and a native Android QIDK application running on the Snapdragon 8 Elite Hexagon NPU under controlled thermal conditions.

---

## 2. Approach Taken

### A. Accelerated Edge Runtimes
* **SNPE / QNN DLC Containers**: Models were compiled to Qualcomm Deep Learning Containers (`.dlc`) targeted at the Hexagon v79 HTP architecture.
* **FastRPC User-Space Runtime**: Executed directly on the Hexagon NPU using the Qualcomm DSP daemon bridge (`/dev/fastrpc-cdsp`), bypassing host CPU emulation.
* **Native C++ Denoising Loop**: Stable Diffusion RePaint was executed via a compiled native C++ runner (`sd_qidk_runner_encoder`) executing 20-step Euler latent sampling on HTP v79.

### B. Standardized Benchmarks
* **200-Pair Stratified Academic Benchmark (`Benchmark/input/`)**:
  - Ingested 200 high-resolution images ($512 \times 512$) paired with irregular brush masks across varied occlusion ratios ($5\% - 47\%$).
  - Partitioned into Tier 1 (Light: $1\% - 15\%$), Tier 2 (Medium: $15\% - 25\%$), and Tier 3 (Heavy: $> 25\%$).
* **102-Sample Standardized Benchmark (`Benchmark/input_102/`)**:
  - Standardized dataset with 102 verified samples consisting of 512x512 RGB images, 512x512 binary uint8 masks ($0 = \text{keep}, 255 = \text{hole}$), and bicubic downsampled ground-truth images.

### C. Thermal Isolation & PMIC Hardware Telemetry
* **Sensor Filtering**: Real-time sampling of silicon compute thermal zones (`cpu*`, `cpuss*`, `gpuss*`, `nsphvx*`, `nsphmx*`, `ddr*`, `aoss*`), filtering out battery/chassis passive zones.
* **Accurate Active Power Formula**: Ingested battery sysfs and Qualcomm PMIC current rails (`in_current_pmih010x_ichg_fb_input`), calculating active wattage:
  $$\text{Power (Watts)} = \left(\frac{|\text{current\_uA}|}{10^6}\right) \times \left(\frac{\text{voltage\_uV}}{10^6}\right)$$
* **Mandatory Thermal Cooldown Barrier**: Enforced an automated cooldown phase between every model run, requiring peak SoC temperatures to stabilize below $\le 45.0^\circ\text{C}$ (with a minimum 25-second delay) to eliminate residual thermal soaking contamination.

### D. Decision Router Classifier & Strict Alpha Compositing
* **Decision Tree Heuristics**: Lightweight feature extractors (MediaPipe FaceMesh, Laplacian variance, Sobel edge density, Tesseract OCR) classify incoming images and route to MIGAN, LaMa, or AOT-GAN in $<30\,\text{ms}$.
* **Isolated Input Polarity**: Separate model-specific tensor inversion from post-processing compositing to avoid mask shadowing.
* **Strict Alpha Blend**: Background is preserved 1:1 using Gaussian feathering:
  $$\text{Final} = \text{Original} \times (1.0 - \text{Weight}) + \text{ModelOutput} \times \text{Weight}$$

---

## 3. Implementation Details

### A. Repository Architecture
```text
ImageInpainting/
├── Benchmark/
│   ├── input/                       # 200 Ground truth images & irregular masks (1.png .. 200.png)
│   ├── input_102/                   # 102 Standardized benchmark dataset (image/, mask/, ground_truth/)
│   └── output/
│       ├── migan/results/           # MIGAN inpainted PNGs
│       ├── aotgan/results/          # AOT-GAN inpainted PNGs
│       ├── lama/results/            # LaMa inpainted PNGs
│       ├── sd/results/              # Stable Diffusion RePaint PNGs
│       ├── figures/                 # 300 DPI publication plots and qualitative montages
│       ├── all_pairs_detailed_metrics.csv
│       ├── master_metrics_summary.csv
│       ├── FINAL_EVALUATION_REPORT.md
│       └── DEEP_DIVE_STATISTICAL_REPORT.md
├── models/
│   ├── Migan/migan_htp_v79.dlc      # Quantized Hexagon v79 container
│   ├── AOT-GAN/aotgan.dlc           # AOT-GAN Hexagon container
│   └── LamaDilated/lama_dilated.dlc # LaMa Dilated Hexagon container
├── qidk-inpaint-app/                # Standalone Native Android (Kotlin) App for QIDK
│   ├── app/src/main/
│   │   ├── AndroidManifest.xml      # Includes launcher intent-filter & permissions
│   │   ├── java/com/qualcomm/qidk/inpaint/
│   │   │   ├── MainActivity.kt      # Main controller & UI event handling
│   │   │   ├── inference/           # OnDeviceProcessDriver (direct FastRPC process launcher)
│   │   │   ├── router/              # OnDeviceRouter (on-device heuristic routing engine)
│   │   │   ├── segmentation/        # GrabCutSegmenter (<50ms silhouette extraction)
│   │   │   └── ui/                  # InpaintCanvasView (Two-Tap Bounding Box & HUD)
│   │   └── res/drawable/            # Vector launcher icons (ic_launcher, ic_launcher_round)
│   └── build.gradle.kts
├── src/
│   ├── app_gui.py                   # Interactive Gradio Web UI with on-device NPU telemetry
│   ├── auto_masking.py              # GrabCut auto-masking, brush stroke parser & alpha blend
│   └── router.py                    # Decision router engine & live SNPE/SD execution harness
├── scripts/
│   ├── master_benchmark.py          # Master single-command evaluation & telemetry harness
│   ├── fix_and_regenerate_grids.py  # Generates 300 DPI aligned dual montages (Grids 1 & 2)
│   ├── plot_scout_dashboard.py      # Generates 4-panel scout executive dashboard
│   ├── deep_dive_analysis.py        # Statistical modeling, regressions & distribution plots
│   ├── prep_102_benchmark.py        # 102-sample benchmark standardization script
│   └── run_sd_benchmark.py          # Native batch runner for Stable Diffusion RePaint
├── ROUTER_AUDIT_AND_PIPELINE_REPORT.md  # Detailed heuristic audit & pipeline analysis
├── SETUP.md                         # Authoritative documentation (this file)
└── README.md                        # Project landing page
```

### B. Evaluated Model Architectures & Mask Polarities
| Model | Binary / Container | Input Dimension | Mask Polarity | Architectural Characteristics |
| :--- | :---: | :---: | :---: | :--- |
| **MIGAN** | `migan_htp_v79.dlc` | $1 \times 3 \times 512 \times 512$ | **Inverted** ($0 = \text{hole}, 1 = \text{keep}$) | Multi-scale depthwise separable convolutions; facial optimization |
| **AOT-GAN** | `aotgan.dlc` | $1 \times 3 \times 512 \times 512$ | **Standard** ($1 = \text{hole}, 0 = \text{keep}$) | Aggregated Contextual Transformations; stacked dilated bottlenecks |
| **LaMa** | `lama_dilated.dlc` | $1 \times 3 \times 512 \times 512$ | **Standard** ($1 = \text{hole}, 0 = \text{keep}$) | Fast Fourier Transform (FFT) convolutions; global receptive field |
| **Stable Diffusion** | `sd_qidk_runner_encoder` | $1 \times 4 \times 64 \times 64$ (Latent) | **Standard** ($1 = \text{hole}, 0 = \text{keep}$) | 20-step Euler latent diffusion; text conditioning + VAE encoding |

### C. Decision Router Decision Tree

```mermaid
graph TD
    Input[Input 512x512 RGB Image & Mask] --> DetectFaces{Face Detected?<br/>MediaPipe FaceMesh}
    DetectFaces -- Yes (Faces >= 1 & Area <= 25%) --> MIGAN1[Route to MIGAN<br/>Specialized facial GAN, 216ms, 0.62J]
    DetectFaces -- No --> CheckVoid{Large Void or Fragmented?<br/>Area > 25% or Fragments >= 3}
    CheckVoid -- Yes --> LAMA[Route to LaMa Dilated<br/>Fast Fourier Convolutions, Global RF, 321ms]
    CheckVoid -- No --> CheckTexture{High Texture / OCR?<br/>Edge Dens > 0.08 or LapVar > 500 or Text}
    CheckTexture -- Yes --> AOT[Route to AOT-GAN<br/>Contextual Transformations, 389ms, High SSIM]
    CheckTexture -- No --> MIGAN2[Route to MIGAN<br/>Optimal power efficiency, 216ms]
```

### D. Metric Evaluation Suite
1. **Global PSNR (dB)**: Overall peak signal-to-noise ratio across all pixels.
2. **Hole-Only PSNR (dB)**: Mean squared error strictly computed on missing/corrupted pixels ($M \ge 128$), exposing true hallucination quality without background inflation:
   $$\text{MSE}_{\text{hole}} = \frac{1}{\sum M} \sum_{i,j,c} (GT_{i,j,c} - Pred_{i,j,c})^2 \cdot M_{i,j}$$
3. **SSIM**: Structural Similarity Index measuring luminance, contrast, and structural preservation.
4. **LPIPS (VGG)**: Perceptual feature distance using deep features from a pretrained VGG network.
5. **Energy per Sample (Joules)**: $\text{Joules} = \text{Average Power (Watts)} \times \text{Latency (Seconds)}$.
6. **Energy Delay Product (EDP)**: $\text{EDP} = \text{Joules} \times \text{Latency}$ ($\text{J}\cdot\text{s}$).

---

## 4. Setup and Execution Steps

### Step 1: Host Prerequisites & Environment
Ensure Python 3.10+ is installed on your Linux host.

```bash
# Clone the repository
git clone https://github.com/TheAshSolver/ImageInpainting.git
cd ImageInpainting

# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install \
    torch torchvision \
    numpy pandas scipy \
    matplotlib pillow scikit-image lpips \
    gradio opencv-python mediapipe pytesseract
```

### Step 2: Interactive Inpainting Web GUI (`src/app_gui.py`)
The interactive Gradio prototype allows live drawing, GrabCut bounding-box segmentation, automated heuristic routing, and on-device Hexagon NPU inference with full hardware telemetry.

```bash
# 1. Run automated headless test suite (verifies NPU execution, SD RePaint, and stroke extraction)
python3 src/app_gui.py --test

# 2. Launch the Web GUI server
python3 src/app_gui.py --host 0.0.0.0 --port 7860
```

To access the interface directly on the QIDK board's display:
```bash
# Reverse port 7860 to the connected QIDK board
adb reverse tcp:7860 tcp:7860

# Launch Chrome on the device screen
adb shell am start -a android.intent.action.VIEW -d "http://localhost:7860"
```

### Step 3: Standalone Native QIDK Android APK (`qidk-inpaint-app/`)
The native Android APK provides a standalone on-device application running directly on the Snapdragon 8 Elite hardware without a tethered PC.

**Key Features:**
* **App Drawer Launcher**: Contains launcher intent-filters and vector launcher icons (`ic_launcher`, `ic_launcher_round`).
* **Two-Tap Box Selection**: Tap 1 places a cyan crosshair corner handle; Tap 2 places the opposite corner and draws an interactive box with 4 corner handles, bypassing touchscreen drag jitter.
* **On-Device GrabCut**: Extracts foreground silhouette in $<50\,\text{ms}$ ($41\,\text{ms}$ recorded).
* **On-Device Process Driver**: Executes Qualcomm FastRPC / `snpe-net-run` binaries directly on the device.

**Building & Installing:**
```bash
cd qidk-inpaint-app

# Build debug APK
./gradlew assembleDebug

# Install on Snapdragon 8 Elite board
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Launch directly via ADB (or tap icon in App Drawer)
adb shell am start -n com.qualcomm.qidk.inpaint/.MainActivity
```

### Step 4: Reproduce Visual Dashboards & Qualitative Montages (Offline)
All benchmark evaluation logs, raw metrics, and output images are already committed and available on disk. To regenerate the publication figures without a connected hardware device:

```bash
# 1. Generate the 4-panel Executive Scout Dashboard (scout_report_dashboard.png)
HEADLESS=1 python3 scripts/plot_scout_dashboard.py

# 2. Generate the Fixed Dual Qualitative Montages (05_gan_domain_stress_6grid.jpg & 06_sd_vs_gans_5grid.jpg)
python3 scripts/fix_and_regenerate_grids.py

# 3. Generate the Statistical Distributions and Regression Plots
python3 scripts/deep_dive_analysis.py
```

### Step 5: Running Full On-Device Benchmark Harness
To run the automated benchmark across an attached Qualcomm Snapdragon 8 Elite device:

```bash
# 1. Verify ADB connection
adb devices
# Expected output: 8f27557f    device

# 2. Execute full automated pipeline with mandatory thermal cooldown barriers:
python3 scripts/master_benchmark.py --models migan aotgan lama sd --sd_samples 5
```

### Step 6: Starting Qprof Hardware Profiler
To profile hardware performance using Qualcomm Qprof:

```bash
# Forward Qprof gRPC port
adb forward tcp:62472 tcp:62472

# Launch gRPC monitor on device
adb shell
export QMONITOR_BACKEND_LIB_PATH=/vendor/qprof/backends
export LD_LIBRARY_PATH=/apex/com.android.i18n/lib64:/apex/com.android.runtime/lib64:/apex/com.android.art/lib64:/system/lib64:/vendor/lib64:/vendor/qprof/libs:$LD_LIBRARY_PATH
/vendor/bin/qmonitor-grpc-server -n 127.0.0.1 -p 62472
```

---

## 5. Assumptions and Constraints

1. **Static Tensor Dimensions**:
   The Hexagon v79 HTP compiler requires static input buffer shapes ($1 \times 3 \times 512 \times 512$). Dynamic input resolutions require offline graph recompilation into distinct DLC containers.
2. **$O(1)$ Feed-Forward Complexity**:
   For feed-forward architectures (MIGAN, AOT-GAN, LaMa), runtime latency and active power draw are **$O(1)$ constant** regardless of mask shape, size, or complexity. The full spatial grid is computed in a single tensor pass.
3. **VTCM vs. DRAM Streaming Bottlenecks**:
   - MIGAN, AOT-GAN, and LaMa fit comfortably within the Hexagon NPU's Vector Tightly-Coupled Memory (VTCM), achieving sustained throughput with minimal DRAM paging.
   - Stable Diffusion 1.5 RePaint's $860\text{M}$-parameter UNet exceeds VTCM capacity, requiring continuous weight streaming over the LPDDR5X bus across 20 Euler sampling steps, accounting for its $50.93\text{s}$ latency.
4. **Android 15 Linker Restrictions**:
   Do **NOT** include `/system/lib64` or `/vendor/lib64` in the device's `LD_LIBRARY_PATH`. Doing so causes a symbol collision in `libbinder_ndk.so` under Android 15 Bionic libc. Use only the isolated QNN runtime directory `/data/local/tmp/lama/lib:/data/local/tmp/sd_runtime`.

---

## 6. Results and Outcomes

### A. Master Performance, Energy & Quality Comparison

| Model | Acceleration Hardware | Inference Latency | Active Power | Energy / Image | Energy Delay Product (EDP) | Peak SoC Temp | Thermal Rise ($\Delta T$) | Global PSNR ↑ | Hole-Only PSNR ↑ | SSIM ↑ | LPIPS (VGG) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIGAN** | **Hexagon NPU (HTP v79)** | **$0.216\,\text{s}$** | $2.86\,\text{W}$ | **$0.62\,\text{J}$** | **$0.133\,\text{J}\cdot\text{s}$** | $48.4^\circ\text{C}$ | **$+9.6^\circ\text{C}$** | $27.17\,\text{dB}$ | $19.65\,\text{dB}$ | $0.9028$ | $0.1245$ |
| **LaMa Dilated** | **Hexagon NPU (HTP v79)** | **$0.321\,\text{s}$** | $3.10\,\text{W}$ | **$0.99\,\text{J}$** | **$0.319\,\text{J}\cdot\text{s}$** | $70.3^\circ\text{C}$ | **$+24.6^\circ\text{C}$** | $28.22\,\text{dB}$ | $20.73\,\text{dB}$ | $0.9193$ | $0.1195$ |
| **AOT-GAN** | **Hexagon NPU (HTP v79)** | **$0.389\,\text{s}$** | $3.34\,\text{W}$ | **$1.30\,\text{J}$** | **$0.505\,\text{J}\cdot\text{s}$** | $68.0^\circ\text{C}$ | **$+25.4^\circ\text{C}$** | **$28.34\,\text{dB}$** | **$20.86\,\text{dB}$** | **$0.9214$** | **$0.1058$** |
| **Stable Diffusion** | **Hexagon NPU (HTP v79)** | **$50.934\,\text{s}$** | $2.65\,\text{W}$ | **$134.97\,\text{J}$** | **$6874.8\,\text{J}\cdot\text{s}$** | $74.9^\circ\text{C}$ | **$+30.0^\circ\text{C}$** | $10.13\,\text{dB}$ | $9.66\,\text{dB}$ | $0.3183$ | $0.7216$ |

### B. Occlusion Stress Degradation Summary ($N=200$)
* **Tier 1 (Light: $1\% - 15\%$ area)**: All models retain high structural integrity ($\text{SSIM} > 0.94$, $\text{LPIPS} \le 0.078$).
* **Tier 2 (Medium: $15\% - 25\%$ area)**: AOT-GAN and LaMa maintain sharp edge continuation ($\text{Hole PSNR} \approx 20.8\,\text{dB}$).
* **Tier 3 (Heavy: $>25\%$ area)**: Global PSNR drops by $\sim 10.9\,\text{dB}$. LaMa's Fourier convolutions demonstrate the greatest structural stability under wide missing regions ($14.70\,\text{dB}$ Hole PSNR).

### C. Key Deliverables
* **Executive Scout Dashboard**: [`Benchmark/output/figures/scout_report_dashboard.png`](Benchmark/output/figures/scout_report_dashboard.png)
* **GAN Domain Stress 6-Grid**: [`Benchmark/output/figures/05_gan_domain_stress_6grid.jpg`](Benchmark/output/figures/05_gan_domain_stress_6grid.jpg)
* **SD vs GANs 5-Grid**: [`Benchmark/output/figures/06_sd_vs_gans_5grid.jpg`](Benchmark/output/figures/06_sd_vs_gans_5grid.jpg)
* **Web UI Verified Test Image**: [`Benchmark/output/figures/webui_fixed_test.png`](Benchmark/output/figures/webui_fixed_test.png)
* **Router Audit Report**: [`ROUTER_AUDIT_AND_PIPELINE_REPORT.md`](ROUTER_AUDIT_AND_PIPELINE_REPORT.md)
* **Statistical Modeling Report**: [`Benchmark/output/DEEP_DIVE_STATISTICAL_REPORT.md`](Benchmark/output/DEEP_DIVE_STATISTICAL_REPORT.md)
* **Full Benchmark Markdown Report**: [`Benchmark/output/FINAL_EVALUATION_REPORT.md`](Benchmark/output/FINAL_EVALUATION_REPORT.md)

---

## 7. References & Resources

1. **Qualcomm Neural Processing SDK**:
   Qualcomm Technologies, Inc., *QNN & SNPE Architecture and Development Guide*, 2024.
2. **FastRPC Framework**:
   Qualcomm Developer Network, *Hexagon DSP Architecture and FastRPC Architecture Guide*.
3. **LaMa (Large Mask Inpainting)**:
   R. Suvorov, E. Logacheva, A. Mashikhin, A. Remizova, A. Ashukha, A. Silvestrov, N. Kong, H. Goka, P. Park, V. Lempitsky, *"Resolution-robust Large Mask Inpainting with Fourier Convolutions"*, WACV 2022.
4. **AOT-GAN (Aggregated Contextual Transformations)**:
   Y. Zeng, J. Lin, J. Zhang, H. Chao, Q. Tian, *"Aggregated Contextual Transformations for High-Resolution Image Inpainting"*, ICCV 2021.
5. **MIGAN (Mask-Independent Generative Adversarial Network)**:
   T. Wang, B. Brattoli, M. Ommer, *"MIGAN: Mask-Independent Generative Adversarial Network for Image Inpainting"*, ACM MM 2022.
6. **RePaint & Latent Diffusion**:
   A. Lugmayr, M. Danelljan, A. Romero, F. Yu, R. Timofte, L. Van Gool, *"RePaint: Inpainting using Denoising Diffusion Probabilistic Models"*, CVPR 2022.  
   R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, *"High-Resolution Image Synthesis with Latent Diffusion Models"*, CVPR 2022.
