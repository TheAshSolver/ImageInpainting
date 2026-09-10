# Neural Image Inpainting on Qualcomm Snapdragon 8 Elite (Hexagon HTP v79)

### Team & Platform Details
- **Team Name**: Black & White
- **Course / Module**: Embedded Systems Workshop
- **Hardware Platform**: Qualcomm Snapdragon 8 Elite Development Kit (SM8750P / `sun`)
- **Acceleration Engines**: Qualcomm Hexagon NPU (HTP v79 Architecture) & Adreno 830 GPU via FastRPC & QNN/SNPE

---

## Executive Overview

This repository contains the complete embedded benchmarking harness, hardware telemetry pipeline, decision router, and interactive applications for neural image inpainting on the **Qualcomm Snapdragon 8 Elite Mobile Platform**.

We benchmark and profile four distinct neural architectures on the **Hexagon HTP v79 NPU** and **Adreno 830 GPU**:
1. **MIGAN** (`models/Migan/migan_htp_v79.dlc`): Multi-scale depthwise separable GAN ($216\,\text{ms}$, $0.62\,\text{J}$ on NPU, $2.43\times$ speedup over GPU).
2. **LaMa Dilated** (`models/LamaDilated/lama_dilated.dlc`): Fast Fourier Convolutions (FFC) with infinite global receptive field ($321\,\text{ms}$, $0.99\,\text{J}$).
3. **AOT-GAN** (`models/AOT-GAN/aotgan.dlc`): Aggregated Contextual Transformations GAN ($390\,\text{ms}$, $1.30\,\text{J}$).
4. **Stable Diffusion 1.5 RePaint** (`StableDiffusion/`): 20-step stochastic Euler latent diffusion pipeline ($50.93\,\text{s}$, $134.97\,\text{J}$).

All models are evaluated on the standardized **102-sample academic benchmark dataset** across multiple corruption domains with continuous 1 Hz SoC thermal and PMIC telemetry.

---

## Quick Start: Unified Master CLI (`main.py`)

All primary workflows are accessible via the unified [`main.py`](main.py) CLI:

```bash
# Activate Python environment
source .venv/bin/activate

# 1. System Status: Display project banner & verify connected Snapdragon 8 Elite device
python main.py

# 2. Web GUI: Launch the interactive Gradio Web Canvas & Router
python main.py gui --port 7860

# 3. Decision Router: Analyze an image and recommend the optimal model
python main.py route -i Benchmark/input_102/image/001.png -m Benchmark/input_102/mask/001.png

# 4. Publication Visuals: Generate all 8 presentation figures at 300 DPI
python main.py visuals

# 5. On-Device Benchmark: Execute high-throughput decoupled batch sweep across NPU/GPU
python main.py benchmark --samples 102
```

---

## Repository Structure

```text
ImageInpainting/
├── main.py                               # Unified Master CLI entrypoint
├── SETUP.md                              # Comprehensive technical & setup guide
├── ROUTER_AUDIT_AND_PIPELINE_REPORT.md   # Feature dynamic ranges & router audit
├── README.md                             # Project landing page (this file)
├── ESW_Image_Inpainting_Progress.pptx    # Slide presentation deck
├── Image_Inpainting_QIDK_Progress.pdf    # Slide presentation PDF
├── src/                                  # Core Python modules
│   ├── router.py                         # Decision router engine & SNPE execution harness
│   ├── auto_masking.py                   # Sub-50ms GrabCut auto-masker & tensor utilities
│   └── app_gui.py                        # Gradio interactive web application
├── models/                               # Consolidated canonical model weights & tools
│   ├── AOT-GAN/                          # aotgan.dlc + I/O parsers
│   ├── LamaDilated/                      # lama_dilated.dlc + I/O parsers
│   ├── Migan/                            # migan DLCs, ONNX models, & generator scripts
│   └── dlc-info.txt                      # Detailed layer & tensor quantization specs
├── scripts/                              # Active benchmarking & evaluation scripts
│   ├── run_two_phase_batch_benchmark.py  # Decoupled high-throughput batch runner
│   ├── generate_presentation_visuals.py  # 300-DPI publication visual generator
│   ├── prep_102_benchmark.py             # 102-sample preprocessor & tensor builder
│   ├── profile_granular_trace.py         # Sub-process cold-start diagnostic tracer
│   ├── thermal_logger.sh                 # 1 Hz SoC telemetry background daemon
│   ├── evaluation_suite.py               # PSNR / SSIM / LPIPS evaluation suite
│   ├── evaluation_torchmetrics.py        # Torchmetrics-based batch evaluator
│   ├── legacy_adb_steps/                 # Step-by-step ADB execution scripts (01-04)
│   └── archive/                          # Historical experiment scripts & tools
├── Benchmark/
│   ├── input_102/                        # Standardized 102-sample dataset & .raw tensors
│   └── output/
│       ├── presentation_figures/         # 8 publication-grade figures (300 DPI)
│       ├── previous_dataset_benchmark.csv# Authoritative 619-row empirical sweep
│       ├── PREVIOUS_DATASET_BENCHMARK_REPORT.md # Master empirical benchmark report
│       └── reconstructions/              # Model inpainting outputs (102 per model)
├── qidk-inpaint-app/                     # Native Snapdragon 8 Elite Android 15 App
└── StableDiffusion/                      # Native QNN C++ runner for SD 1.5
```

---

## Master Performance & Telemetry Summary

Empirical validation across the 102-sample dataset on Snapdragon 8 Elite hardware:

| Model Pipeline | Target Hardware | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Active Latency | Active Energy | EDP ($\text{J}\cdot\text{s}$) | Peak Temp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIGAN** | **Hexagon HTP v79** | $27.17$ | $0.9028$ | $0.1245$ | **$115.0\,\text{ms}$** | **$0.33\,\text{J}$** | **$0.038$** | $48.4^\circ\text{C}$ |
| **LaMa Dilated** | **Hexagon HTP v79** | **$29.84$** | **$0.9312$** | **$0.0891$** | $321.0\,\text{ms}$ | $0.99\,\text{J}$ | $0.318$ | $70.3^\circ\text{C}$ |
| **AOT-GAN** | **Adreno 830 GPU** | $28.45$ | $0.9184$ | $0.1012$ | $390.0\,\text{ms}$ | $1.30\,\text{J}$ | $0.507$ | $68.0^\circ\text{C}$ |
| **Stable Diffusion 1.5** | **HTP / GPU Hybrid** | $26.50$ | $0.9420$ | $0.0612$ | $50,930.0\,\text{ms}$ | $134.97\,\text{J}$ | $6,874.8$ | $74.9^\circ\text{C}$ |
| **Decision Router** | **Heterogeneous** | **$29.41$** | **$0.9304$** | **$0.0882$** | **$216.0\,\text{ms}$** | **$0.62\,\text{J}$** | **$0.134$** | **$<52.0^\circ\text{C}$** |

---

## Publication Figures & Visual Deliverables

All 8 publication-grade presentation figures are located in [`Benchmark/output/presentation_figures/`](Benchmark/output/presentation_figures/):

1. **`01_qualitative_domain_stress_grid.png`**: 6-sample $\times$ 5-column qualitative grid across representative domains.
2. **`02_architectural_showdown_radar_cdf.png`**: 5-axis Radar, Hole Repair bar chart, and 10–30 dB Robustness CDF.
3. **`03_hardware_telemetry_stress_trace.png`**: 3-tier synchronized time series of Peak SoC Temp, Active Power, and RAM.
4. **`04_regression_psnr_lpips_sensitivity.png`**: OLS regressions of Hole PSNR (invariant) & LPIPS ($R^2=0.80$) vs. Mask Area %.
5. **`05_statistical_metric_distributions_boxplots.png`**: 4-panel distribution boxplots with overlaid jittered points ($N=102$).
6. **`06_npu_vs_gpu_efficiency_and_thermal_throttling.png`**: NPU vs. GPU speedup factors and thermal throttling curves.
7. **`07_layerwise_operator_cycle_breakdown.png`**: DSP cycle breakdown parsed from SNPE profiling logs.
8. **`08_global_fid_and_edp_master.png`**: Global FID vs. Active EDP master Pareto frontier.

---

## Starting Qprof Hardware Profiler

```bash
adb forward tcp:62472 tcp:62472
adb shell
export QMONITOR_BACKEND_LIB_PATH=/vendor/qprof/backends
export LD_LIBRARY_PATH=/apex/com.android.i18n/lib64:/apex/com.android.runtime/lib64:/apex/com.android.art/lib64:/system/lib64:/vendor/lib64:/vendor/qprof/libs:$LD_LIBRARY_PATH
/vendor/bin/qmonitor-grpc-server -n 127.0.0.1 -p 62472
```

For full architectural audits, Android deployment, and benchmarking details, see **[SETUP.md](SETUP.md)**.