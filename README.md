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
1. **MIGAN** (`models/Migan/migan_htp_v79.dlc`): Multi-scale depthwise separable GAN ($49.8\,\text{ms}$/image steady-state on NPU, $4.8\times$ faster than the GPU).
2. **LaMa Dilated** (`models/LamaDilated/lama_dilated.dlc`): Fast Fourier Convolutions (FFC) with infinite global receptive field ($104\,\text{ms}$/image on NPU, $5.3\times$ faster than the GPU).
3. **AOT-GAN** (`models/AOT-GAN/aotgan.dlc`): Aggregated Contextual Transformations GAN ($159\,\text{ms}$/image on NPU, $4.0\times$ faster than the GPU).
4. **Stable Diffusion 1.5 Inpainting** (`StableDiffusion/`): 12-step DPM-Solver++ (2M) latent diffusion pipeline on Hexagon HTP v79 NPU ($13.3\,\text{s}$ pipeline, $25.2\,\text{s}$ wall-clock per launch including model loading), accelerated from the earlier 20-step Euler baseline ($50.9\,\text{s}$).

> All figures in this README were re-measured on 2026-10-04 with [`scripts/fresh_benchmark/`](scripts/fresh_benchmark/) (see [`Benchmark/output/fresh_benchmark_2026-10-04/`](Benchmark/output/fresh_benchmark_2026-10-04/), including the presentation figures in its `figures/` folder). Energy / power are **not** reported: the board exposes no usable system-power sensor (see [Known issues](#known-issues--measurement-caveats)).

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
│   ├── fresh_benchmark/                  # Measured benchmark harness (latency, thermals, quality, SD), figure generator, init-cache tool
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

## Master Performance Summary (measured 2026-10-04)

102-sample dataset (`Benchmark/input_102`, 512×512), `snpe-net-run --perf_profile burst`, Snapdragon 8 Elite QIDK, Android 15, SNPE 2.49. Quality = PSNR / hole-PSNR / SSIM / LPIPS-VGG of `image·(1−mask) + output·mask` vs. ground truth. *Steady per image* = (102-image batch − cold launch) / 101 (3 repeats, spread < 1 % on NPU). *Cold launch* = one fresh `snpe-net-run` process for one image (what the app pays per tap today).

| Model | Runtime | Steady per image | Cold launch | PSNR (dB) ↑ | Hole-PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIGAN** | **NPU (HTP v79)** | **49.8 ms** | 0.32 s | 23.01 | 17.30 | 0.7112 | 0.2412 |
| **LaMa Dilated** | NPU | 104.2 ms | 7.93 s (**0.47 s** with init cache) | 23.54 | 18.18 | 0.7381 | 0.2355 |
| **AOT-GAN** | NPU | 158.8 ms | 8.15 s (**0.57 s** with init cache) | 22.96 | 16.79 | 0.7301 | 0.2380 |
| MIGAN | GPU (Adreno 830) | 239.5 ms | 1.17 s | 23.42 | 18.12 | 0.7270 | 0.2270 |
| LaMa Dilated | GPU | 551.4 ms | 1.25 s | 23.53 | 18.17 | 0.7380 | 0.2355 |
| AOT-GAN | GPU | 631.4 ms | 1.25 s | 22.95 | 16.73 | 0.7300 | 0.2380 |
| **Stable Diffusion 1.5** (12-step, n = 10) | NPU | 13.3 s pipeline / 25.2 s per launch | 25.2 s | 18.87 | 13.46 | 0.6048 | 0.2815 |
| *Decision Router* | – | *not re-measured* | – | – | – | – | – |

* NPU vs GPU: 4.8× (MIGAN), 5.3× (LaMa), 4.0× (AOT-GAN) faster; quality is the same within ≤ 0.1 dB.
* GPU runs of LaMa / AOT-GAN throttle (+8 % / +10 % batch time over 3 repeats, GPU zone ≈ 90 °C); NPU runs do not (≤ 1 %).
* On the same 10 samples as SD: MIGAN 20.17 / 16.53 / 0.638 / 0.260, LaMa 20.37 / 17.23 / 0.673 / 0.253, AOT-GAN 19.99 / 15.37 / 0.660 / 0.254 (PSNR / hole-PSNR / SSIM / LPIPS).
* LaMa is the most robust for masks > 30 % of the image (hole-PSNR 17.27 vs 15.4 / 15.2 for MIGAN / AOT-GAN).
* The 102-sample set is an **object-removal** benchmark (the input still contains the object; ground truth is the scene without it), so compare models against each other rather than to inpainting papers.
* Cold-start fix: build an SNPE HTP init cache once per DLC (`scripts/fresh_benchmark/make_init_cache.sh`); outputs are bit-identical on all 102 images.

Full tables, per-mask-size breakdown, thermals, raw logs and the **presentation figures** (box plots, Pareto, radar, telemetry, CDFs, per-sample sheets): [`Benchmark/output/fresh_benchmark_2026-10-04/`](Benchmark/output/fresh_benchmark_2026-10-04/). Re-run everything with `python scripts/fresh_benchmark/run_fresh_benchmark.py --lpips --sd`, then regenerate figures with `python scripts/fresh_benchmark/make_figures.py`.

### Known issues / measurement caveats

1. **No energy / power numbers.** `/sys/class/power_supply/battery/current_now` stays within ±5 mA even with all CPU cores at 100 % (updates every ~1.5 s, `power_now` = 0). `scripts/thermal_logger.sh` substitutes 350 mA × 8.97 V ≈ 3.1 W whenever the reading is < 50 mA, so the earlier "J per image" / EDP figures were essentially a constant times latency. A real number needs an external power meter or Qualcomm Profiler rail data.
2. `scripts/run_two_phase_batch_benchmark.py` uses hard-coded `pure_kernel_baseline_ms` constants when a model's output folder is already populated, adds a fixed +450 ms for "wall" latency, falls back to 2.85 W telemetry, and falls back to the *input image* when an output is missing. Use `scripts/fresh_benchmark/run_fresh_benchmark.py` for latency numbers.
3. The earlier README table mixed quality columns from another run (e.g. MIGAN 27.17 dB / 0.9028 / 0.1245) with latency columns that do not reproduce; the router row (29.41 dB) was never re-measured.
4. Stable Diffusion's "12.2 s" is the pipeline-internal time; each launch also spends ~12 s loading ~1.75 GB of QNN context binaries (25.2 s wall) unless the runner is kept resident.

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