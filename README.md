# Neural Image Inpainting on Qualcomm Snapdragon 8 Elite (Hexagon HTP v79)

### Team Details
- **Team Name**: Black & White
- **Course / Module**: Embedded Systems Workshop
- **Hardware Platform**: Qualcomm Snapdragon 8 Elite (SM8750P / `sun`)
- **Acceleration**: Qualcomm Hexagon NPU (HTP v79) via FastRPC & QNN Runtime

---

## Executive Overview

This repository contains the complete embedded benchmarking harness, hardware telemetry pipeline, and comparative evaluation suite for neural image inpainting on the **Qualcomm Snapdragon 8 Elite Mobile Platform**. 

We benchmark and profile four distinct neural architectures on the **Hexagon HTP v79 NPU**:
1. **MIGAN** (`migan_htp_v79.dlc`): Multi-scale feed-forward GAN ($0.22\text{s}$, $0.62\text{J}$).
2. **AOT-GAN** (`aotgan.dlc`): Aggregated Contextual Transformations GAN ($0.39\text{s}$, $1.30\text{J}$).
3. **LaMa Dilated** (`lama_dilated.dlc`): Fast Fourier Transform (FFT) dilated convolutions ($0.32\text{s}$, $0.99\text{J}$).
4. **Stable Diffusion 1.5 RePaint** (`sd_qidk_runner_encoder`): 20-step iterative Euler latent diffusion pipeline ($50.93\text{s}$, $134.97\text{J}$).

All models are evaluated on a **200-pair academic benchmark dataset** (Places365 scenes + NVIDIA PConv irregular masks) across 3 mask corruption tiers with continuous Qualcomm PMIC power and thermal telemetry.

---

## Documentation Quick Links

* 📘 **[SETUP.md](SETUP.md)**: Authoritative documentation conforming to course standards (Problem, Approach, Implementation, Setup Steps, Assumptions, Results, References).
* 📊 **[FINAL_EVALUATION_REPORT.md](Benchmark/output/FINAL_EVALUATION_REPORT.md)**: Master benchmark metrics, tier stratification, and thermal rise analysis.
* 📈 **[DEEP_DIVE_STATISTICAL_REPORT.md](Benchmark/output/DEEP_DIVE_STATISTICAL_REPORT.md)**: Descriptive statistics (Mean ± $\sigma$, Median, IQR, 95% CI) and linear regression sensitivity modeling.

---

## Master Performance & Perceptual Quality Summary

| Model | Acceleration Hardware | Latency / Image | Active Power | Energy / Image | EDP ($\text{J}\cdot\text{s}$) | Peak Temp | Thermal Rise ($\Delta T$) | Global PSNR ↑ | Hole-Only PSNR ↑ | SSIM ↑ | LPIPS (VGG) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MIGAN** | **Hexagon NPU (HTP v79)** | **$0.216\,\text{s}$** | $2.86\,\text{W}$ | **$0.62\,\text{J}$** | **$0.133\,\text{J}\cdot\text{s}$** | $48.4^\circ\text{C}$ | **$+9.6^\circ\text{C}$** | $27.17\,\text{dB}$ | $19.65\,\text{dB}$ | $0.9028$ | $0.1245$ |
| **LaMa Dilated** | **Hexagon NPU (HTP v79)** | **$0.321\,\text{s}$** | $3.10\,\text{W}$ | **$0.99\,\text{J}$** | **$0.319\,\text{J}\cdot\text{s}$** | $70.3^\circ\text{C}$ | **$+24.6^\circ\text{C}$** | $28.22\,\text{dB}$ | $20.73\,\text{dB}$ | $0.9193$ | $0.1195$ |
| **AOT-GAN** | **Hexagon NPU (HTP v79)** | **$0.389\,\text{s}$** | $3.34\,\text{W}$ | **$1.30\,\text{J}$** | **$0.505\,\text{J}\cdot\text{s}$** | $68.0^\circ\text{C}$ | **$+25.4^\circ\text{C}$** | **$28.34\,\text{dB}$** | **$20.86\,\text{dB}$** | **$0.9214$** | **$0.1058$** |
| **Stable Diffusion** | **Hexagon NPU (HTP v79)** | **$50.934\,\text{s}$** | $2.65\,\text{W}$ | **$134.97\,\text{J}$** | **$6874.8\,\text{J}\cdot\text{s}$** | $74.9^\circ\text{C}$ | **$+30.0^\circ\text{C}$** | $10.13\,\text{dB}$ | $9.66\,\text{dB}$ | $0.3183$ | $0.7216$ |

---

## Visual Deliverables

* **Executive Scout Dashboard**: [`Benchmark/output/figures/scout_report_dashboard.png`](Benchmark/output/figures/scout_report_dashboard.png)
* **GAN Domain Stress 6-Grid**: [`Benchmark/output/figures/05_gan_domain_stress_6grid.jpg`](Benchmark/output/figures/05_gan_domain_stress_6grid.jpg)
* **SD vs GANs 5-Grid**: [`Benchmark/output/figures/06_sd_vs_gans_5grid.jpg`](Benchmark/output/figures/06_sd_vs_gans_5grid.jpg)
* **Perceptual Degradation Trajectories**: [`Benchmark/output/figures/02_mask_stress_regression_scatter.png`](Benchmark/output/figures/02_mask_stress_regression_scatter.png)
* **Live Hardware Telemetry Timeline**: [`Benchmark/output/hardware_telemetry_timeline.png`](Benchmark/output/hardware_telemetry_timeline.png)

---

## Quick Start (Offline Visualization)

```bash
# 1. Install prerequisites
pip install --break-system-packages torch torchvision numpy pandas scipy matplotlib pillow scikit-image lpips

# 2. Render executive dashboard
HEADLESS=1 python3 scripts/plot_scout_dashboard.py

# 3. Render 300 DPI qualitative montages
python3 scripts/fix_and_regenerate_grids.py

# 4. View generated figures
eog Benchmark/output/figures/05_gan_domain_stress_6grid.jpg \
    Benchmark/output/figures/06_sd_vs_gans_5grid.jpg &
```

For complete instructions on hardware setup, FastRPC library configuration, and on-device execution, refer to **[SETUP.md](SETUP.md)**.
