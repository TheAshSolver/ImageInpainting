# QIDK benchmarks - measured 2026-10-04

Fresh, measured benchmark of MIGAN / LaMa-Dilated / AOT-GAN (NPU and GPU) and Stable Diffusion 1.5 inpainting on the connected QIDK
(Snapdragon 8 Elite, serial `8f27557f`, Android 15, SNPE 2.49), 102-sample dataset from `ImageInpainting/Benchmark/input_102`.

**Start with [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md)** (tables, per-mask-size breakdown, SD stage profile, thermals, and problems found in the
repo's earlier numbers) and **[`figures/`](figures/)** (presentation figures).

## Headline (steady-state per image, 102 samples)

| Model | NPU | GPU | Cold launch NPU | PSNR / SSIM / LPIPS |
|---|---:|---:|---:|---|
| MIGAN | **49.8 ms** | 239.5 ms | 0.32 s | 23.01 / 0.711 / 0.241 |
| LaMa-Dilated | 104.2 ms | 551.4 ms | 7.93 s (0.47 s with init cache) | 23.54 / 0.738 / 0.236 |
| AOT-GAN | 158.8 ms | 631.4 ms | 8.15 s (0.57 s with init cache) | 22.96 / 0.730 / 0.238 |
| SD 1.5 (12-step, n=10) | 13.3 s pipeline, 25.2 s per launch | - | - | 18.87 / 0.605 / 0.282 |

Energy is not reported: the board has no usable system-power sensor (see report section 5).

## Figures (`figures/`)

| File | Content |
|---|---|
| `01_metric_distributions_boxplots.png` | Global PSNR, hole PSNR, SSIM, LPIPS: box + jittered samples for the three GANs (n=102) and SD (n=10) |
| `02_latency_vs_quality_pareto.png` | Steady latency vs PSNR (per-sample clouds, NPU vs GPU) and vs LPIPS with Pareto frontier |
| `03_architectural_radar.png` | Seven-axis radar of the three GANs on the NPU |
| `04_telemetry_latency_coldstart_thermals.png` | Throughput NPU vs GPU, cold-start (incl. init-cache), thermals and throttling |
| `05_quality_robustness_cdf.png` | Empirical CDFs of PSNR and SSIM with pass rates |
| `06_mask_size_sensitivity_regression.png` | Hole PSNR / LPIPS vs mask coverage with OLS fits |
| `07_quality_by_mask_tier.png` | Hole PSNR / LPIPS by light / medium / heavy masks |
| `sample_<id>_qualitative.png` (x10) | Per-sample sheet: input + mask, ground truth, MIGAN / LaMa / AOT-GAN / SD outputs with per-sample metrics |

Regenerate with `scripts_used/make_figures.py` (also shipped as `scripts/fresh_benchmark/make_figures.py` in both repos).

## Contents

| Path | What |
|---|---|
| `BENCHMARK_REPORT.md` | Full report |
| `summary.json` | All aggregated numbers (machine-readable) |
| `qualitative_comparison.png` | Compact overview: input / mask / ground truth / MIGAN / LaMa / AOT-GAN / SD |
| `figures/` | Presentation figures (above) |
| `per_sample_csv/` | Per-sample PSNR, hole-PSNR, SSIM (+ LPIPS in `*_lpips.csv`) for every configuration; SD vs GAN on the SD subset |
| `raw_logs/` | Raw on-device timings and temperatures, SD stage profiles, init-cache experiment, LPIPS output |
| `scripts_used/` | The exact scripts used to take these measurements (device scripts, evaluators, aggregator, probes for the power sensor, figure generator) |
| `tool_verification_run/` | Output of the packaged tool `scripts/fresh_benchmark/run_fresh_benchmark.py` re-run on the board (cached LaMa / AOT-GAN + 2 SD samples) - reproduces the numbers above |

## Repo changes made alongside (both `ImageInpainting` and `esw-m26-19_black_and_white`, uncommitted)

* `README.md`, `SETUP.md`: results sections replaced with the measured numbers; "Known issues / measurement caveats" added
* `main.py`: fixed `benchmark --samples` crash; added `fresh-benchmark` subcommand
* `scripts/fresh_benchmark/`: measured-benchmark harness, figure generator, init-cache tool, README
* `Benchmark/output/fresh_benchmark_2026-10-04/`: copy of this folder's report, summary, per-sample CSVs, logs, comparison image and figures

## Left on the board

`/data/local/tmp/lama/bench_cache_lama_dilated.dlc` and `bench_cache_aotgan.dlc` (init-cache copies, originals untouched), `bench_in/` (input tensors),
`list*.txt` input lists, and the benchmark scripts in `/data/local/tmp/`. `sd_runtime/image.raw` and `mask.raw` were overwritten by the SD runs.
