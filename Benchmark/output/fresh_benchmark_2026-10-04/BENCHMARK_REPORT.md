# QIDK (Snapdragon 8 Elite, SM8750P) – Fresh Inpainting Benchmark

Measured 2026-10-04 on the connected board (serial `8f27557f`, Android 15, SNPE 2.49 runtime, HTP v79), using the
102-sample dataset (`Benchmark/input_102`) from `TheAshSolver/ImageInpainting`. Every number below was measured in this
session by the harness in `scripts_used/` (`bench_cfg.sh`, `sd_one.sh`, `eval_quality.py`, `eval_sd.py`, `aggregate.py`) – none are copied from the repo README. The same method is packaged for re-use as `scripts/fresh_benchmark/run_fresh_benchmark.py` in both repos (verified on the board: it reproduces these numbers – see `tool_verification_run/`).

## 1. Headline table (102 samples, 512x512, `snpe-net-run --perf_profile burst`)

| Model | Runtime | Steady-state per image | Throughput | Cold single launch (what the app pays per tap) | PSNR dB | Hole-PSNR dB | SSIM | LPIPS-VGG |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MIGAN | **NPU (HTP v79)** | **49.8 ms** | 20.1 img/s | 0.32 s | 23.01 | 17.30 | 0.7112 | 0.2412 |
| LaMa-Dilated | NPU | 104.2 ms | 9.6 img/s | **7.93 s** | 23.54 | 18.18 | 0.7381 | 0.2355 |
| AOT-GAN | NPU | 158.8 ms | 6.3 img/s | **8.15 s** | 22.96 | 16.79 | 0.7301 | 0.2380 |
| MIGAN | GPU (Adreno 830) | 239.5 ms | 4.2 img/s | 1.17 s | 23.42 | 18.12 | 0.7270 | 0.2270 |
| LaMa-Dilated | GPU | 551.4 ms | 1.8 img/s | 1.25 s | 23.53 | 18.17 | 0.7380 | 0.2355 |
| AOT-GAN | GPU | 631.4 ms | 1.6 img/s | 1.25 s | 22.95 | 16.73 | 0.7300 | 0.2380 |

* Steady-state per image = (102-image batch wall time − mean cold single-image time) / 101. Three batch repeats each; spread < 1 % on NPU.
* NPU speed-up over GPU: MIGAN 4.8x, LaMa 5.3x, AOT-GAN 4.0x.
* Quality is identical between NPU and GPU within numerical noise (differences ≤ 0.1 dB) – the runtime choice does not cost accuracy.
* The quality numbers reproduce the team's own `PREVIOUS_DATASET_BENCHMARK_REPORT.md` for the 102-sample set (e.g. MIGAN/NPU 23.01 / 17.30 / 0.7112 / 0.2412 – exact match), which validates the evaluation pipeline.

### Quality by mask size (NPU)

| Model | <15 % (n=68) hole-PSNR / LPIPS | 15–30 % (n=25) | >30 % (n=9) |
|---|---|---|---|
| MIGAN | 18.35 / 0.203 | 15.15 / 0.295 | 15.41 / 0.379 |
| LaMa | 18.99 / 0.201 | 16.28 / 0.288 | 17.27 / 0.354 |
| AOT-GAN | 17.36 / 0.204 | 15.82 / 0.289 | 15.19 / 0.356 |

LaMa is the most robust on large masks (best hole-PSNR and LPIPS at >30 %), consistent with its global receptive field.

## 2. Stable Diffusion 1.5 inpainting (`sd_qidk_runner_inpaint`, 12-step DPM-Solver++ 2M, guidance 2)

10 samples evenly spread across mask coverage (002, 024, 034, 040, 044, 046, 066, 074, 080, 096), cooldown ≤ 45 °C before each, prompt `cinematic photo restoration`.

| Stage | mean ms |
|---|---:|
| VAE encoder | 1670 |
| Text encoder (CLIP) | 1559 |
| Denoising, 12 steps (~745 ms/step) | 8945 |
| VAE decode + composite | 1013 |
| **Pipeline-reported end-to-end** | **13 309** |
| **Wall-clock per launch** (process start → exit) | **25 172** (range 25 120–25 230) |

About 12 s per launch is spent loading ~1.75 GB of QNN context binaries before the pipeline's own clock starts. Peak compute-zone temperature after a run: 51.5 °C.

Same 10 samples, all models (so the comparison is apples-to-apples):

| Model | PSNR | Hole-PSNR | SSIM | LPIPS-VGG |
|---|---:|---:|---:|---:|
| SD 1.5 12-step | 18.87 | 13.46 | 0.6048 | 0.2815 |
| MIGAN (NPU) | 20.17 | 16.53 | 0.6382 | 0.2604 |
| LaMa (NPU) | 20.37 | 17.23 | 0.6734 | 0.2525 |
| AOT-GAN (NPU) | 19.99 | 15.37 | 0.6599 | 0.2542 |

SD is ~270x slower than MIGAN per image (13.3 s pipeline vs 0.05 s; ~500x counting the 25 s launch) and scores lower on every reference metric on this set. See `qualitative_comparison.png`.
Caveat: PSNR/SSIM/LPIPS against a single reference penalise generative output; n = 10 is small.

## 3. Cold-start fix found: SNPE init cache (verified)

LaMa and AOT-GAN spend ~8 s in every fresh `snpe-net-run` launch (the DLC is re-prepared for HTP each time). That is the "8 s lag" seen in the app. Re-running with `--enable_init_cache` on a *copy* of each DLC builds a cache once (one 8 s run), after which:

| Model | Cold launch before | Cold launch after (cached copy) | Output vs. baseline |
|---|---:|---:|---|
| LaMa-Dilated | 7.93 s | **0.47–0.50 s** | bit-identical on all 102 outputs |
| AOT-GAN | 8.15 s | **0.56–0.58 s** | bit-identical on all 102 outputs |

The cached copies are left on the board as `/data/local/tmp/lama/bench_cache_lama_dilated.dlc` (280 MB) and `bench_cache_aotgan.dlc` (97 MB); the originals are untouched. MIGAN's DLC already launches in 0.3 s.

## 4. Thermals (compute zones: CPU/GPU/NSP max, start → end of 3 back-to-back 102-image batches)

| Config | °C |
|---|---|
| MIGAN NPU | 40.7/39.2/38.6 → 46.1/44.2/44.8 |
| LaMa NPU | 49.2/48.4/47.4 → 62.3/58.8/61.2 |
| AOT-GAN NPU | 51.9/49.9/49.0 → 73.4/61.5/64.3 |
| MIGAN GPU | 46.9/44.2/43.6 → 68.8/70.3/63.9 |
| LaMa GPU | 47.3/45.7/44.4 → 87.2/89.9/79.6 |
| AOT-GAN GPU | 49.2/47.6/45.9 → 86.5/91.5/79.6 |

GPU runs of LaMa/AOT-GAN throttle: batch time grows 8.1 % / 10.2 % from rep 1 to rep 3. NPU runs show no measurable throttling (≤ 1 %).

## 5. Things in the repo that do not hold up (please fix before presenting)

1. **Energy / power numbers are not real measurements.** On this board `battery/current_now` stays within ±5 mA even with all 8 CPU cores pegged and updates only every ~1.5 s; `power_now` is 0. `scripts/thermal_logger.sh` substitutes 350 mA × 8.97 V ≈ 3.1 W whenever the reading is < 50 mA, and the logs hold only 2–63 one-second samples per model. The "2.6–3.3 W", "0.62 J", "35 J" values are therefore essentially a constant times latency. I did **not** report energy. A real number needs an external power meter or the Qualcomm Profiler rail data.
2. **`run_two_phase_batch_benchmark.py` reports hard-coded latencies.** If a model's output folder already has 102 results it skips the run and uses `pure_kernel_baseline_ms` constants (68/189/237/165/275/260 ms); wall latency is that constant + 450; telemetry falls back to 2.85 W; a missing output silently falls back to the *input* image, inflating metrics.
3. **`python main.py benchmark --samples 102` crashes** – it passes `--samples`, which the script does not accept.
4. **README master table mixes sources.** Its quality columns (e.g. MIGAN 27.17 dB / 0.9028 / 0.1245; LaMa 29.84 / 0.9312 / 0.0891; SD 26.50 / 0.9420 / 0.0612; Router 29.41 / 0.9304 / 0.0882) do not match the repo's own authoritative 102-sample report or my measurements (MIGAN 23.01 / 0.7112 / 0.2412; LaMa 23.54 / 0.7381 / 0.2355). Latency columns (115 / 321 / 390 ms; SD 12.2 s) also do not match measurement (49.8 / 104 / 159 ms steady; SD 13.3 s pipeline, 25.2 s wall). I did not run the router, so its row is unverified.
5. **SD "12.2 s"** is the pipeline-internal time; the user-visible time per launch is ~25 s unless the runner stays resident.
6. The 102-sample set is an **object-removal** benchmark (input still contains the object; ground truth is the scene without it), so absolute PSNR/SSIM are bounded; compare models against each other, not against inpainting papers.

## 6. Method and limits

* Latency: three cold `N=1` launches (after a warm-up launch so flash caching is not charged) and three `N=102` batch launches per model/runtime; monotonic `/proc/uptime` clock (the board's wall clock steps – one early run showed a negative duration, so it was discarded). 10 ms resolution.
* Cooldown to ≤ 42 °C (GAN runs) / ≤ 45 °C (SD) on the hottest CPU/GPU/NSP zone before each configuration (timeout 150 s / 120 s; start temperatures are recorded in `raw_logs/device_runs_plain.txt`; LaMa/AOT-GAN NPU runs started warmer, ~49–52 °C, with no measurable effect).
* Quality: composite `image·(1−mask) + output·mask`, PSNR/SSIM via scikit-image, LPIPS-VGG with the official VGG16 ImageNet weights. A custom VGG16 loader was used because torchvision's native extension does not load in this environment; it reproduces the team's LPIPS values exactly.
* Not measured: energy, CPU runtime, the Android app end-to-end, the decision router, FID, the 200-pair dataset. `Benchmark/dataset_previous/ideal` was not downloaded.
* The `ESW-M26/esw-m26-19_black_and_white` repo returned 404 (private / needs login), so only `TheAshSolver/ImageInpainting` was used. The chat says it holds the same code.

## Files

* `figures/` - presentation figures (box plots, Pareto, radar, telemetry, CDFs, mask-size regression/tiers, 10 per-sample sheets)
* `raw_logs/device_runs_plain.txt` - raw timings/temps for all GAN configs; `raw_logs/sd_runs_plain.txt` - raw SD timings and stage profile; `raw_logs/cache_probe.txt` - init-cache experiment; `raw_logs/lpips_summary.txt` and `sd_lpips_summary.txt` - LPIPS scoring output
* `summary.json` - aggregated numbers; `per_sample_csv/quality_*.csv` - per-sample quality (`*_lpips.csv` include LPIPS); `per_sample_csv/quality_sd_vs_gan_subset*.csv`
* `qualitative_comparison.png` - input / mask / ground truth / MIGAN / LaMa / AOT-GAN / SD
