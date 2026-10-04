# fresh_benchmark – measured on-device benchmark for the QIDK

Replaces the constants-based numbers of `scripts/run_two_phase_batch_benchmark.py` with real measurements.

| File | Purpose |
|---|---|
| `run_fresh_benchmark.py` | Host orchestrator: builds tensors from `Benchmark/input_102` PNGs, pushes them, runs every model/runtime on the board, pulls outputs, scores quality, writes `summary.json` + `BENCHMARK_RESULTS.md` |
| `make_figures.py` | Presentation figures from a results folder: box plots, Pareto, radar, telemetry (latency / cold start / thermals), CDFs, mask-size regression and tier bars, per-sample qualitative sheets |
| `device/bench_cfg.sh` | On-device: cooldown barrier, 3 cold launches (N=1) + 3 warm 102-image batches, temps before/after each run, monotonic clock |
| `device/sd_one.sh` | On-device: one Stable Diffusion run with stage profile and wall-clock |
| `make_init_cache.sh` | One-off: builds SNPE HTP init-cache copies of the LaMa / AOT-GAN DLCs (cold launch 8 s → ~0.5 s, bit-identical outputs) |

```bash
# device connected + authorized, adb on PATH (or --adb /path/to/adb)
python scripts/fresh_benchmark/run_fresh_benchmark.py                       # 6 GAN configs, PSNR/hole-PSNR/SSIM
python scripts/fresh_benchmark/run_fresh_benchmark.py --lpips --sd --keep-raw   # + LPIPS-VGG + Stable Diffusion (10 samples)
python scripts/fresh_benchmark/run_fresh_benchmark.py --configs migan_npu lama_npu
# cached DLCs:
adb push scripts/fresh_benchmark/make_init_cache.sh /data/local/tmp/ && adb shell sh /data/local/tmp/make_init_cache.sh
python scripts/fresh_benchmark/run_fresh_benchmark.py --configs lama_npu aotgan_npu --cached-dlc
# or through the master CLI:
python main.py fresh-benchmark --lpips --sd

# figures (needs a run made with --lpips --sd; add --keep-raw for the per-sample sheets)
python scripts/fresh_benchmark/make_figures.py --results Benchmark/output/fresh_benchmark/<timestamp> \
    --ds Benchmark/input_102 --raw Benchmark/output/fresh_benchmark/<timestamp>/raw_outputs \
    --sd Benchmark/output/fresh_benchmark/<timestamp>/sd --out Benchmark/output/fresh_benchmark/<timestamp>/figures
```

Results land in `Benchmark/output/fresh_benchmark/<timestamp>/`. Prerequisites: models and the SNPE/QNN runtime already on the board
(`/data/local/tmp/lama`, `/data/local/tmp/sd_runtime`), Python with `numpy pillow scikit-image` (+ `torch lpips` for `--lpips`, `matplotlib` for figures).
If torchvision's native extension cannot load on the host, set `VGG16_WEIGHTS=/path/to/vgg16-397923af.pth` and the script
builds the VGG16 backbone LPIPS needs directly.

## Method
* Cold launch = a fresh `snpe-net-run` process for one image (after one warm-up launch so flash caching is not charged); batch = one process for all 102 images.
  Steady per image = (batch − mean cold) / 101. Three repeats each; the board's wall clock steps, so timing uses `/proc/uptime` (10 ms resolution).
* Cooldown to ≤ 42 °C on the hottest CPU/GPU/NSP thermal zone (150 s timeout) before each configuration; temperatures before/after every run are logged.
* Quality: `final = image·(1−mask) + output·mask` against the ground truth; PSNR, hole-PSNR, SSIM (scikit-image), LPIPS-VGG. A missing, non-finite or wrongly-sized output aborts the run.
* **Energy is intentionally not reported**: battery `current_now` stays within ±5 mA under full load and `power_now` is 0, so any "joules per image" derived from it is a constant times latency.
