# DATA PROVENANCE & EMPIRICAL TRACEABILITY REPORT
**Date**: 2026-11-19  
**Audit Scope**: Provenance trace of `artifacts/train_metrics_4800.csv`, `artifacts/test_metrics_1200.csv`, and `artifacts/dataset_100_features_6000.csv`.

---

## 1. Provenance Audit: `train_metrics_4800.csv` & `test_metrics_1200.csv`

A source-code audit traced `artifacts/train_metrics_4800.csv` and `artifacts/test_metrics_1200.csv` directly back to [split_dataset_unbiased.py](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L48-L147).

### Column-by-Column Provenance Classification:

| Column Name | Provenance Tag | Source / Generation Formula | Evidence |
| :--- | :---: | :--- | :--- |
| `sample_id` | **MEASURED_OFFLINE** | Ingested from `Benchmark/6000_images/compiled/dataset_manifest.csv` | Real image filenames and sample IDs |
| `category` | **MEASURED_OFFLINE** | Ingested from `Benchmark/6000_images/compiled/dataset_manifest.csv` | Real COCO/Places365 semantic class tags |
| `model` | **CATEGORICAL** | 7 target model arms: `migan_npu`, `lama_npu`, `aotgan_npu`, `migan_gpu`, `lama_gpu`, `aotgan_gpu`, `telea_cpu` | Explicit arm definition |
| `mask_coverage_pct` | **MEASURED_OFFLINE** | Derived from `Benchmark/6000_images/compiled/masks_manifest.csv` (`coverage_pct`) | Real bitmap hole pixel count |
| `edge_density` | **MEASURED_OFFLINE** | Ingested from `dataset_100_features_6000.csv` (`grad_canny_density`) | Computed via Canny filter on 6k images |
| `silicon_temp_c` | **MODELED** | `np.random.uniform(34.0, 78.0, size=n_samples)` | [split_dataset_unbiased.py#L77](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L77) |
| `is_throttled` | **MODELED** | `(temp >= 75.0)` | [split_dataset_unbiased.py#L89](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L89) |
| `latency_ms` | **MODELED** | Anchored on `HARDWARE_BASE` device averages, plus thermal step $+15\%$ and Gaussian noise `np.random.normal(0, 0.5)` | [split_dataset_unbiased.py#L93-L102](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L93-L102) |
| `active_power_w` | **MODELED** | Anchored on `HARDWARE_BASE` PMIC averages, plus throttling step and `np.random.normal(0, 0.03)` | [split_dataset_unbiased.py#L96-L103](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L96-L103) |
| `energy_joules` | **DERIVED** | `active_power_w * (latency_ms / 1000.0)` | [split_dataset_unbiased.py#L104](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L104) |
| `psnr` | **MODELED** | Closed-form heuristic: $33.5 - 26 \cdot c + \epsilon$ (per-arm domain formulas) | [split_dataset_unbiased.py#L111-L128](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L111-L128) |
| `ssim` | **MODELED** | Closed-form heuristic: $0.952 - 0.55 \cdot \max(0, c - 0.09) - 0.12 \cdot hf + \epsilon$ | [split_dataset_unbiased.py#L110-L126](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L110-L126) |
| `lpips` | **MODELED** | Closed-form heuristic: $0.098 + 1.15 \cdot \max(0, c - 0.09) + 0.18 \cdot hf + \epsilon$ | [split_dataset_unbiased.py#L110-L127](file:///home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/split_dataset_unbiased.py#L110-L127) |

### Key Scientific Takeaway:
`artifacts/train_metrics_4800.csv` and `artifacts/test_metrics_1200.csv` are **physically-anchored modeled datasets**, NOT raw per-image oscilloscope or on-device profiler traces. They are parameterized by physical hardware baselines (`HARDWARE_BASE`), but sample-level noise and thermal assignments were synthesized mathematically.

---

## 2. Plain Scope of What the 6k Dataset Can Support

1. **What the 6k Dataset CAN Support (Valid Claims)**:
   - **Image-Aware Routing**: Evaluating how routers trade off void geometry (`mask_coverage_pct`, `mask_area_ratio`) and structural edge density (`grad_canny_density`) against model domain strengths (e.g. Telea on tiny voids, MIGAN on faces/small voids, LaMa on large voids, AOT-GAN on dense textures).
   - **Feature Selection & Complexity Budget**: Measuring the CPU execution cost and information gain of 100+ vision features extracted from the 6,000 real images.
   - **Offline Replay Comparison**: Benchmarking algorithm convergence (Heuristic vs LinUCB vs RL vs Deep MLP vs Oracle) under offline replay.

2. **What the 6k Dataset CANNOT Support (Invalid / Banned Claims)**:
   - **System-Load-Aware Routing**: The 6k dataset does NOT contain real system-load dimensions (no background contention tiers L0-L4, no memory bus thrashing, no concurrent GPU/NPU workloads).
   - **Thermal Throttling Dynamics**: The temperature values (`silicon_temp_c`) are uniform random numbers (`uniform(34, 78)`), not real thermal trajectories or heating curves.
   - **Conclusion**: Any claim regarding system contention or thermal adaptation evaluated on this 6k dataset must be explicitly tagged **`SIMULATED`** or **`MODELED`**. Publication-grade claims of load-aware routing require physical on-device execution with the load-generator app.
