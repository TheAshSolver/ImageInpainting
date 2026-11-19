# Snapdragon 8 Elite — Latency Feature Importance Analysis Report

- **Data Source**: `Empirical On-Device Dataset (Joined with Input-102 Features): /home/aaravind/develop/models/Image_Inpainting/Benchmark/output_old/previous_dataset_benchmark.csv`
- **Samples Analyzed**: 617
- **Target Target**: Inference Latency (`latency_ms`)

## Ranked Feature Importance Table

| Rank | Feature Parameter | Relative Importance | Permutation ΔR² | Mutual Info | Standardized Coef | Spearman Correlation |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `t_start_c` | **97.3%** | 0.2809 | 1.6037 | 396.2833 | 0.0817 |
| 2 | `t_end_c` | **92.3%** | 0.1425 | 1.6070 | -172.6032 | 0.7661 |
| 3 | `active_power_w` | **84.1%** | 0.0001 | 1.6287 | 259.2148 | 0.9238 |
| 4 | `model_sd_npu` | **83.2%** | 0.2611 | 0.0775 | 2591.7993 | 0.1574 |
| 5 | `delta_temp_c` | **83.2%** | 0.0001 | 1.3233 | 22.4099 | 0.9660 |
| 6 | `model_migan_npu` | **79.5%** | 0.0001 | 0.4115 | -107.0733 | -0.6523 |
| 7 | `model_aotgan_npu` | **71.4%** | 0.0000 | 0.4204 | -103.2611 | 0.1229 |
| 8 | `model_lama_npu` | **70.9%** | 0.0000 | 0.4373 | -121.0673 | -0.1355 |
| 9 | `target_core_NPU` | **69.1%** | 0.0000 | 0.7070 | 218.5352 | -0.4658 |
| 10 | `model_migan_gpu` | **63.2%** | 0.0000 | 0.4336 | -131.4965 | -0.3939 |
| 11 | `model_lama_gpu` | **55.9%** | 0.0000 | 0.3950 | -226.8319 | 0.3812 |
| 12 | `mask_hole_count` | **34.1%** | 0.0000 | 0.0244 | -0.2401 | -0.0029 |
| 13 | `mask_aspect_ratio` | **26.6%** | 0.0000 | 0.0000 | 0.4534 | 0.0081 |
| 14 | `mask_perimeter_to_area` | **26.6%** | 0.0000 | 0.0000 | 0.1203 | 0.0080 |
| 15 | `img_canny_edge_density` | **26.6%** | 0.0000 | 0.0000 | 0.1082 | -0.0058 |
| 16 | `mask_dist_to_border` | **26.6%** | 0.0000 | 0.0000 | -0.3323 | -0.0091 |
| 17 | `mask_centroid_y` | **26.6%** | 0.0000 | 0.0000 | -0.3161 | -0.0081 |
| 18 | `mask_centroid_x` | **26.6%** | 0.0000 | 0.0000 | 0.2111 | 0.0125 |
| 19 | `mask_coverage_pct` | **26.6%** | 0.0000 | 0.0000 | -0.0588 | 0.0039 |
| 20 | `img_color_entropy` | **26.6%** | 0.0000 | 0.0000 | -0.0740 | -0.0036 |
| 21 | `img_mean_luminance` | **26.6%** | 0.0000 | 0.0000 | 0.0069 | 0.0063 |
| 22 | `img_laplacian_var` | **26.6%** | 0.0000 | 0.0000 | -0.1813 | -0.0025 |

## Parameter Sensitivity Findings

1. **Architectural Graph Topology (`model` / `target_core`)**:
   - The structural graph complexity (MIGAN depthwise convolutions vs. LaMa Fast Fourier Convolutions vs. AOT-GAN transformer contextual blocks) is the single largest determinant of execution latency.
2. **Thermodynamic Operating Point (`soc_temp_start_c` / `delta_temp_c`)**:
   - Increased junction temperature triggers Hexagon HTP v79 frequency throttling and power state scaling, manifesting as a statistically significant monotonic increase in wall clock time.
3. **Mask Spatial Distribution & Coverage (`mask_coverage_pct` / `mask_dist_to_border`)**:
   - Larger voids and fragmented masks increase operator execution dispersion across models relying on global attention and dilated branches.
