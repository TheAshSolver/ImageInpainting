#!/usr/bin/env python3
"""
context.py
================================================================================
🤖 AI-TO-AI CONTEXT HANDOVER & REPOSITORY SPECIFICATION
================================================================================
This file is explicitly designed for seamless, lossless handover between AI coding
assistants, autonomous agents, and systems engineers. It encapsulates the full
architectural blueprint, file-by-file ontology, model tensor contracts, hardware
acceleration quirks, known bugs/fixes, and execution protocols for:

  PROJECT: Neural Image Inpainting on Qualcomm Snapdragon 8 Elite
  TARGET SOC: Qualcomm Snapdragon 8 Elite Development Kit (SM8750P / sun)
  NPU ARCH: Qualcomm Hexagon HTP v79 Architecture via FastRPC & QNN/SNPE
  COMPANION GPU: Qualcomm Adreno 830 GPU
  UPSTREAM REPO: https://github.com/TheAshSolver/ImageInpainting.git
  CURRENT SYNCED COMMIT: 0c5cfb1 (Merged main branch with Spacex2006 credentials)

Usage by AI Agents:
  - Query programmatically:
      import context
      ctx = context.get_context()
      model_info = context.MODEL_REGISTRY["migan"]
  - Run from terminal:
      python context.py --summary        # Human/AI readable high-level overview
      python context.py --json           # Machine-readable full JSON dump
      python context.py --models         # Detailed model specs and tensor polarities
      python context.py --quirks         # Critical Qualcomm hardware & Android 15 quirks
      python context.py --file-map       # Comprehensive file-to-role directory map
"""

import sys
import json
from typing import Dict, Any

# ==============================================================================
# 1. PROJECT METADATA & ARCHITECTURE OVERVIEW
# ==============================================================================
PROJECT_METADATA: Dict[str, Any] = {
    "project_name": "Qualcomm Snapdragon 8 Elite Neural Image Inpainting",
    "version": "2.4.0 (Synchronized 2026-10-05)",
    "upstream_repository": "https://github.com/TheAshSolver/ImageInpainting.git",
    "primary_maintainers": ["Spacex2006", "rnn27", "Tarsh"],
    "target_platform": {
        "soc": "Qualcomm Snapdragon 8 Elite (SM8750P / APQ8750)",
        "board_alias": "sun / Sun for arm64",
        "os": "Android 15 (API level 35) / Linux kernel 7.0+",
        "npu_core": "Qualcomm Hexagon HTP v79 NPU",
        "dsp_bridge": "Qualcomm FastRPC CDSP daemon (/dev/fastrpc-cdsp)",
        "gpu_core": "Qualcomm Adreno 830 GPU",
        "ddr": "LPDDR5X (up to 4.8 GHz, 76.8 GB/s)"
    },
    "core_purpose": (
        "High-throughput, thermally stable, zero-cloud on-device image inpainting "
        "and object removal. Combines feed-forward convolutional GANs (sub-50ms) "
        "and generative latent diffusion (12-step DPM-Solver++) orchestrated by an "
        "intelligent multi-modal decision router running directly on mobile silicon."
    )
}

# ==============================================================================
# 2. MODEL REGISTRY & TENSOR CONTRACTS
# ==============================================================================
MODEL_REGISTRY: Dict[str, Any] = {
    "migan": {
        "canonical_name": "MIGAN (Multi-scale Inpainting GAN)",
        "role": "Facial symmetry, portrait restoration, and ultra-low-latency inpainting",
        "container_filename": "migan_htp_v79.dlc",
        "fallback_filename": "migan.dlc",
        "target_hardware": "Qualcomm Hexagon HTP v79 NPU",
        "runtime_flag": "--use_dsp",
        "staging_dir": "/data/local/tmp/lama",
        "input_tensor_shape": [1, 3, 512, 512],
        "input_tensor_dtype": "float32 (little-endian raw bytes)",
        "input_normalization": "[0.0, 1.0] RGB",
        "mask_polarity": {
            "value_at_hole": 0.0,
            "value_at_keep": 1.0,
            "label": "INVERTED (0 = hole / missing, 1 = background / keep)",
            "warning": "Passing standard polarity causes severe white block artifacts"
        },
        "output_tensor_shape": [1, 3, 512, 512],
        "output_tensor_dtype": "float32",
        "measured_performance": {
            "steady_latency_npu_ms": 49.8,
            "steady_latency_gpu_ms": 239.5,
            "npu_speedup_vs_gpu": "4.8x",
            "cold_start_latency_s": 0.32,
            "global_psnr_db": 23.01,
            "hole_psnr_db": 17.30,
            "ssim": 0.7112,
            "lpips": 0.2412,
            "thermal_stability": "Strictly ≤ 46 °C SoC"
        }
    },
    "lama": {
        "canonical_name": "LaMa Dilated (Large Mask Inpainting with Fast Fourier Convolutions)",
        "role": "Large irregular voids, wide object removal, global repetitive textures",
        "container_filename": "lama_dilated.dlc",
        "target_hardware": "Qualcomm Hexagon HTP v79 NPU (or Adreno 830 GPU)",
        "runtime_flag": "--use_dsp (NPU) or --use_gpu (GPU)",
        "staging_dir": "/data/local/tmp/lama",
        "input_tensor_shape": [1, 3, 512, 512],
        "input_tensor_dtype": "float32 (little-endian raw bytes)",
        "input_normalization": "[0.0, 1.0] RGB",
        "mask_polarity": {
            "value_at_hole": 1.0,
            "value_at_keep": 0.0,
            "label": "STANDARD (1 = hole / missing, 0 = background / keep)"
        },
        "output_tensor_shape": [1, 3, 512, 512],
        "output_tensor_dtype": "float32",
        "measured_performance": {
            "steady_latency_npu_ms": 104.2,
            "steady_latency_gpu_ms": 551.4,
            "npu_speedup_vs_gpu": "5.3x",
            "cold_start_uncached_s": 7.93,
            "cold_start_with_init_cache_s": 0.47,
            "global_psnr_db": 23.54,
            "hole_psnr_db": 18.18,
            "ssim": 0.7381,
            "lpips": 0.2355,
            "robustness": "Most resilient model for masks > 30% area"
        }
    },
    "aotgan": {
        "canonical_name": "AOT-GAN (Aggregated Contextual Transformations GAN)",
        "role": "Dense textures, high-frequency edges, architecture, cobblestone, vegetation",
        "container_filename": "aotgan.dlc",
        "target_hardware": "Qualcomm Hexagon HTP v79 NPU (or Adreno 830 GPU)",
        "runtime_flag": "--use_dsp (NPU) or --use_gpu (GPU)",
        "staging_dir": "/data/local/tmp/lama",
        "input_tensor_shape": [1, 3, 512, 512],
        "input_tensor_dtype": "float32 (little-endian raw bytes)",
        "input_normalization": "[0.0, 1.0] RGB",
        "mask_polarity": {
            "value_at_hole": 1.0,
            "value_at_keep": 0.0,
            "label": "STANDARD (1 = hole / missing, 0 = background / keep)"
        },
        "output_tensor_shape": [1, 3, 512, 512],
        "output_tensor_dtype": "float32",
        "measured_performance": {
            "steady_latency_npu_ms": 158.8,
            "steady_latency_gpu_ms": 631.4,
            "npu_speedup_vs_gpu": "4.0x",
            "cold_start_uncached_s": 8.15,
            "cold_start_with_init_cache_s": 0.57,
            "global_psnr_db": 22.96,
            "hole_psnr_db": 16.79,
            "ssim": 0.7301,
            "lpips": 0.2380
        }
    },
    "sd_inpaint": {
        "canonical_name": "Stable Diffusion 1.5 Inpaint (Optimized 12-Step DPM-Solver++)",
        "role": "Generative hallucination, complex semantic synthesis",
        "runner_binary": "sd_qidk_runner_inpaint",
        "target_hardware": "Qualcomm Hexagon HTP v79 NPU",
        "staging_dir": "/data/local/tmp/sd_runtime",
        "model_container": "models/unet.bin (graph_wlqbe2kd)",
        "source_file": "StableDiffusion/sd_runner_inpaint.cpp",
        "input_latent_shape": [1, 4, 64, 64],
        "algorithm": "12-step DPM-Solver++ (2M) with Karras sigmas on UFIX16 quantized UNet",
        "prompt_contract": "MANDATORY: Pass empty string '' for restoration. Text prompts cause hallucinations.",
        "measured_performance": {
            "pipeline_internal_ms": 13300.0,
            "wall_clock_per_launch_ms": 25200.0,
            "context_load_overhead_ms": 11900.0,
            "global_psnr_db": 18.87,
            "hole_psnr_db": 13.46,
            "ssim": 0.6048,
            "lpips": 0.2815,
            "active_power_w": 2.87,
            "thermal_delta_c": "+12.0 °C"
        }
    },
    "sd_inefficient": {
        "canonical_name": "Stable Diffusion 1.5 RePaint (Legacy 20-Step Euler Baseline)",
        "role": "Baseline comparative benchmark evaluating iterative stochastic schedule",
        "runner_binary": "sd_qidk_runner_inefficient",
        "target_hardware": "Qualcomm Hexagon HTP v79 NPU",
        "staging_dir": "/data/local/tmp/sd_runtime",
        "algorithm": "20-step Euler stochastic sampling schedule on 4-channel standard UNet",
        "measured_performance": {
            "wall_clock_per_launch_ms": 50930.0,
            "active_power_w": 2.65,
            "active_energy_j": 134.97,
            "thermal_delta_c": "+30.0 °C"
        }
    }
}

# ==============================================================================
# 3. CRITICAL HARDWARE QUIRKS & SYSTEM CAVEATS
# ==============================================================================
HARDWARE_AND_ENVIRONMENT_QUIRKS: Dict[str, str] = {
    "ANDROID_15_BIONIC_LINKER_COLLISION": (
        "CRITICAL: Do NOT export /system/lib64 or /vendor/lib64 into LD_LIBRARY_PATH "
        "when running on-device binaries under Android 15 (Bionic libc). Doing so triggers "
        "a fatal symbol collision on 'libbinder_ndk.so', aborting the FastRPC process. "
        "Always use isolated paths: LD_LIBRARY_PATH=/data/local/tmp/lama/lib:/data/local/tmp/lama."
    ),
    "STABLE_DIFFUSION_PROMPT_HALLUCINATION": (
        "CRITICAL: When executing sd_qidk_runner_inpaint, NEVER supply descriptive text prompts "
        "like 'high quality clean photo restoration'. The 860M UNet interprets text conditioning "
        "literally, hallucinating textual characters, logos, and faces into the inpainted hole. "
        "Always pass an empty string: ./sd_qidk_runner_inpaint ''."
    ),
    "BATTERY_CURRENT_SYSFS_LIMITATION": (
        "The sysfs node /sys/class/power_supply/battery/current_now on the Snapdragon 8 Elite "
        "QIDK development board updates only once every ~1.5s and hovers within ±5 mA even under "
        "100% CPU/NPU saturation. Raw sysfs cannot measure sub-second GAN bursts. For true active "
        "wattage numbers, use Qualcomm Qprof hardware profiler rails (port 62472)."
    ),
    "DSP_COLD_START_AND_INIT_CACHE": (
        "SNPE on Hexagon HTP v79 incurs an ~8-second graph compilation overhead on cold launch "
        "for LaMa and AOT-GAN. Pre-compiling the SNPE HTP init cache (via 'python benchmark.py --init-cache' "
        "or scripts/fresh_benchmark/make_init_cache.sh) reduces cold start to 0.47s (LaMa) and 0.57s (AOT-GAN), "
        "producing bit-identical tensors across all samples."
    ),
    "FAST_RPC_FILE_PERMISSIONS": (
        "Android app sandbox prevents the Hexagon CDSP daemon user from reading/writing raw tensors "
        "created by the app UID. Any file staged in /data/local/tmp must explicitly have permissions "
        "set via file.setReadable(true, false) and file.setWritable(true, false) or 'chmod 777'."
    ),
    "COMPOSITING_GROUND_TRUTH_PRESERVATION": (
        "Inpainting models occasionally introduce faint color drift across untouched pixels. Ground-truth "
        "preservation MUST be enforced via mathematical alpha compositing: "
        "I_out = (M * I_model) + ((1 - M) * I_original), where M is normalized [0.0, 1.0] hole weight."
    ),
    "SCREEN_ROTATION_LIFECYCLE": (
        "In MainActivity.kt, loadDefaultSample() must ONLY execute when savedInstanceState == null. "
        "Otherwise, screen rotation destroys the user's painted strokes and resets the canvas."
    )
}

# ==============================================================================
# 4. REPOSITORY FILE & DIRECTORY ONTOLOGY
# ==============================================================================
FILE_REGISTRY: Dict[str, str] = {
    # Root Level Control & Orchestration
    "main.py": "Master CLI entrypoint: routes commands to Web UI, visuals, benchmarks, or decision router",
    "benchmark.py": "Unified master benchmark orchestrator with interactive terminal menu and CLI dispatch",
    "context.py": "AI-to-AI context handover, architectural schema, and system specifications (this file)",
    "SETUP.md": "Comprehensive technical onboarding, first-time setup guide, dependencies, and hardware docs",
    "function.md": "Complete terminal reference manual containing all CLI, ADB, and script commands",
    "README.md": "Project landing page with executive summary, benchmark tables, and architecture diagrams",
    "pyproject.toml": "PEP 518 / 621 project configuration with core and dev dependencies",
    "uv.lock": "Deterministic lockfile pinning exact versions for reproducible Python 3.12 installs",
    "input_list_102.txt": "Staged file path list formatted for snpe-net-run on-device batch sweep",
    "ROUTER_AUDIT_AND_PIPELINE_REPORT.md": "Empirical calibration and dynamic range audit of decision router",

    # Core Python Modules (src/)
    "src/router.py": "Multi-modal decision router engine, image feature extractors, and live FastRPC execution bridge",
    "src/auto_masking.py": "Sub-40ms GrabCut edge snapper, distance transform sampler, and tensor format converters",
    "src/app_gui.py": "Interactive Gradio Web Canvas UI with real-time on-device Hexagon NPU telemetry HUD",

    # Model Weights & DLC Containers (models/)
    "models/Migan/migan_htp_v79.dlc": "Optimized MIGAN DLC quantized for Qualcomm Hexagon HTP v79",
    "models/LamaDilated/lama_dilated.dlc": "LaMa Dilated Fast Fourier Convolution DLC container",
    "models/AOT-GAN/aotgan.dlc": "AOT-GAN stacked dilated bottleneck DLC container",
    "models/dlc-info.txt": "Quantization profiles, tensor layer dimensions, and execution parameters",

    # Stable Diffusion Native Pipeline & Models (StableDiffusion/)
    "StableDiffusion/CMakeLists.txt": "Android NDK / QNN CMake build configuration for native SD inpainting runners",
    "StableDiffusion/sd_runner_inpaint.cpp": "C++ native QIDK inpainting runner executing 12-step DPM-Solver++ on Hexagon HTP (unet.bin / graph_wlqbe2kd)",
    "StableDiffusion/sd_runtime/": "Staged runtime bundle pushed to /data/local/tmp/sd_runtime/ containing runner binaries, QNN libraries, and serialized models",

    # Benchmarking & Diagnostic Scripts (scripts/)
    "scripts/fresh_benchmark/run_fresh_benchmark.py": "Measured on-device benchmark harness (monotonic clock, cooldown barrier, thermals)",
    "scripts/fresh_benchmark/make_figures.py": "Generates Pareto, radar, telemetry, and distribution figures from fresh runs",
    "scripts/fresh_benchmark/make_init_cache.sh": "Compiles SNPE HTP init cache on device for instant cold-starts",
    "scripts/benchmark_sd_models.py": "Dedicated Stable Diffusion 1.5 comparative profiler (12-step DPM vs 20-step Euler)",
    "scripts/diagnose_migan_hardware.py": "MIGAN hardware diagnostics and layer-by-layer HTP v79 cycle audit",
    "scripts/generate_sd_comparison_figures.py": "Generates 6 publication-grade figures comparing dual SD runners",
    "scripts/generate_presentation_visuals.py": "Generates the 8 master publication figures (300 DPI)",
    "scripts/run_two_phase_batch_benchmark.py": "Decoupled 102-sample batch benchmark runner",
    "scripts/prep_102_benchmark.py": "Generates raw float32 input tensors and ground truth from dataset",
    "scripts/evaluation_suite.py": "Offline evaluation suite calculating PSNR, SSIM, LPIPS, and boundary quality",
    "scripts/thermal_logger.sh": "Background 1 Hz PMIC and thermal zone logging daemon",
    "scripts/render_inpaint_comparisons.py": "Renders multi-model qualitative comparison strips across benchmark samples",
    "scripts/generate_benchmark_deck.py": "Builds automated 9-slide PowerPoint presentation deck",

    # Native Android Application (qidk-inpaint-app/)
    "qidk-inpaint-app/app/src/main/java/.../MainActivity.kt": "Android UI activity, lifecycle management, sample picker, and button wiring",
    "qidk-inpaint-app/app/src/main/java/.../engine/OnDeviceProcessDriver.kt": "ProcessBuilder driver executing snpe-net-run and SD runners via FastRPC",
    "qidk-inpaint-app/app/src/main/java/.../ui/InpaintCanvasView.kt": "Custom touch canvas supporting pinch-to-zoom, pan, brush, and 2-tap/drag box",
    "qidk-inpaint-app/app/src/main/java/.../utils/GrabCutEngine.kt": "Sub-40ms native OpenCV GrabCut silhouette extractor",
    "qidk-inpaint-app/app/src/main/java/.../engine/DeepMaskRefiner.kt": "Multi-modal mask refiner fusing MediaPipe segmenter and Fast Guided Filter",
    "qidk-inpaint-app/app/src/main/java/.../engine/MediaPipeSegmenter.kt": "Interactive segmentation engine using magic_touch.tflite asset",
    "qidk-inpaint-app/app/src/main/java/.../engine/ModelPreloadManager.kt": "Preloads DLC containers into tmpfs RAM on app boot",

    # Presentation Deliverables (Presentation/)
    "Presentation/ESW_Image_Inpainting_Progress.pptx": "Master technical progress presentation slide deck",
    "Presentation/Image_Inpainting_QIDK_Progress.pdf": "Exported presentation slides in PDF format",
    "Presentation/smart_object_detection_benchmark.pptx": "Presentation deck covering smart object detection and mask refinement",
    "Presentation/test_slide_8.pptx": "Executive summary slide with high-resolution visual proof"
}


# ==============================================================================
# 5. CONTEXT ACCESSORS & CLI INTERFACE
# ==============================================================================
def get_context() -> Dict[str, Any]:
    """Returns the complete structured context dictionary."""
    return {
        "metadata": PROJECT_METADATA,
        "models": MODEL_REGISTRY,
        "quirks": HARDWARE_AND_ENVIRONMENT_QUIRKS,
        "file_map": FILE_REGISTRY
    }


def print_summary():
    """Prints a structured human/AI readable overview."""
    print("=" * 80)
    print("🤖 QUALCOMM SNAPDRAGON 8 ELITE INPAINTING — AI HANDOVER CONTEXT")
    print("=" * 80)
    print(f"Project       : {PROJECT_METADATA['project_name']}")
    print(f"Version       : {PROJECT_METADATA['version']}")
    print(f"Target SoC    : {PROJECT_METADATA['target_platform']['soc']}")
    print(f"Acceleration  : {PROJECT_METADATA['target_platform']['npu_core']}")
    print(f"Upstream Repo : {PROJECT_METADATA['upstream_repository']}")
    print("\n[EVALUATED ARCHITECTURES]")
    for key, info in MODEL_REGISTRY.items():
        name = info["canonical_name"]
        hw = info.get("target_hardware", "NPU")
        polarity = info.get("mask_polarity", {}).get("label", "Standard")
        perf = info.get("measured_performance", {})
        latency = perf.get("steady_latency_npu_ms", perf.get("wall_clock_per_launch_ms", "N/A"))
        print(f"  • {key.upper():<16} : {name}")
        print(f"    Hardware: {hw} | Polarity: {polarity} | Latency: {latency} ms")

    print("\n[CRITICAL HARDWARE RULES]")
    for k, text in HARDWARE_AND_ENVIRONMENT_QUIRKS.items():
        print(f"  • {k}: {text[:100]}...")

    print("\n[PRIMARY ENTRYPOINTS]")
    print("  • python main.py [gui | benchmark | visuals | route]")
    print("  • python benchmark.py [interactive menu or --fresh, --sd, --figures]")
    print("  • python context.py [--summary | --json | --models | --quirks | --file-map]")
    print("=" * 80)


def main():
    if len(sys.argv) == 1 or "--summary" in sys.argv:
        print_summary()
        return

    if "--json" in sys.argv:
        print(json.dumps(get_context(), indent=2))
        return

    if "--models" in sys.argv:
        print(json.dumps(MODEL_REGISTRY, indent=2))
        return

    if "--quirks" in sys.argv:
        print(json.dumps(HARDWARE_AND_ENVIRONMENT_QUIRKS, indent=2))
        return

    if "--file-map" in sys.argv:
        print(json.dumps(FILE_REGISTRY, indent=2))
        return

    print("Unrecognized flag. Options: --summary, --json, --models, --quirks, --file-map")


if __name__ == "__main__":
    main()
