# Terminal Command Reference Manual (`function.md`)
## Complete CLI & Terminal Commands for Qualcomm Snapdragon 8 Elite Image Inpainting

This document is an exhaustive reference manual of all terminal commands available in the repository. It covers everything from top-level entrypoints down to raw ADB shell execution on Qualcomm Hexagon NPU and Adreno GPU.

---

## 1. Master Control CLI (`main.py`)

The unified master entrypoint handles top-level execution:

```bash
# Display general help banner and detected ADB hardware
python3 main.py --help

# 1. Launch Interactive Gradio Web Canvas
python3 main.py gui --host 0.0.0.0 --port 7860
python3 main.py gui --share                  # Create public Gradio tunnel link

# 2. Execute Multi-Modal Decision Router
python3 main.py route -i Benchmark/input_102/image/001.png -m Benchmark/input_102/mask/001.png
python3 main.py route -i sample.png -m mask.png --json decision.json

# 3. Generate 8 Publication-Grade Presentation Figures (300 DPI)
python3 main.py visuals

# 4. Launch Unified Benchmark Suite (Interactive Menu)
python3 main.py benchmark

# 5. Execute Fresh Measured Benchmark directly via Master CLI
python3 main.py fresh-benchmark --lpips --sd
python3 main.py fresh-benchmark --configs migan_npu lama_npu
```

---

## 2. Unified Benchmark & Diagnostic Suite (`benchmark.py`)

The master benchmark orchestrator provides both an interactive terminal interface and granular CLI switches:

### Interactive Terminal Menu
```bash
python3 benchmark.py
# or explicitly:
python3 benchmark.py -m
python3 benchmark.py --menu
```

### Non-Interactive & Scriptable Commands
```bash
# Environment Verification Smoke Test
python3 benchmark.py --smoke-test

# Measured On-Device Benchmark Sweep (Monotonic device clock & thermal trace)
python3 benchmark.py --fresh
python3 benchmark.py --fresh --lpips              # Include LPIPS deep perceptual loss
python3 benchmark.py --fresh --lpips --sd         # Full sweep including Stable Diffusion 1.5
python3 benchmark.py --fresh --configs migan_npu lama_npu aotgan_npu

# Stable Diffusion Head-to-Head Comparison & Telemetry Profiler
python3 benchmark.py --sd                         # Evaluates 12-step DPM vs 20-step Euler
python3 benchmark.py --sd --limit 10              # Run on a 10-sample subset

# Live Resilient Stable Diffusion Runner (tqdm progress bar, auto-reconnect on wire pull, smart resume)
python3 benchmark.py --sd-live                    # Full live sweep across 102 samples
python3 benchmark.py --sd-live --limit 5          # Quick 5-sample verification
python3 benchmark.py --sd-live --force-rerun      # Ignore cache and re-run all
python3 scripts/run_sd_benchmark_live.py --limit 10  # Standalone direct invocation

# MI-GAN Hardware Diagnostics & DSP Cycle Audit
python3 benchmark.py --migan-diag

# Decoupled Two-Phase Batch Sweep (102 evaluation samples)
python3 benchmark.py --two-phase

# SNPE HTP Init Cache Builder (Sub-500ms graph init)
python3 benchmark.py --init-cache

# Generate Benchmark Visual Figures (300 DPI)
python3 benchmark.py --figures all                # All figures (Fresh, SD, Publication)
python3 benchmark.py --figures fresh              # Fresh Pareto, Radar, Telemetry, CDF
python3 benchmark.py --figures sd                 # SD Comparative Figures
python3 benchmark.py --figures publication        # 8 Master Publication Figures

# Generate Presentation Slide Deck & Qualitative Strips
python3 benchmark.py --deck                       # Build PPTX deck & inpaint comparison strips
```

---

## 3. Direct Benchmark & Analysis Scripts (`scripts/`)

All standalone scripts can also be invoked directly:

### Fresh Measured Benchmark Harness
```bash
# Run measured latency and quality sweep
python3 scripts/fresh_benchmark/run_fresh_benchmark.py --lpips --sd

# Generate fresh benchmark presentation figures
python3 scripts/fresh_benchmark/make_figures.py \
    --results Benchmark/output/fresh_benchmark_2026-10-04 \
    --ds Benchmark/input_102 \
    --out Benchmark/output/fresh_benchmark_2026-10-04/figures

# Build SNPE HTP init cache on device
bash scripts/fresh_benchmark/make_init_cache.sh
```

### Stable Diffusion Profiler & Visualizations
```bash
# Run SD profiler
python3 scripts/benchmark_sd_models.py
python3 scripts/benchmark_sd_models.py --limit 5

# Generate SD comparative radar, Pareto, and telemetry charts
python3 scripts/generate_sd_comparison_figures.py
```

### Hardware Diagnostics & Profiling
```bash
# Deep-dive MI-GAN diagnostics on Hexagon NPU
python3 scripts/diagnose_migan_hardware.py

# Run sub-process cold-start granular trace
python3 scripts/profile_granular_trace.py

# Background thermal & PMIC power logging daemon (1 Hz)
bash scripts/thermal_logger.sh /data/local/tmp/lama/telemetry.csv 1.0
```

### Two-Phase Batch Benchmark & Preprocessing
```bash
# Prepare 102 benchmark dataset from raw inputs
python3 scripts/prep_102_benchmark.py

# Execute batch runner across models
python3 scripts/run_two_phase_batch_benchmark.py

# Run standalone perceptual evaluation on predictions
python3 scripts/evaluation_suite.py \
    --pred_dir Benchmark/output/reconstructions/migan_npu \
    --gt_dir Benchmark/input_102/ground_truth \
    --output_csv Benchmark/output/migan_eval.csv
```

### Executive Slides & Inpainting Strips
```bash
# Render high-resolution side-by-side inpaint comparisons
python3 scripts/render_inpaint_comparisons.py

# Generate automated PowerPoint presentation deck
python3 scripts/generate_benchmark_deck.py
```

---

## 4. Native Android APK Commands (`qidk-inpaint-app/`)

Commands for building, deploying, and debugging the native Android 15 application:

```bash
# Navigate to Android project root
cd qidk-inpaint-app

# 1. Clean & Build Debug APK
./gradlew clean
./gradlew assembleDebug

# 2. Run Unit Tests (GrabCut & Overhaul Pipeline)
./gradlew test

# 3. Install APK on Snapdragon 8 Elite Device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 4. Grant Runtime Permissions (Scoped Storage / Camera)
adb shell pm grant com.qualcomm.qidk.inpaint android.permission.READ_MEDIA_IMAGES
adb shell pm grant com.qualcomm.qidk.inpaint android.permission.CAMERA

# 5. Launch Application Directly via ADB
adb shell am start -n com.qualcomm.qidk.inpaint/.MainActivity

# 6. Force-Stop Application
adb shell am force-stop com.qualcomm.qidk.inpaint

# 7. Stream Real-Time App Logcat (Filtering Inpainting Engine)
adb logcat -v time -s "InpaintCanvasView" "OnDeviceProcessDriver" "MainActivity" "ModelPreloadManager" "DeepMaskRefiner"
```

---

## 5. Direct On-Device Shell Commands (Hexagon NPU & Adreno GPU)

Commands executed directly inside `adb shell` on the Snapdragon 8 Elite device:

### Setting Up Environment Variables (Android 15 Safe)
```bash
adb shell
# CRITICAL: Never include /system/lib64 or /vendor/lib64 in LD_LIBRARY_PATH (Bionic crash)
export LD_LIBRARY_PATH=/data/local/tmp/lama/lib:/data/local/tmp/lama
export ADSP_LIBRARY_PATH='/data/local/tmp/lama/dsp/lib;/data/local/tmp/lama/dsp;/dsp'
export PATH=$PATH:/data/local/tmp/lama
```

### Running Feed-Forward Models with `snpe-net-run`
```bash
cd /data/local/tmp/lama

# 1. MI-GAN on Hexagon NPU (HTP v79 DSP)
./snpe-net-run \
    --container migan_htp_v79.dlc \
    --input_list live_input.txt \
    --output_dir app_live_output \
    --use_dsp

# 2. LaMa Dilated on Hexagon NPU (HTP v79 DSP)
./snpe-net-run \
    --container lama_dilated.dlc \
    --input_list live_input.txt \
    --output_dir app_live_output \
    --use_dsp

# 3. LaMa Dilated on Adreno 830 GPU
./snpe-net-run \
    --container lama_dilated.dlc \
    --input_list live_input.txt \
    --output_dir app_live_output \
    --use_gpu

# 4. AOT-GAN on Hexagon NPU (HTP v79 DSP)
./snpe-net-run \
    --container aotgan.dlc \
    --input_list live_input.txt \
    --output_dir app_live_output \
    --use_dsp

# 5. AOT-GAN on Adreno 830 GPU
./snpe-net-run \
    --container aotgan.dlc \
    --input_list live_input.txt \
    --output_dir app_live_output \
    --use_gpu
```

### Building & Staging Native Stable Diffusion Inpainting Runner
```bash
# 1. Compile native QIDK runner with Android NDK & Qualcomm AI Direct SDK (QAIRT)
cd StableDiffusion
cmake -B build -S . \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-34
cmake --build build --target sd_qidk_runner_inpaint

# 2. Push compiled runner and optimized unet.bin model container to device
adb push sd_runtime/sd_qidk_runner_inpaint /data/local/tmp/sd_runtime/
adb push sd_runtime/models/unet.bin /data/local/tmp/sd_runtime/models/unet.bin
adb shell "chmod +x /data/local/tmp/sd_runtime/sd_qidk_runner_inpaint"
```

### Running Stable Diffusion 1.5 on Hexagon NPU
```bash
cd /data/local/tmp/sd_runtime
export LD_LIBRARY_PATH=/data/local/tmp/sd_runtime:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH='/data/local/tmp/sd_runtime;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'

# 1. Optimized 12-Step DPM-Solver++ Runner (Empty prompt prevents hallucinations)
rm -f sd_output.png
./sd_qidk_runner_inpaint ""

# 2. Legacy 20-Step Euler RePaint Runner
rm -f sd_output.png
./sd_qidk_runner_inefficient ""
```

---

## 6. Qualcomm Qprof Hardware Profiler Commands

Commands to run hardware-level telemetry and PMIC profiling:

```bash
# 1. Forward gRPC daemon port over ADB
adb forward tcp:62472 tcp:62472

# 2. Launch Qmonitor gRPC Server on Snapdragon 8 Elite device
adb shell
export QMONITOR_BACKEND_LIB_PATH=/vendor/qprof/backends
export LD_LIBRARY_PATH=/apex/com.android.i18n/lib64:/apex/com.android.runtime/lib64:/apex/com.android.art/lib64:/system/lib64:/vendor/lib64:/vendor/qprof/libs:$LD_LIBRARY_PATH
/vendor/bin/qmonitor-grpc-server -n 127.0.0.1 -p 62472

# 3. Check Thermal Zones on Device
adb shell "cat /sys/class/thermal/thermal_zone*/type"
adb shell "cat /sys/class/thermal/thermal_zone*/temp"

# 4. Check Hexagon DSP FastRPC Driver Node
adb shell "ls -l /dev/fastrpc-cdsp"
```

---

## 7. ADB File Staging & Transfer Commands

Commands for pushing and pulling models, datasets, and benchmark outputs:

```bash
# Push DLC models to device
adb push models/Migan/migan_htp_v79.dlc /data/local/tmp/lama/
adb push models/LamaDilated/lama_dilated.dlc /data/local/tmp/lama/
adb push models/AOT-GAN/aotgan.dlc /data/local/tmp/lama/

# Push sample images to device gallery / app accessible storage
adb push Benchmark/input_102/image/ /sdcard/Pictures/Inpainting102/

# Set read/write/executable permissions on all staged files
adb shell "chmod -R 777 /data/local/tmp/lama /data/local/tmp/sd_runtime"

# Pull benchmark reconstructions back to host
adb pull /data/local/tmp/lama/output_migan_npu Benchmark/output/reconstructions/
adb pull /data/local/tmp/lama/output_lama_npu Benchmark/output/reconstructions/

# Pull raw PMIC and thermal trace logs
adb pull /data/local/tmp/lama/telemetry.csv Benchmark/output/telemetry/
```

---

## 8. AI Handover & Context CLI (`context.py`)

Commands for AI agents and automated scripts to inspect project architecture:

```bash
# Print high-level project summary and current status
python3 context.py --summary

# Dump complete machine-readable architecture and file registry as JSON
python3 context.py --json

# List model specifications, tensor dimensions, and polarities
python3 context.py --models

# Display critical Qualcomm hardware and runtime quirks
python3 context.py --quirks

# Display complete repository file-to-role mapping
python3 context.py --file-map
```
