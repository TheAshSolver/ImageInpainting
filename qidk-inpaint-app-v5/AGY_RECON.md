# AGY_RECON.md — System Reconnaissance & Architecture Baseline

**Target System**: Qualcomm Snapdragon 8 Elite Development Kit (SM8750P / APQ8750, Board `sun`, Serial `8f27557f`)  
**OS Version**: Android 15 (API 35, Build Release 15, Kernel 6.6.56, SELinux: Permissive)  
**Target Project**: `qidk-inpaint-app-v5` (`com.qualcomm.qidk.inpaint.v5`)  
**Workspace**: `/home/tarsh/Desktop/college/ImageInpainting`  

---

## 1. Canvas Architecture & Coordinate Mapping
- **Class**: `com.qualcomm.qidk.inpaint.ui.InpaintCanvasView` (`app/src/main/java/com/qualcomm/qidk/inpaint/ui/InpaintCanvasView.kt:17-901`), custom Android `View`.
- **View <-> Bitmap Mapping**:
  - `updateTransformMatrices()` (lines 430-446) computes `bitmapToViewMatrix` using `minOf(vw / 512f, vh / 512f)` (uniform `fitCenter` scaling + centering translation $(dx, dy)$) concatenated with pinch `zoomScale` centered at $(vw/2, vh/2)$ and pan offsets $(panX, panY)$.
  - Inverted to `viewToBitmapMatrix = Matrix()`.
  - Touches mapped via `mapTouchToBitmap()` (line 448) and `screenToCanonical()` (line 457), clamping coordinates strictly to $[0, 511]$.
- **Rendering in `onDraw`**:
  - Line 779-780: `canvas.save()` followed by `canvas.concat(bitmapToViewMatrix)`.
  - Line 782: Destination is `canonicalRect = Rect(0, 0, 512, 512)`. Source bitmap and `maskBitmap` are drawn directly into canonical space.
- **Zoom / Pan**:
  - `zoomScale: Float` (default 1.0f, clamped 1.0f - 5.0f).
  - `panX: Float`, `panY: Float` managed via `ScaleGestureDetector` and `GestureDetector` in `onTouchEvent`.
- **Modes**:
  - `Mode.CIRCLE_SELECT`, `Mode.BRUSH`, `Mode.BOUNDING_BOX`.
  - `SegmentationEngine.AUTO_GRABCUT`, `SegmentationEngine.AI_MAGIC_TOUCH`, `SegmentationEngine.EXACT_SELECT`.

---

## 2. Smart Touch & Segmentation
- **Classes**:
  - `AutoGrabCutEngine.kt` (`app/src/main/java/com/qualcomm/qidk/inpaint/segmentation/AutoGrabCutEngine.kt:33-371`).
  - `MediaPipeSegmenter.kt` (`.../segmentation/MediaPipeSegmenter.kt`).
  - `MagicTouchEngine.kt` (`.../utils/MagicTouchEngine.kt`).
  - `GrabCutEngine.kt` (`.../utils/GrabCutEngine.kt`).
- **Disjoint Blobs & Edge Leakage Root Cause**:
  - When GrabCut runs on complex textures (e.g. water surface with hull shadows), GMM color modeling assigns `GC_PR_FGD` to disjoint reflections across the bounding crop.
  - If components are not pruned strictly to the gesture medial axis distance transform, large disjoint blobs survive.
  - Replaced by `SmartTouchSegmenter.kt` (`qidk.fix.SmartTouchSegmenter`), which extracts the medial axis seed (top 30% of distance transform for lasso, 4px eroded stroke for scribble), enforces lateral background sinks (`GC_BGD` outer border), downscales the crop to $\le 256\text{ px}$, and retains solely the primary connected component.

---

## 3. OpenCV Java & Native Setup
- **Java API**:
  - Artifacts: `com.github.jenly1314.WeChatQRCode:opencv:2.2.0` and `com.github.jenly1314.WeChatQRCode:opencv-armv64:2.2.0` in `app/build.gradle.kts:50-51`.
  - Provides full `org.opencv.*` (`Imgproc`, `Core`, `Photo`, `CvType`, `Mat`, `Point`, `Rect`, `Scalar`).
- **Native Libraries**:
  - `System.loadLibrary("opencv_java4")` initialized in `AutoGrabCutEngine.kt:39`.
  - `System.loadLibrary("snpe_inpaint_jni")` loaded in `SnpeNativeBridge.kt:9`.
  - Classical C++ library `libqidk_classical.so` added in `app/src/main/cpp/CMakeLists.txt` exporting `Java_qidk_fix_ClassicalNative_criminisi` and `Java_qidk_fix_ClassicalNative_biharmonic`.

---

## 4. Inference Dispatch & Classical Execution
- **Dispatch Flow**:
  - `OnDeviceProcessDriver.executeInference(...)` in `app/src/main/java/com/qualcomm/qidk/inpaint/engine/OnDeviceProcessDriver.kt:70-854`.
  - **Classical Algorithms**: Telea and Navier-Stokes run via OpenCV `Photo.inpaint`. Criminisi and Biharmonic previously lacked the native `.so` linkage in the APK, leading to `UnsatisfiedLinkError`. Now routed to `ClassicalInpainter.kt` with a 5px elliptical dilation.
  - **GAN Models**: Dispatched via TCP loopback socket `127.0.0.1:55555` to resident `snpe_resident_daemon` or fallback to `snpe-net-run`.
  - **Stable Diffusion**: Dispatched via subprocess shell `/system/bin/sh -c` to `/data/local/tmp/sd_runtime/sd_qidk_runner_inpaint`.

---

## 5. FastRPC Resident Daemon
- **Source**: `/home/tarsh/Desktop/college/ImageInpainting/snpe_resident_daemon.cpp`.
- **Protocol**: TCP port 55555. Commands: `PING`, `INFER <key> <img.raw> <mask.raw> <out.raw>`, `STATUS`, `STOP`. Replies: `PONG`, `OK <ms>`, `ERROR <msg>`, `BYE`.
- **Runtimes & Models**:
  - DSP (Hexagon HTP v79 NPU): `migan`, `lama`, `aotgan`.
  - GPU/CPU: Added via `snpe_resident_daemon.diff` lazy-loading `migan_gpu`, `migan_cpu`, `lama_gpu`, `lama_cpu`, `aotgan_gpu`, `aotgan_cpu`.
- **Startup**: `ModelPreloadManager.kt` launches daemon via background shell in `/data/local/tmp/lama/`.

---

## 6. Stable Diffusion Runner & Argument Bug
- **Source**: `/home/tarsh/Desktop/college/ImageInpainting/StableDiffusion/sd_runner_inpaint.cpp`.
- **Argument Mismatch Confirmed**:
  - Line in runner: `int numSteps = (argc > 2) ? std::atoi(argv[2]) : 12;`
  - The Android app passed `positivePrompt` in `argv[1]` and `negativePrompt` in `argv[2]`.
  - `std::atoi(negativePrompt)` evaluated to 0 (clamped to 2 steps), causing SD to run 2 denoising steps and return noisy output regardless of prompt!
- **Fix**: Replaced by `sd_args.hpp` with explicit flags (`--prompt`, `--neg`, `--steps`, `--cfg`, `--seed`), passing user text safely via `$SD_PROMPT` and `$SD_NEG` environment variables to prevent shell injection.
- **Binaries in `/data/local/tmp/sd_runtime`**: `sd_qidk_runner_inpaint`, `sd_qidk_runner_inpaint_20step_cfg20`, `unet.bin` (883 MB), `vocab.json`, `merges.txt`.
- **`text_embeds_fixed.raw`**: Does not exist; dynamic NPU text encoding is active.

---

## 7. Contextual Bandit Router & Feature Manifest
- **Policy File**: `/home/tarsh/Desktop/college/ImageInpainting/artifacts/production_ccb_router.json`.
- **Top-Level Keys**:
  - `version`: `"4.0.0-production-ccb"`
  - `mathematical_framework`: `"Constrained Contextual Bandit (CCB)"`
  - `context_dimension`: 20
  - `feature_names`: exactly 20 features:
    - 10 Image: `grad_canny_density`, `grad_sobel_total_mag`, `fft_high_freq_power_ratio`, `glcm_contrast_0`, `dwt_detail_to_approx_ratio`, `color_lab_delta_e_variance`, `color_temperature_proxy`, `lbp_uniform_entropy`, `contour_curvature_mean`, `structural_isophote_coherence`.
    - 5 Mask: `mask_area_ratio`, `mask_compactness`, `mask_max_inscribed_radius`, `mask_hole_count`, `mask_solidity`.
    - 5 Telemetry: `T_silicon`, `dT_dt`, `CPU_util`, `GPU_util`, `BatteryLevel`.
  - `operating_profiles`: `mode_1_max_quality`, `mode_2_balanced_pareto`, `mode_3_thermal_contention_safe`.
  - `action_models`: `telea`, `navier_stokes`, `criminisi`, `migan`, `lama`, `aotgan`, `sd_15`.
- **Dataset**: `/home/tarsh/Desktop/college/ImageInpainting/artifacts/dataset_100_features_6000.csv` (6,000 samples).
- **Feature Extractor**: `/home/tarsh/Desktop/college/ImageInpainting/scripts/benchmarks/extract_100_unbiased_features.py`.

---

## 8. Sub-Pixel Super-Resolution & Handbuilt Engine
- **Tarsh Reconstructor**: FOUND at `/home/tarsh/Desktop/personal/image_processing/meow.py`.
  - Mathematical formulation: 4-quadrant edge-padded sub-kernel accumulation ($sub_{TL}, sub_{TR}, sub_{BL}, sub_{BR}$ with weight 4 on center and weights 2/1 on cardinal/diagonal neighbors) interleaved into a $2W \times 2H$ grid, followed by full 8-neighbor Laplacian sharpening.
  - Ported line-by-line to Kotlin in `TarshSrEngine.kt`.
- **Benchmark Recipe**: Interleaved 75% structural checkerboard mask evaluated in `scripts/benchmarks/benchmark_super_resolution_upscaling.py`.
