# SUPER_STUDIO_REPAIR_GUIDE — `qidk-inpaint-app-v5`, five defects, Snapdragon 8 Elite (SM8750P "sun", serial `8f27557f`, Android 15)

Everything below was derived from the code in `for_ai.zip` and checked on the host (g++ 15, Kotlin 2.2.20 at language level 1.9, NDK 28.2 for arm64, aapt2 35).
**Nothing could be run on the device in the session that produced this bundle**: every claim marked *[device]* is what `tools/verify_super_studio.sh` must establish.

| Host-verified | Result |
|---|---|
| NSDT kernels, `tools/native/test_sr_core.cpp` | 44,139 checks pass; `libqidk_sr.so` links for arm64 without warnings |
| `sd_img2img.hpp`, `tools/native/test_sd_img2img.cpp` | 243 checks pass |
| JUnit (`./gradlew testDebugUnitTest`) | 70 tests pass (46 new + 24 existing; one obsolete Hann-window test removed with the lattice code) |
| Whole patched Kotlin tree | compiles against android.jar + AndroidX jars (stubs only for OpenCV/MediaPipe/Material/R) |
| Resources | `aapt2 compile` + `aapt2 link` clean (Material attributes stood in by stubs) |

---------------------------------------------------------------------------------------------------------------------------------

## 1. Neural super-resolution (defect 1)

### 1.1 What the code in the zip really does, and why every model failed

The old `NeuralSrEngine` builds, per 2x pass, a 1024x1024 canvas whose pixel (2x,2y) is the true source pixel and whose other three
pixels are a bilinear pre-fill, with a mask that marks **75 % of the pixels as holes in a stride-2 lattice**. Four 512x512 quadrants go
to the inpainting network and the output is feathered with a 32 px Hann ramp. Measured on the example outputs in `upscale_example/`
(phase analysis of the sub-pixel grid against the reconstructed canvas):

| Engine | Measured | Mechanism |
|---|---|---|
| MIGAN | holes equal the pre-fill (phase means 11.85/11.87/11.79, MAE vs canvas 59.5) | the MIGAN graph composites `image*blur(maxpool3x3(keep)) + out*(1-...)`; a 3x3 max-pool of the *keep* mask is 1 everywhere when keep pixels are every second pixel, so the composite returns the input: **output == input** |
| AOT-GAN | output equals the canvas on 99 % of the holes | `isInverted = migan || aotgan` sends hole=0; the Inpaint screen (validated on device) sends **hole=1 for LaMa and AOT-GAN, keep=1 only for MIGAN**, all in [0,1]. With the wrong polarity the network "inpaints" the 25 % anchors and passes the rest through |
| LaMa | hole phases ~193 grey, roughness 64 (the "screen door") | LaMa zeroes the hole content inside its graph, so the bilinear pre-fill is invisible; a 75 % stride-2 hole pattern is far outside the large-hole training distribution, the result is a noise field, then the anchors are restored on top: a mesh |
| earlier builds | dark / blown-out meshes | `[-1,1]` normalisation for MIGAN/LaMa while the Inpaint path (which works) uses `[0,1]` |
| SD latent SR | distortion | see 3.1 |

Conclusion: LaMa, MIGAN and AOT-GAN are **inpainters**. Asking them to interpolate between isolated pixels is ill-posed and
out of distribution; no polarity or scaling fix makes the lattice work. The root-cause fix is a different formulation.

### 1.2 Replacement: Null-Space Detail Transfer (NSDT)

Per 2x pass (all coordinates on the output grid):

* `U` = Keys cubic (a = -0.5) 2x up-sample, reflect-101 borders: the classical base. It carries every low-frequency fact.
* `A` = 2x2 box average, `P(Z) = Z - UP(A(Z))`: removes what the low-resolution grid can already represent. `A(P(Z)) ~ 0`, so detail cannot move geometry.
* Two **complementary checkerboards of 32 px cells** (50 % context, holes of a size the networks were trained for). Phase p's output is `O_p`; it differs from `U` only in the holes.
* Weight of phase p at a pixel: `w_p = clamp((s_p*e + m)/(2m), 0, 1)`, `e` = distance of the pixel centre to the nearest cell edge, `s_p = +1` inside a hole of phase p and `-1` outside, `m = 1`. For two complementary phases `w_0 + w_1 = 1` exactly.
* Detail `D = sum_p w_p * P(O_p - U)`, `X = clamp01(U + alpha * tau * tanh(D/tau))`, `tau = 0.08`, `alpha` = the "Detail strength" slider (default 0.6).

Why this cannot produce the old failures: there is no 1-px lattice (so no max-pool erasure and no stride-2 periodicity); the polarity/range
contract is **probed per model** before the first window (`ContractProbe`: a smooth gradient with a 160 px hole is sent under four candidate
contracts, the one under which the network really fills the hole wins; identity, NaN and black outputs are rejected); the output is
`U` plus a null-space residual, so a wrong network can at worst add bounded texture (soft clamp, per-window re-composition with a
smaller alpha when the mean detail exceeds 6 levels, per-window rejection of NaN / black / blown-out phases).

Measured on the host (`test_sr_core.cpp`): detail amplitude is uniform across the cell (max/min 1.40 at m = 1; 1.11 at m = 0, 1.77 at m >= 3,
because only the phase that owns a pixel carries real detail there); a Nyquist checker texture is flagged (diagonal Hadamard share 1.0 vs 0.33 for natural
texture); the LR-domain change of the added detail is < 2.5 levels rms.

**Quality gates** (`SrGates`, evaluated after every pass; a failing pass is **replaced by the classical base** and the reason is shown in the UI
and in `QIDK.SR` logs, so the result is never worse than bicubic):

| Gate | Threshold | Catches |
|---|---|---|
| rejected phases | <= 34 % | NaN, black or blown-out network output |
| mesh (diagonal Hadamard share of the detail) | <= 0.55 | the "screen door" |
| LR change of the detail | <= 2.5 levels rms | geometric drift |
| soft-clamp hits | <= 50 % | a network that fights the base |
| identity fraction / detail level | < 0.98 / >= 0.08 levels | MIGAN/AOT "output == input" (reported as NO_DETAIL) |
| early check after 6 windows | mesh, unstable | saves NPU time: a lattice is caught after 6 of 36 windows, not 36 |

Cost: windows of 512 px (core 384 + pad 64): 9 / 36 / 121 windows for 2x / 4x / 8x final passes, two inferences each (the 8x chain is 3 passes).
No full-size float planes exist anywhere; `TarshSrEngine` was also made streaming (bit-identical to the old algorithm, test `TarshCoreTest`)
because the old version needed ~400 MB of Java heap at the 4096 pass; the manifest now sets `android:largeHeap="true"`.

**Honest expectation** *[device]*: NSDT guarantees no mesh, no identity and no drift; it does not guarantee that inpainting networks invent *beautiful*
detail. `sr_metrics.py` reports `detail_added`, `lap_gain` and the gate verdicts so this can be judged per model. If detail is not
convincing, the architecture already has the two right extension points: (a) the SD img2img engine below, (b) a real SR network exported to
QNN as another `NeuralSrEngine` (same `SrKernels`/`NsdtPass`, only the tile model differs).

### 1.3 Files

`app/src/main/cpp/sr/{sr_core.h,sr_core.cpp,sr_jni.cpp}`, `CMakeLists.txt` (target `qidk_sr`: `jnigraphics`, 16 KB page size), `SrNative.kt` (JNI), `SrPlan.kt` (pure: windows, contract probe,
gates, `NsdtPass`), `NeuralSrEngine.kt` (`DaemonTileModel`, `NeuralSrEngine`, `SdSuperResEngine`), `TarshSrEngine.kt`, `SuperResolution.kt`.

JNI (class `qidk.fix.SrNative`, static): `sampleBase(Bitmap,int,int,int,float[])`, `holeMask(int,int,int,int,int,float[])`,
`compose(float[],float[]?,float[]?,int,int,int,int,int,float,float,float,int,int,int,int,Bitmap,int,int) -> double[14]`, `writeBase(...)`, `lrPsnr(Bitmap,Bitmap,int) -> double`.
Bitmaps must be ARGB_8888 (RGBA bytes in memory); the kernels return false/null otherwise.

---------------------------------------------------------------------------------------------------------------------------------

## 2. SR telemetry (defect 2) and the measurement errors behind it

The persistent ticker (`TelemetryBarView.kt`) is deleted (nothing referenced it). After **Upscale** (and after every Inpaint run) one card shows
`t_prep, t_infer, t_post, t_total, ΔT, active power, battery drain`, produced by one class (`RunTelemetry`, `PowerMeter.kt`) for both screens.

What was wrong in the zip's measurements (it applies to the router's inputs too, see 6):

* temperature = **max over every thermal zone** of the phone (battery, skin, charger included) -> now the zones of the silicon that did the work (NPU: `nsp*`/`cdsp`; GPU: `gpuss*`; CPU: `cpu*`), never battery/skin/xo/charger sensors;
* current = `in_current_pmih010x_ichg_fb_input` first, i.e. the **charger** feedback -> now `battery/current_now` + `voltage_now`;
* power = `|I|*V` of the whole phone (screen, radios, USB charging) -> now **incremental**: `P_inc = s*(I - I_baseline)*V`, `s` = load direction (+1/-1) found once per install by a 1.6 s CPU burn (so a charger that adds +800 mA cannot contaminate it);
* energy = mean power x latency -> now the **trapezoid integral** of `P_inc` over the real sample times;
* a run shorter than the fuel-gauge refresh (~1 s), or during which the gauge never left its pre-run value, is reported as `n/a` with the reason (`powerNote`) instead of an invented number: a 60 ms NPU inference cannot be energy-measured this way.

`PowerMonitor` samples at 5 Hz while the app is in front, so an idle baseline always exists; `PhaseClock` accumulates the phases (`t_infer` = time inside the accelerator call).

---------------------------------------------------------------------------------------------------------------------------------

## 3. Stable Diffusion: Smart SD (defect 3) and SD super-resolution

### 3.1 Why "sunglasses" produced a bare face

1. `SmartPrompt.BASE_NEGATIVE` contained **`frame, border, duplicate`**, the words that describe eyeglass frames and a second copy of an object: the negative prompt pushed the sample away from the thing requested.
2. CFG was fixed at 2.0: too weak for an object the model has to invent.
3. The composer wrote `a photorealistic photograph of sunglasses` (a product shot) instead of a person *wearing* them, and added `matching the ... skin tone of the subject` without knowing a subject.
4. The "context" was a colour statistic of the ring around the hole, which classified skin as `sandy dry ground`.
5. The drawer was drawn for **Standard** SD too, from a second analysis separate from what was dispatched.

### 3.2 New behaviour

* **Standard SD** = vanilla: the user's text and negative go to the runner untouched, CFG 2.0, no analysis, **no drawer**.
* **Smart SD** = its own model entry (no redundant switch). One `SmartSdPlan` per run (pure `SmartSd`, Android side `SmartSdAnalyzer`: 512 px copies, `android.media.FaceDetector`, ring analysis) is shown in the drawer **and** passed to the driver, so the strings you read are the strings sent.
  Stages: 1 Full-Image Context (scene class, lighting direction/luma/contrast, colour temperature, environment), 2 Masked Target Context (void size and box, which face zone it covers: eyewear / eyes / forehead / mouth / cheek / background), 3 Raw User Prompt (verbatim or `[Empty]`), 4 Final Synthesized Positive Prompt, 5 Active Negative Prompt (+ the sampler line in the drawer header).
* Composer: the user's words lead; wearables become `X, a person wearing X, X worn over the eyes, <lighting>, <tone>, photorealistic, sharp focus, natural skin texture, seamless blend ...`; features `a person with X`; objects `X clearly visible in the scene`; an expert prompt (> 7 words) is kept verbatim and first; an empty prompt infills the scene (skin and facial features on a face). Token budget 68 (CLIP reads 77 positions); the user's text is never cut.
* Negative: user's negative first, generic list **without frame/border/duplicate**, face terms on faces, and for wearables the absence of the thing (`no sunglasses, without sunglasses, bare eyes, no eyewear`); any negative term that appears in the intent (or its synonyms, for example `lens`/`frame` for glasses) is dropped.
* CFG by intent: wearable/feature 7.0, object 6.0, expert prompt 5.0, empty 3.5.

Example for sample 001 *(the actual output of `SmartSd.plan`, host-run)*: S4 = `sunglasses, a person wearing sunglasses, sunglasses worn over the eyes, soft diffused light from the right, warm tones, photorealistic, sharp focus, natural skin texture, seamless blend with the surrounding photo`.

### 3.3 SD "super-resolution" and the runner patch (optional, `sd_sr` is greyed out until installed)

Old path: a uniform 0.35-grey mask (not a hole), no start from the image (the runner begins at pure noise), tiles denoised independently, composite with the original: the tile is re-imagined (warp/hallucination) or returned unchanged.
Proper SD upscale = **SDEdit on overlapping tiles**: start at schedule index `k = N - round(N*strength)` from `x_k = alpha_t[k]*z0 + sigma_t[k]*eps` (`alpha_t = 1/sqrt(sigma^2+1)`, `sigma_t = sigma*alpha_t`, variance preserving), denoise the remaining steps, **do not composite with the input**. `SdSuperResEngine` then keeps only the null-space part of the result (`compose` in uniform mode), so even a hallucinating tile cannot move geometry.
Patch for the runner: `REPO/StableDiffusion/include/sd_img2img.hpp` (three edit points are documented at the top of the file: `--strength`, `--no-composite`, loop start index). After installing the patched runner: `adb shell touch /data/local/tmp/sd_runtime/sd_img2img_v1`. Only the first 2x pass uses diffusion (9 tiles); later passes use the classical base.

---------------------------------------------------------------------------------------------------------------------------------

## 4. Dual viewport (defect 4)

`SharedViewport` (pure, 9 tests) is the **only** owner of zoom `z >= 1` and centre `(cx, cy)` in normalised image coordinates. `ImagePaneView`s own no transform; they draw `destRect()`.
For a view with letter-boxed fit scale `s = min(Vw/bw, Vh/bh)`, `Lx = s*bw`, `Ly = s*bh`:

* screen position `x = Vw/2 + (u - cx)*Lx*z` (same for y);
* focal pinch about pixel `f`: `c' = p - (f - V/2)/(L*z')` with `p` the image point under the fingers before the zoom;
* drag by `d` pixels: `c' = c - d/(L*z)`;
* clamp: visible fraction `V/(L*z)`; centre in `[frac/2, 1-frac/2]`, exactly `0.5` when the whole axis is visible (smallest fraction over all views, so every view can reach the whole image).

Because the state is resolution independent, a 512 px input and a 4096 px output show the same scene region, and a gesture on either pane moves both (the old per-view handler mirrored callbacks and lost the pointer on `ACTION_POINTER_UP`; now the pan reference is re-anchored to the remaining fingers). Rendering clips to the visible source rectangle, draws real pixels when magnified (> 2.5x) and a mip level (`MipPyramid`) when shrunk, so a 4096 px output does not alias in a 330 px pane. **1:1 Reset** appears when the state differs from fit and re-centres both.

---------------------------------------------------------------------------------------------------------------------------------

## 5. Visual redesign and theme (defect 5)

* Theme `Theme.QIDKInpaint` = `Theme.Material3.DayNight.NoActionBar`; **every colour is a token** (`values/colors.xml`, `values-night/colors.xml`): dark `#0B0F19` canvas / `#161F30` cards, light `#F8FAFC` / `#FFFFFF`; accents cyan, emerald, purple, amber, crimson (darker light-mode variants so text on white keeps contrast).
* The old `applyThemeMode()` painted ~30 views from code with hard-coded colours (the source of black-on-grey text: `#FFFFFF` EditText text on a light field, spinner text from the default theme). It is deleted. The ☀️/🌙 button calls `AppCompatDelegate.setDefaultNightMode()` (persisted); `uiMode` is no longer in `configChanges`, so the activity is recreated and every card, text, spinner, chip and canvas well re-resolves its tokens. `UiSession` (+ `onPause` save) carries the picture, mask, model, prompts, results and tab across the recreation.
* `activity_main.xml`: 19 literal colours -> tokens (script-verified: none left); 3-way **segmented control** `[🖌️ Manual Brush | ✨ Smart AI Touch | 🔲 Bounding Box]` (`MaterialButtonToggleGroup`); compact toolbar Undo / Clear / 1:1 Reset; **Execution Target** card = model spinner + NPU/GPU/CPU **chips** (disabled when `BackendMatrix` forbids them) directly above the primary **Run Inpaint** button; the Smart-SD drawer only for Smart SD.
* New `view_super_resolution.xml`: engine, scale chips (2x/4x/8x with output size), backend chips, detail slider, progress + Cancel, stacked dual viewport, post-run card. Inpaint | Super-Resolution switch above both pages (`SuperResolutionTab`).

---------------------------------------------------------------------------------------------------------------------------------

## 6. Known limits and things to watch

* **Not run on the device.** The daemon file paths reuse the Inpaint path's directories (`/data/local/tmp/lama/input`, `.../app_live_output_v4`); the SD `sd_img2img_v1` marker is created by hand.
* NSDT quality depends on what each network does inside 32 px holes; read `lap_gain`/`detail_added` before believing "better than bicubic".
* `readPeakTemperatureC()` / `readActiveCurrentUa()` in `OnDeviceProcessDriver` are untouched: they feed the **router's** telemetry features (LinUCB state). Changing them would shift the policy's input distribution; that needs a retrain decision, not a patch. The telemetry *card* no longer uses them.
* Energy of very short runs (a 60 ms inference) is `n/a` by design.
* The review text pasted with the request (regret curves, latency/power tables) contains a point this bundle applies: report power as incremental and energy as an integral, never `P x latency` of a total-phone reading.
* Rollback: every overwritten file is in `<APP>/.solution_backup_<stamp>/` (see `apply_solution.sh`); `git revert` of the phase commits restores everything.

---------------------------------------------------------------------------------------------------------------------------------

## 7. Phases for the executing agent (`agy`) — in order, no questions

**Phase 0 — recon.** `git status`; create branch `agy/super-studio`; `export APP` (has `gradlew`) and `REPO` (has `StableDiffusion/`); `adb -s 8f27557f devices -l` must list the device; record `sha256sum` of the files named in `BASELINE_SHA256.txt` (drift is reported by the next step; a drifted file is replaced anyway, its original is in the backup folder, note it in `AGY_REPORT.md`).
**Phase 1 — install.** `bash tools/apply_solution.sh <unzipped dir> "$APP" "$REPO"` (copy `tools/` from the unzipped dir first if the project has none yet). Commit `phase 1: install bundle`.
**Phase 2 — host checks.** NSDT kernels and img2img host tests (commands in the headers of the two `.cpp` test files); `./gradlew --console=plain assembleDebug testDebugUnitTest` must end `BUILD SUCCESSFUL` (70+ tests). If Gradle reports an unresolved symbol, fix it in the smallest possible way and record it; do not rewrite the design.
**Phase 3 — optional runner patch** (SD upscale): locate `sd_runner_inpaint.cpp`, apply the three edit points of `sd_img2img.hpp`, rebuild the runner, push it next to the old one (back the old one up as `<name>.bak.$(date +%s)`), `touch /data/local/tmp/sd_runtime/sd_img2img_v1`. If the source is not on the machine, skip and record `sd_sr: not installed`.
**Phase 4 — install and verify.** `adb -s 8f27557f install -r -t app/build/outputs/apk/debug/app-debug.apk`, then `bash tools/verify_super_studio.sh --serial 8f27557f` (add `--skip-8x` only if a single 8x job exceeds 15 minutes, `--skip-sd` if the runner patch was skipped). Read `verify_out/REPORT.md`, the six screenshots and the SR result PNGs.
**Phase 5 — repair loop.** For every FAIL: read `QIDK.*` logcat, find the cause, fix, rebuild, re-run only the failing section. WARN lines about a rejected gate are findings, not failures: report which engine/scale was rejected and the gate metrics. Never weaken a gate to make it pass; never run `adb root`, `adb remount`, `setenforce`; never overwrite `/data/local/tmp/lama` or `/data/local/tmp/sd_runtime` contents without a `.bak.$(date +%s)` copy.
**Phase 6 — report.** Write `AGY_REPORT.md` (what was installed, drift list, test counts, per-engine SR table with `lr_psnr`, `nyq_share`, `mae_vs_bicubic`, `lap_gain`, gate verdicts, timing/power rows, Smart-SD stages, theme/viewport results, screenshots list, every deviation), commit `phase 6: report` (never commit `verify_out/` images larger than 5 MB, `*.dlc`, `*.bin`, `*.onnx`).

### Acceptance (all must hold)

1. `assembleDebug testDebugUnitTest` green; both host C++ tests print `0 failed`.
2. Smart plan for `sunglasses` (sample 001 + mask): S4 starts `sunglasses, a person wearing sunglasses`; S5 has no `frame/border/duplicate` and has `no sunglasses`; Standard SD sends the raw prompt; `[Empty]` shown for an empty prompt.
3. Every inpainting model x backend pair that `BackendMatrix` allows returns an image and a telemetry line; classical algorithms return.
4. SR: bicubic and Tarsh pass `sr_metrics.py` at 2x/4x/8x; every neural engine either passes (and reports detail) or reports a gate rejection with the classical base (still passing); no engine ever produces `nyq_share > 0.55` or `lr_psnr < 38 dB`.
5. Dual viewport: both panes report the same normalised region at fit, at zoom 3.5, after a real `adb input swipe` on the top pane, and after Reset (`zoom=1.000 center=(0.5000,0.5000)`).
6. Theme: light and dark both applied, contrast >= 7:1 text, >= 4.5:1 muted on cards, >= 3:1 accents, picture preserved across two toggles, screenshots without unreadable text.
7. No `FATAL EXCEPTION` in logcat; no thermal throttling kill of the daemon during 8x.
