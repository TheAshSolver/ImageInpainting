# Verification 2026-11-11T09:20:15Z  serial=8f27557f

## 0. Host tests
PASS  NSDT kernels: 44139 checks, 0 failed
PASS  img2img start maths: 167 checks, 0 failed
PASS  sr_metrics.py runs (numpy + cv2 present)

## 1. Device and install
PASS  device 8f27557f attached
board: sun
PASS  APK installed (/home/tarsh/Desktop/college/ImageInpainting/qidk-inpaint-app-v5/app/build/outputs/apk/debug/app-debug.apk)
PASS  resident daemon answers PING on 55555
PASS  app starts without a crash
PASS  libqidk_sr.so loaded

## 2. Inpainting models (sample 001 + its mask)
PASS  inpaint migan/NPU: total=1851 ms prep=96 infer=867 post=881 dT=-0.52000046 W=null J=null
PASS  inpaint migan/GPU: total=1323 ms prep=67 infer=1167 post=82 dT=0.066661835 W=null J=null
PASS  inpaint migan/CPU: total=466 ms prep=68 infer=363 post=26 dT=32.03636 W=null J=null
PASS  inpaint lama/NPU: total=2330 ms prep=66 infer=2170 post=87 dT=-0.5166626 W=null J=null
PASS  inpaint lama/GPU: total=3152 ms prep=71 infer=3034 post=41 dT=0.5818176 W=null J=null
PASS  inpaint lama/CPU: total=1750 ms prep=70 infer=1645 post=27 dT=58.781826 W=null J=null
PASS  inpaint aotgan/NPU: total=1853 ms prep=70 infer=1719 post=56 dT=-1.0545425 W=null J=null
PASS  inpaint aotgan/GPU: total=3499 ms prep=74 infer=3374 post=45 dT=-0.25455093 W=null J=null
PASS  inpaint aotgan/CPU: total=3078 ms prep=75 infer=2971 post=27 dT=56.545452 W=null J=null
PASS  classical telea: used=Fast Marching (Telea) ms=41 out=/storage/emulated/0/Android/data/com.qualcomm.qidk.inpaint.v5/files/out/classical_telea.png
PASS  classical ns: used=Fluid Streamlines (Navier-Stokes) ms=38 out=/storage/emulated/0/Android/data/com.qualcomm.qidk.inpaint.v5/files/out/classical_ns.png
PASS  classical criminisi: used=Criminisi Exemplar ms=492 out=/storage/emulated/0/Android/data/com.qualcomm.qidk.inpaint.v5/files/out/classical_criminisi.png
PASS  classical biharmonic: used=Biharmonic Poisson ms=376 out=/storage/emulated/0/Android/data/com.qualcomm.qidk.inpaint.v5/files/out/classical_biharmonic.png

## 3. Standard SD vs Smart SD (prompt trace, no NPU needed)
PASS  Smart S4 leads with the user's intent: sunglasses, a person wearing sunglasses, sunglasses worn over the eyes, dim low-key light from below, warm tones, photorealistic, sharp focus, natural skin texture, seamless blend with the surrounding photo
PASS  Smart S5 asks for the absence of nothing it needs: negative has 'no sunglasses' and no eyeglass-frame words
PASS  stages 1-3 present (scene, target, raw prompt)
PASS  Standard SD sends the raw prompt untouched
PASS  empty prompt is shown as [Empty]

## 4. Super-resolution (engines x scales; metrics from tools/sr_metrics.py)
      timing: total=11 prep=0 infer=5 post=0 dT=9.75 W=null J=null
PASS  sr_bicubic_2x: lr_psnr=53.51 (bicubic 53.51) nyq=0.0 mae_vs_bicubic=0.0 lap_gain=1.0 detail_added=False
      timing: total=17 prep=0 infer=8 post=1 dT=5.3916626 W=null J=null
PASS  sr_bicubic_4x: lr_psnr=52.31 (bicubic 52.31) nyq=0.333 mae_vs_bicubic=0.0 lap_gain=1.0 detail_added=False
      timing: total=37 prep=0 infer=24 post=4 dT=9.66 W=null J=null
PASS  sr_bicubic_8x: lr_psnr=52.03 (bicubic 52.03) nyq=0.332 mae_vs_bicubic=0.002 lap_gain=1.0 detail_added=False
      timing: total=77 prep=0 infer=72 post=0 dT=18.419998 W=null J=null
PASS  sr_tarsh_2x: lr_psnr=45.75 (bicubic 53.51) nyq=0.069 mae_vs_bicubic=0.889 lap_gain=2.554 detail_added=True
      timing: total=222 prep=10 infer=150 post=55 dT=18.30833 W=null J=null
PASS  sr_tarsh_4x: lr_psnr=46.2 (bicubic 52.31) nyq=0.091 mae_vs_bicubic=0.807 lap_gain=2.0 detail_added=True
      timing: total=912 prep=33 infer=607 post=266 dT=23.400002 W=null J=null
PASS  sr_tarsh_8x: lr_psnr=46.29 (bicubic 52.03) nyq=0.139 mae_vs_bicubic=0.896 lap_gain=1.299 detail_added=True
      note: I/O contract of migan: hole=0 range=[0,1]
      note: pass 1 (512 -> 1024): accepted | detail 0.49 lv | mesh 0.07 | identity 0.07 | LR change 0.36 lv | clamped 0.00 | rejected phases 0.00 | windows 9
      note: LR consistency (box 2x down-sample vs input): 49.0 dB
      timing: total=17271 prep=1497 infer=16400 post=239 dT=-0.19000244 W=null J=null
PASS  sr_migan_sr_2x: lr_psnr=48.97 (bicubic 53.51) nyq=0.081 mae_vs_bicubic=0.65 lap_gain=1.053 detail_added=True
WARN  sr_migan_sr_2x: a quality gate rejected a pass (classical base used for it): see notes above
      note: I/O contract of migan: hole=0 range=[0,1]
      note: pass 1 (512 -> 1024): accepted | detail 0.49 lv | mesh 0.07 | identity 0.07 | LR change 0.36 lv | clamped 0.00 | rejected phases 0.00 | windows 9
      note: pass 2 (1024 -> 2048): accepted | detail 0.21 lv | mesh 0.08 | identity 0.09 | LR change 0.14 lv | clamped 0.00 | rejected phases 0.00 | windows 36
      note: LR consistency (box 4x down-sample vs input): 47.4 dB
      timing: total=81617 prep=2343 infer=77967 post=1234 dT=0.12000275 W=null J=null
PASS  sr_migan_sr_4x: lr_psnr=47.43 (bicubic 52.31) nyq=0.126 mae_vs_bicubic=0.675 lap_gain=1.355 detail_added=True
WARN  sr_migan_sr_4x: a quality gate rejected a pass (classical base used for it): see notes above
      note: I/O contract of migan: hole=0 range=[0,1]
      note: pass 1 (512 -> 1024): accepted | detail 0.49 lv | mesh 0.07 | identity 0.07 | LR change 0.36 lv | clamped 0.00 | rejected phases 0.00 | windows 9
      note: pass 2 (1024 -> 2048): accepted | detail 0.21 lv | mesh 0.08 | identity 0.09 | LR change 0.14 lv | clamped 0.00 | rejected phases 0.00 | windows 36
      note: pass 3 (2048 -> 4096): accepted | detail 0.13 lv | mesh 0.08 | identity 0.13 | LR change 0.08 lv | clamped 0.00 | rejected phases 0.00 | windows 121
      note: LR consistency (box 8x down-sample vs input): 47.1 dB
      timing: total=301144 prep=8891 infer=287031 post=4990 dT=-0.37272644 W=null J=null
PASS  sr_migan_sr_8x: lr_psnr=47.11 (bicubic 52.03) nyq=0.166 mae_vs_bicubic=0.689 lap_gain=1.245 detail_added=True
WARN  sr_migan_sr_8x: a quality gate rejected a pass (classical base used for it): see notes above
      note: I/O contract of lama: hole=1 range=[0,1]
      note: pass 1 (512 -> 1024): accepted | detail 0.49 lv | mesh 0.17 | identity 0.05 | LR change 0.34 lv | clamped 0.00 | rejected phases 0.00 | windows 9
      note: LR consistency (box 2x down-sample vs input): 48.8 dB
      timing: total=41706 prep=2673 infer=40940 post=255 dT=0.14999771 W=null J=null
PASS  sr_lama_sr_2x: lr_psnr=48.77 (bicubic 53.51) nyq=0.11 mae_vs_bicubic=0.637 lap_gain=0.993 detail_added=True
WARN  sr_lama_sr_2x: a quality gate rejected a pass (classical base used for it): see notes above
      note: I/O contract of lama: hole=1 range=[0,1]
      note: pass 1 (512 -> 1024): accepted | detail 0.49 lv | mesh 0.17 | identity 0.05 | LR change 0.34 lv | clamped 0.00 | rejected phases 0.00 | windows 9
      note: pass 2 (1024 -> 2048): accepted | detail 0.28 lv | mesh 0.23 | identity 0.06 | LR change 0.13 lv | clamped 0.00 | rejected phases 0.00 | windows 36
      note: LR consistency (box 4x down-sample vs input): 47.3 dB
      timing: total=197392 prep=2341 infer=193776 post=1197 dT=0.31666565 W=null J=null
PASS  sr_lama_sr_4x: lr_psnr=47.26 (bicubic 52.31) nyq=0.158 mae_vs_bicubic=0.666 lap_gain=1.414 detail_added=True
WARN  sr_lama_sr_4x: a quality gate rejected a pass (classical base used for it): see notes above
      note: I/O contract of lama: hole=1 range=[0,1]
      note: pass 1 (512 -> 1024): accepted | detail 0.49 lv | mesh 0.17 | identity 0.05 | LR change 0.34 lv | clamped 0.00 | rejected phases 0.00 | windows 9
      note: pass 2 (1024 -> 2048): accepted | detail 0.28 lv | mesh 0.23 | identity 0.06 | LR change 0.13 lv | clamped 0.00 | rejected phases 0.00 | windows 36
      note: pass 3 (2048 -> 4096): accepted | detail 0.25 lv | mesh 0.25 | identity 0.08 | LR change 0.09 lv | clamped 0.00 | rejected phases 0.00 | windows 121
      note: LR consistency (box 8x down-sample vs input): 47.0 dB
      timing: total=729652 prep=9150 infer=715189 post=5068 dT=-0.07999802 W=null J=null
PASS  sr_lama_sr_8x: lr_psnr=46.96 (bicubic 52.03) nyq=0.205 mae_vs_bicubic=0.694 lap_gain=1.642 detail_added=True
WARN  sr_lama_sr_8x: a quality gate rejected a pass (classical base used for it): see notes above
      note: I/O probe failed for aotgan (hole=1 range=[0,1]: outside 0.000 hole 0.058 (identity 0.090); hole=0 range=[0,1]: outside 0.231 hole 0.090 (identity 0.090); hole=1 range=[-1,1]: outside 0.000 hole 0.179 (identity 0.090); hole=0 range=[-1,1]: outside 0.286 hole 0.090 (identity 0.090)): the classical base is used
      note: LR consistency (box 2x down-sample vs input): 52.0 dB
      timing: total=7201 prep=7268 infer=6898 post=29 dT=0.23999786 W=null J=null
PASS  sr_aotgan_sr_2x: lr_psnr=51.96 (bicubic 53.51) nyq=0.1 mae_vs_bicubic=0.312 lap_gain=0.895 detail_added=True
      note: I/O probe failed for aotgan (hole=1 range=[0,1]: outside 0.000 hole 0.058 (identity 0.090); hole=0 range=[0,1]: outside 0.231 hole 0.090 (identity 0.090); hole=1 range=[-1,1]: outside 0.000 hole 0.179 (identity 0.090); hole=0 range=[-1,1]: outside 0.286 hole 0.090 (identity 0.090)): the classical base is used
      note: LR consistency (box 4x down-sample vs input): 50.2 dB
      timing: total=50 prep=0 infer=0 post=42 dT=0.036361694 W=null J=null
PASS  sr_aotgan_sr_4x: lr_psnr=50.21 (bicubic 52.31) nyq=0.213 mae_vs_bicubic=0.362 lap_gain=0.965 detail_added=True
      note: I/O probe failed for aotgan (hole=1 range=[0,1]: outside 0.000 hole 0.058 (identity 0.090); hole=0 range=[0,1]: outside 0.231 hole 0.090 (identity 0.090); hole=1 range=[-1,1]: outside 0.000 hole 0.179 (identity 0.090); hole=0 range=[-1,1]: outside 0.286 hole 0.090 (identity 0.090)): the classical base is used
      note: LR consistency (box 8x down-sample vs input): 49.9 dB
      timing: total=171 prep=0 infer=0 post=154 dT=0.099998474 W=null J=null
PASS  sr_aotgan_sr_8x: lr_psnr=49.9 (bicubic 52.03) nyq=0.187 mae_vs_bicubic=0.4 lap_gain=0.948 detail_added=True
SKIP  sd_sr

## 5. Dual viewport synchronisation
PASS  fit: both panes show the same region (-0.0174,0.0000,1.0174,1.0000)
PASS  zoom 3.5 at (0.30,0.62): both panes show the same region (0.1522,0.4771,0.4478,0.7629)
PASS  after a real swipe on the TOP pane: both panes show the same region (0.2424,0.4771,0.5381,0.7629)
PASS  the swipe moved the shared centre
PASS  1:1 Reset re-centres both panes
PASS  badges present: INPUT 512x512 bottomImg=OUTPUT

## 6. Theme toggle
PASS  light and dark mode both take effect (activity recreated)
PASS  contrast ok (false): text=17.9 muted=7.6 cyan=4.1 emerald=5.5
PASS  contrast ok (true): text=15.8 muted=6.4 cyan=7.7 emerald=6.5
PASS  the loaded picture survived two theme switches (hash 8ee407a461845b65)
INFO  look at verify_out/0[1-6]_*.png: no unreadable black-on-grey text in either theme, spinners/chips/cards follow the theme

## 7. Logcat sweep
PASS  no crash in logcat
PASS  no QIDK error lines

RESULT: 0 failed, 6 warnings
