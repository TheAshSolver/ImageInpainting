#!/usr/bin/env bash
# Hardware & Model Verification Protocol for the Super Studio build.
#   tools/verify_super_studio.sh [--serial 8f27557f] [--apk <path>] [--out verify_out] [--skip-8x] [--skip-sd]
# Writes <out>/REPORT.md, screenshots, result PNGs and logcat; exit code = number of FAIL lines (0 = every hard check passed).
# WARN = something to read (a quality gate rejected a pass, an engine is unavailable); SKIP = precondition missing.
set -u
SERIAL=8f27557f; APK=app/build/outputs/apk/debug/app-debug.apk; OUT=verify_out; SKIP8=0; SKIPSD=0
while [ $# -gt 0 ]; do case "$1" in
  --serial) SERIAL=$2; shift 2;; --apk) APK=$2; shift 2;; --out) OUT=$2; shift 2;;
  --skip-8x) SKIP8=1; shift;; --skip-sd) SKIPSD=1; shift;; *) echo "unknown argument $1"; exit 2;; esac; done

ADB="adb -s $SERIAL"; PKG=com.qualcomm.qidk.inpaint.v5; ACT=com.qualcomm.qidk.inpaint.MainActivity
HERE="$(cd "$(dirname "$0")/.." && pwd)"; DEV_OUT=/sdcard/Android/data/$PKG/files/out
FAILS=0; WARNS=0; mkdir -p "$OUT"; REPORT="$OUT/REPORT.md"; : > "$REPORT"
say(){ echo "$*" | tee -a "$REPORT"; }
pass(){ say "PASS  $*"; }; skip(){ say "SKIP  $*"; }
fail(){ say "FAIL  $*"; FAILS=$((FAILS+1)); }; warn(){ say "WARN  $*"; WARNS=$((WARNS+1)); }

hook(){ local a=$1; shift; $ADB shell am broadcast -a "$a" -p "$PKG" "$@" >/dev/null 2>&1; }
# wait_hook <action> <regex that the FINAL result line must match> <seconds>
wait_hook(){ local a=$1 rx=$2 t=$3 i=0 l
  while [ "$i" -lt "$t" ]; do
    l=$($ADB logcat -d -s QIDK.Hook:I 2>/dev/null | grep -F "$a ->" | grep -E "$rx" | tail -1)
    [ -n "$l" ] && { echo "$l"; return 0; }; sleep 1; i=$((i+1)); done; return 1; }
logc(){ $ADB logcat -c; }
shot(){ $ADB exec-out screencap -p > "$OUT/$1.png" 2>/dev/null; }
field(){ echo "$1" | sed -n "s/.*[ \[]$2=\([^ ]*\).*/\1/p" | head -1; }     # field "<line>" key

say "# Verification $(date -u +%FT%TZ)  serial=$SERIAL"

# ---------------------------------------------------------------------------------------------------------- 0. host tests
say; say "## 0. Host tests"
if command -v g++ >/dev/null; then
  g++ -O2 -std=c++17 -I "$HERE/app/src/main/cpp/sr" "$HERE/tools/native/test_sr_core.cpp" "$HERE/app/src/main/cpp/sr/sr_core.cpp" -o "$OUT/t_sr_core" 2>"$OUT/t_sr_core.build" \
    && "$OUT/t_sr_core" | tail -1 | grep -q " 0 failed" && pass "NSDT kernels: $("$OUT/t_sr_core" | tail -1)" || fail "NSDT kernel host test (see $OUT/t_sr_core.build)"
  REPO_INC=$(find "$HERE/.." -maxdepth 3 -path '*StableDiffusion/include' -type d 2>/dev/null | head -1)
  if [ -n "$REPO_INC" ] && [ -f "$REPO_INC/sd_img2img.hpp" ]; then
    g++ -O2 -std=c++17 -I "$REPO_INC" "$HERE/tools/native/test_sd_img2img.cpp" -o "$OUT/t_img2img" 2>/dev/null && "$OUT/t_img2img" | tail -1 | grep -q " 0 failed" && pass "img2img start maths: $("$OUT/t_img2img" | tail -1)" || fail "img2img host test"
  else warn "sd_img2img.hpp not found next to the project; copy REPO/StableDiffusion/include/sd_img2img.hpp first"; fi
else warn "g++ not on PATH: host C++ tests not run"; fi
python3 "$HERE/tools/sr_metrics.py" --help >/dev/null 2>&1 && pass "sr_metrics.py runs (numpy + cv2 present)" || fail "sr_metrics.py cannot run (pip install numpy opencv-python)"

# ---------------------------------------------------------------------------------------------------------- 1. device
say; say "## 1. Device and install"
$ADB get-state 2>/dev/null | grep -q device && pass "device $SERIAL attached" || { fail "device $SERIAL not attached"; exit 1; }
$ADB shell getprop ro.product.board 2>/dev/null | tr -d '\r' | sed 's/^/board: /' | tee -a "$REPORT"
if [ -f "$HERE/$APK" ] || [ -f "$APK" ]; then
  A=$APK; [ -f "$HERE/$APK" ] && A="$HERE/$APK"
  $ADB install -r -t "$A" 2>&1 | grep -q Success && pass "APK installed ($A)" || fail "APK install failed"
else fail "APK not found at $APK (run ./gradlew assembleDebug first)"; fi
$ADB shell pm grant "$PKG" android.permission.READ_MEDIA_IMAGES >/dev/null 2>&1
$ADB shell "ls /data/local/tmp/lama /data/local/tmp/sd_runtime 2>/dev/null | head -50" > "$OUT/device_dirs.txt"
$ADB shell "echo PING | nc -w 1 127.0.0.1 55555 2>/dev/null | head -1" | tr -d '\r' | grep -q PONG && pass "resident daemon answers PING on 55555" || warn "resident daemon silent on 55555 (GAN models fall back to snpe-net-run; SR neural engines will fall back to the classical base)"
logc; $ADB shell am force-stop "$PKG"; $ADB shell am start -n "$PKG/$ACT" >/dev/null; sleep 6
$ADB logcat -d | grep -E "FATAL EXCEPTION|AndroidRuntime.*$PKG" > "$OUT/crash_at_start.txt"
[ -s "$OUT/crash_at_start.txt" ] && fail "crash at start (see $OUT/crash_at_start.txt)" || pass "app starts without a crash"
$ADB logcat -d -s QIDK.SR:I | grep -q "native SR kernels loaded=true" && pass "libqidk_sr.so loaded" || fail "libqidk_sr.so did not load (CMake target qidk_sr)"
hook qidk.debug.SAMPLE --es name 001 --ez mask true; sleep 3
shot 01_inpaint_dark

# ---------------------------------------------------------------------------------------------------------- 2. inpainting models
say; say "## 2. Inpainting models (sample 001 + its mask)"
for m in migan lama aotgan; do for b in NPU GPU CPU; do
  logc; hook qidk.debug.INPAINT --es model "$m" --es backend "$b" --es out "$DEV_OUT/inp_${m}_$b.png"
  L=$(wait_hook "qidk.debug.INPAINT" "model=|Exception|error" 120) || { fail "inpaint $m/$b: no answer in 120 s"; continue; }
  if echo "$L" | grep -q "model="; then
    $ADB pull "$DEV_OUT/inp_${m}_$b.png" "$OUT/" >/dev/null 2>&1
    pass "inpaint $m/$b: total=$(field "$L" total) ms prep=$(field "$L" prep) infer=$(field "$L" infer) post=$(field "$L" post) dT=$(field "$L" dT) W=$(field "$L" W) J=$(field "$L" J)"
  else fail "inpaint $m/$b: $L"; fi
done; done
for a in telea ns criminisi biharmonic; do
  logc; hook qidk.debug.CLASSICAL --es algo "$a"
  L=$(wait_hook "qidk.debug.CLASSICAL" "used=|bad algo" 180) && echo "$L" | grep -q "used=" && pass "classical $a: $(echo "$L" | sed 's/.*-> //')" || fail "classical $a: ${L:-no answer}"
done

# ---------------------------------------------------------------------------------------------------------- 3. Smart SD
say; say "## 3. Standard SD vs Smart SD (prompt trace, no NPU needed)"
logc; hook qidk.debug.PLAN --es prompt sunglasses --ez smart true; sleep 3
S=$($ADB logcat -d -s QIDK.SD:I); echo "$S" | grep -E "S[1-5]=|Sampler" > "$OUT/smart_plan.txt"
S4=$(grep '^.*S4=' "$OUT/smart_plan.txt" | sed 's/.*S4=//'); S5=$(grep 'S5=' "$OUT/smart_plan.txt" | sed 's/.*S5=//')
echo "$S4" | grep -q '^sunglasses, a person wearing sunglasses' && pass "Smart S4 leads with the user's intent: $S4" || fail "Smart S4 is not intent-first: '$S4'"
for w in frame border duplicate; do echo "$S5" | grep -qi "\b$w\b" && fail "Smart S5 still contains '$w'"; done; echo "$S5" | grep -q "no sunglasses" && pass "Smart S5 asks for the absence of nothing it needs: negative has 'no sunglasses' and no eyeglass-frame words" || fail "Smart S5 lacks the intent-opposite terms"
grep -q "S1=Scene:" "$OUT/smart_plan.txt" && grep -q "S2=Void:" "$OUT/smart_plan.txt" && grep -q "S3=sunglasses" "$OUT/smart_plan.txt" && pass "stages 1-3 present (scene, target, raw prompt)" || fail "stage 1-3 missing"
logc; hook qidk.debug.PLAN --es prompt "a red car" --ez smart false; sleep 2
$ADB logcat -d -s QIDK.SD:I | grep -q "S4=a red car$" && pass "Standard SD sends the raw prompt untouched" || fail "Standard SD manipulated the prompt"
logc; hook qidk.debug.PLAN --es prompt "" --ez smart true; sleep 2
$ADB logcat -d -s QIDK.SD:I | grep -q "S3=\[Empty\]" && pass "empty prompt is shown as [Empty]" || fail "stage 3 does not show [Empty]"
if [ $SKIPSD -eq 0 ]; then
  for mode in normal smart; do
    logc; hook qidk.debug.SD --es mode $mode --es prompt sunglasses --es tag verify
    L=$(wait_hook "qidk.debug.SD" "out=|Exception|error" 400) && echo "$L" | grep -q "out=" && { $ADB pull "$DEV_OUT/sd_${mode}_verify.png" "$OUT/" >/dev/null 2>&1; pass "SD $mode run finished: $(echo "$L" | sed 's/.*-> //')"; } || fail "SD $mode run: ${L:-no answer}"
  done
  [ -f "$OUT/sd_normal_verify.png" ] && [ -f "$OUT/sd_smart_verify.png" ] && cmp -s "$OUT/sd_normal_verify.png" "$OUT/sd_smart_verify.png" && fail "Smart and Standard SD produced identical pixels" || say "INFO  compare $OUT/sd_normal_verify.png and sd_smart_verify.png by eye: the Smart result should show sunglasses over the eyes"
fi

# ---------------------------------------------------------------------------------------------------------- 4. super-resolution
say; say "## 4. Super-resolution (engines x scales; metrics from tools/sr_metrics.py)"
hook qidk.debug.TAB --es tab sr; sleep 2; shot 02_sr_dark
run_sr(){ # engine backend scale timeout
  local e=$1 b=$2 s=$3 t=$4 tag="sr_${1}_${3}x"
  logc; hook qidk.debug.SR --es engine "$e" --es backend "$b" --ei scale "$s" --es out "$DEV_OUT/$tag.png" --es in "$DEV_OUT/${tag}_in.png"
  local L; L=$(wait_hook "qidk.debug.SR" "engine=|disabled|no source|Exception|error" "$t") || { fail "$tag: no answer in ${t}s"; return; }
  echo "$L" | grep -q "engine disabled" && { skip "$tag: $(echo "$L" | sed 's/.*-> //')"; return; }
  echo "$L" | grep -q "engine=" || { fail "$tag: $L"; return; }
  $ADB pull "$DEV_OUT/$tag.png" "$OUT/" >/dev/null 2>&1; $ADB pull "$DEV_OUT/${tag}_in.png" "$OUT/" >/dev/null 2>&1
  $ADB logcat -d -s QIDK.SR:I | grep "note:" | sed 's/.*note: /      note: /' >> "$REPORT"
  say "      timing: total=$(field "$L" ms) prep=$(field "$L" prep) infer=$(field "$L" infer) post=$(field "$L" post) dT=$(field "$L" dT) W=$(field "$L" W) J=$(field "$L" J)"
  local M; M=$(python3 "$HERE/tools/sr_metrics.py" --input "$OUT/${tag}_in.png" --output "$OUT/$tag.png" --scale "$s")
  echo "$M" > "$OUT/$tag.metrics.json"
  if echo "$M" | grep -q '"verdict": "PASS"'; then pass "$tag: $(echo "$M" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("lr_psnr=%s (bicubic %s) nyq=%s mae_vs_bicubic=%s lap_gain=%s detail_added=%s" % (d["lr_psnr"],d["lr_psnr_bicubic"],d["nyq_share"],d["mae_vs_bicubic"],d["lap_gain"],d["detail_added"]))')"
  else fail "$tag: $M"; fi
  $ADB logcat -d -s QIDK.SR:I | grep -q "rejected" && warn "$tag: a quality gate rejected a pass (classical base used for it): see notes above"
}
for e in bicubic tarsh; do for s in 2 4 8; do [ "$s" = 8 ] && [ $SKIP8 -eq 1 ] && continue; run_sr $e CPU $s 300; done; done
for e in migan_sr lama_sr aotgan_sr; do for s in 2 4 8; do [ "$s" = 8 ] && [ $SKIP8 -eq 1 ] && continue; run_sr $e NPU $s 900; done; done
[ $SKIPSD -eq 0 ] && run_sr sd_sr NPU 2 1500 || skip "sd_sr"

# ---------------------------------------------------------------------------------------------------------- 5. dual viewport
say; say "## 5. Dual viewport synchronisation"
hook qidk.debug.TAB --es tab sr; sleep 1
sync_check(){ local L=$1 what=$2 t b
  t=$(echo "$L" | sed -n 's/.* top=\[\([^]]*\)\].*/\1/p'); b=$(echo "$L" | sed -n 's/.* bottom=\[\([^]]*\)\].*/\1/p')
  python3 - "$t" "$b" "$what" <<'PY' && pass "$what: both panes show the same region ($t)" || fail "$what: panes disagree top=[$t] bottom=[$b]"
import sys
a=[float(x) for x in sys.argv[1].split(',')]; b=[float(x) for x in sys.argv[2].split(',')]
sys.exit(0 if all(abs(x-y) < 1e-3 for x,y in zip(a,b)) else 1)
PY
}
logc; hook qidk.debug.VIEWPORT --ez reset true; sleep 1; L=$($ADB logcat -d -s QIDK.Viewport:I | tail -1)
sync_check "$L" "fit"
logc; hook qidk.debug.VIEWPORT --es zoom 3.5 --es cx 0.30 --es cy 0.62; sleep 1; L=$($ADB logcat -d -s QIDK.Viewport:I | tail -1)
sync_check "$L" "zoom 3.5 at (0.30,0.62)"; shot 03_sr_zoomed
SC=$(echo "$L" | sed -n 's/.*topScreen=\[\([^]]*\)\].*/\1/p'); IFS=, read -r X0 Y0 X1 Y1 <<<"$SC"
if [ -n "${X0:-}" ]; then
  CX=$(( (X0 + X1) / 2 )); CY=$(( (Y0 + Y1) / 2 ))
  logc; $ADB shell input swipe "$((CX+150))" "$CY" "$((CX-150))" "$CY" 400; sleep 1
  L2=$($ADB logcat -d -s QIDK.Viewport:I | tail -1); hook qidk.debug.VIEWPORT; sleep 1; L2=$($ADB logcat -d -s QIDK.Viewport:I | tail -1)
  sync_check "$L2" "after a real swipe on the TOP pane"
  [ "$(field "$L" center)" != "$(field "$L2" center)" ] && pass "the swipe moved the shared centre" || warn "swipe did not change the centre (check adb input and pane bounds $SC)"
  logc; $ADB shell input swipe "$CX" "$((Y1 + 40))" "$((CX + 120))" "$((Y1 + 40))" 300; sleep 1
fi
logc; hook qidk.debug.VIEWPORT --ez reset true; sleep 1; L=$($ADB logcat -d -s QIDK.Viewport:I | tail -1)
echo "$L" | grep -q "zoom=1.000 center=(0.5000,0.5000)" && pass "1:1 Reset re-centres both panes" || fail "reset did not restore fit + centre: $L"
echo "$L" | grep -q "INPUT" && echo "$L" | grep -q "OUTPUT" && pass "badges present: $(echo "$L" | sed 's/.*topImg=//')" || warn "badges not both visible (run an SR job first)"

# ---------------------------------------------------------------------------------------------------------- 6. theme
say; say "## 6. Theme toggle"
logc; hook qidk.debug.THEMEINFO; sleep 1; D=$($ADB logcat -d -s QIDK.Theme:I | tail -1)
H1=$(field "$D" srcHash); N1=$(field "$D" night)
hook qidk.debug.THEME --es mode light; sleep 5; logc; hook qidk.debug.THEMEINFO; sleep 1; LG=$($ADB logcat -d -s QIDK.Theme:I | tail -1); shot 04_sr_light
hook qidk.debug.TAB --es tab inpaint; sleep 2; shot 05_inpaint_light
hook qidk.debug.THEME --es mode dark; sleep 5; logc; hook qidk.debug.THEMEINFO; sleep 1; DK=$($ADB logcat -d -s QIDK.Theme:I | tail -1); shot 06_inpaint_dark
[ "$(field "$LG" night)" = "false" ] && [ "$(field "$DK" night)" = "true" ] && pass "light and dark mode both take effect (activity recreated)" || fail "night flag did not follow the toggle: light='$LG' dark='$DK'"
for L in "$LG" "$DK"; do
  python3 - "$(field "$L" contrast_text)" "$(field "$L" contrast_muted)" "$(field "$L" contrast_cyan)" "$(field "$L" contrast_emerald)" <<'PY' && pass "contrast ok ($(field "$L" night)): text=$(field "$L" contrast_text) muted=$(field "$L" contrast_muted) cyan=$(field "$L" contrast_cyan) emerald=$(field "$L" contrast_emerald)" || fail "contrast below WCAG AA on a card in '$L'"
import sys
t,m,c,e=[float(x) for x in sys.argv[1:5]]
sys.exit(0 if t>=7 and m>=4.5 and c>=3 and e>=3 else 1)
PY
done
[ -n "$H1" ] && [ "$H1" = "$(field "$DK" srcHash)" ] && pass "the loaded picture survived two theme switches (hash $H1)" || fail "picture lost across the theme switch ($H1 -> $(field "$DK" srcHash))"
say "INFO  look at $OUT/0[1-6]_*.png: no unreadable black-on-grey text in either theme, spinners/chips/cards follow the theme"

# ---------------------------------------------------------------------------------------------------------- 7. log sweep
say; say "## 7. Logcat sweep"
$ADB logcat -d > "$OUT/logcat_full.txt"
grep -E "FATAL EXCEPTION|AndroidRuntime" "$OUT/logcat_full.txt" | grep -v "$OUT" > "$OUT/crashes.txt"
[ -s "$OUT/crashes.txt" ] && fail "crashes in logcat (see $OUT/crashes.txt)" || pass "no crash in logcat"
grep -E " E QIDK\.| E OnDeviceProcessDriver" "$OUT/logcat_full.txt" > "$OUT/qidk_errors.txt"
[ -s "$OUT/qidk_errors.txt" ] && warn "$(wc -l < "$OUT/qidk_errors.txt") error lines from QIDK tags (see $OUT/qidk_errors.txt)" || pass "no QIDK error lines"

say; say "RESULT: $FAILS failed, $WARNS warnings"
exit $FAILS
