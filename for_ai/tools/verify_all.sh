#!/usr/bin/env bash
# tools/verify_all.sh - build, install and exercise every phase on the device, print PASS / FAIL per check.
# Run from the project root:   SERIAL=8f27557f ./tools/verify_all.sh        (SKIP_BUILD=1 reuses the existing APK)
# Hooks it drives (all registered by DebugHooks in MainActivity, debug builds only; see Phase 1.7):
#   qidk.debug.SAMPLE --es name 006 [--ez mask true]     qidk.debug.MODE --es mode smart|brush|box
#   qidk.debug.CLASSICAL --es algo telea|ns|criminisi|biharmonic
#   qidk.debug.SR --es engine bicubic --es backend CPU    qidk.debug.ROUTE
#   qidk.debug.SD --es mode normal|smart --es prompt "..." [--es neg "..."] [--ei seed 42]
set -u
SERIAL="${SERIAL:-8f27557f}"
ADB="adb -s $SERIAL"
GRADLE_FILE="$(ls app/build.gradle.kts app/build.gradle 2>/dev/null | head -1)"
PKG="${PKG:-$(grep -m1 -E 'applicationId' "$GRADLE_FILE" | sed -E 's/.*"([^"]+)".*/\1/')}"
APK="app/build/outputs/apk/debug/app-debug.apk"
OUT="/sdcard/Android/data/$PKG/files/out"
pass=0; fail=0

step() {                      # step "name" command...
  local name="$1"; shift
  if "$@" >/tmp/qidk_step.out 2>&1; then echo "PASS  $name"; pass=$((pass+1))
  else echo "FAIL  $name"; tail -n 12 /tmp/qidk_step.out | sed 's/^/        /'; fail=$((fail+1)); fi
}
wait_log() {                  # wait_log "regex" seconds -> prints the matching line, returns 1 on timeout
  local end=$((SECONDS+$2))
  while [ $SECONDS -lt $end ]; do
    local l; l="$($ADB logcat -d | grep -E "$1" | tail -n 1)"
    if [ -n "$l" ]; then echo "$l"; return 0; fi
    sleep 1
  done
  return 1
}
hook() {                      # hook ACTION [am extras...]; clears logcat first so wait_log sees only this call
  local action="$1"; shift
  $ADB logcat -c
  $ADB shell am broadcast -a "$action" -p "$PKG" "$@" >/dev/null
}

echo "== 1. build, unit tests, host-side native tests"
if [ -z "${SKIP_BUILD:-}" ]; then
  step "gradle assembleDebug testDebugUnitTest" ./gradlew --console=plain assembleDebug testDebugUnitTest
fi
step "native self-tests (Criminisi, biharmonic)" bash -c 'g++ -O2 -std=c++17 -pthread -I app/src/main/cpp/classical tools/native/selftest.cpp app/src/main/cpp/classical/criminisi.cpp app/src/main/cpp/classical/biharmonic.cpp -o /tmp/qidk_selftest && /tmp/qidk_selftest'
REPO="${REPO:-$(git rev-parse --show-toplevel 2>/dev/null || echo ..)}"
step "SD runner argument parser" bash -c "g++ -O2 -std=c++17 -I '$REPO/StableDiffusion/include' tools/native/test_sd_args.cpp -o /tmp/qidk_sdargs && /tmp/qidk_sdargs"

echo "== 2. install and launch ($PKG on $SERIAL)"
step "adb install" $ADB install -r -t "$APK"
$ADB shell am force-stop "$PKG"; $ADB logcat -c
$ADB shell am start -n "$PKG/.MainActivity" >/dev/null
sleep 5
step "debug hooks registered" bash -c "$ADB logcat -d -s QIDK.Hook:I | grep -q installed"
$ADB shell mkdir -p "$OUT"

echo "== 3. touch geometry and Smart AI Touch (Phase 1)"
hook qidk.debug.SAMPLE --es name 006; wait_log "QIDK.Hook.*SAMPLE.*loaded" 20 >/dev/null
hook qidk.debug.MODE --es mode brush; sleep 1
step "touch geometry (overlay lands under the finger)" python3 tools/qidk_adb.py geometry --serial "$SERIAL"
hook qidk.debug.MODE --es mode smart; sleep 1
python3 tools/qidk_adb.py lasso --serial "$SERIAL" --cx 0.52 --cy 0.63 --rx 0.31 --ry 0.17 | tee /tmp/qidk_lasso.out
step "smart lasso around the boat snapped to the contour" bash -c "grep -q 'QIDK.Smart.*snapped=true' <($ADB logcat -d -s QIDK.Smart:I) && grep -q '^ok ' /tmp/qidk_lasso.out"

echo "== 4. classical algorithms (Phase 2): none may crash, none may fall back"
for algo in telea ns criminisi biharmonic; do
  hook qidk.debug.SAMPLE --es name 006 --ez mask true; wait_log "QIDK.Hook.*SAMPLE.*loaded" 20 >/dev/null
  hook qidk.debug.CLASSICAL --es algo "$algo"
  line="$(wait_log 'QIDK.Classic: requested=' 90)"
  wait_log "QIDK.Hook.*CLASSICAL -> " 90 >/dev/null
  echo "      $algo: ${line:-no log line}"
  step "classical $algo ran and used its own kernel" bash -c "echo '$line' | grep -q 'requested=' && ! echo '$line' | grep -qi 'fallback'"
  $ADB pull "$OUT/classical_$algo.png" "/tmp/classical_$algo.png" >/dev/null 2>&1 && echo "      pulled /tmp/classical_$algo.png (look at it)"
done

echo "== 5. backends (Phase 3)"
step "daemon answers STATUS" bash -c "$ADB shell 'echo STATUS | nc 127.0.0.1 55555' | grep -q '^OK'"

echo "== 6. Stable Diffusion prompts (Phase 4)"
hook qidk.debug.SAMPLE --es name 006 --ez mask true; wait_log "QIDK.Hook.*SAMPLE.*loaded" 20 >/dev/null
hook qidk.debug.SD --es mode normal --es prompt "empty waters" --ei seed 42; wait_log "QIDK.Hook.*SD.*sha256=" 180 >/tmp/qidk_sd1.txt
hook qidk.debug.SD --es mode normal --es prompt "empty waters" --ei seed 42; wait_log "QIDK.Hook.*SD.*sha256=" 180 >/tmp/qidk_sd2.txt
hook qidk.debug.SD --es mode normal --es prompt "dense green coral reef, colourful fish" --ei seed 42; wait_log "QIDK.Hook.*SD.*sha256=" 180 >/tmp/qidk_sd3.txt
h1=$(grep -o 'sha256=[0-9a-f]*' /tmp/qidk_sd1.txt); h2=$(grep -o 'sha256=[0-9a-f]*' /tmp/qidk_sd2.txt); h3=$(grep -o 'sha256=[0-9a-f]*' /tmp/qidk_sd3.txt)
step "same prompt + seed -> bit-identical output" test -n "$h1" -a "$h1" = "$h2"
step "different prompt -> different output" test -n "$h3" -a "$h1" != "$h3"
step "runner ran the requested step count" bash -c "$ADB logcat -d | grep -E 'steps (12|20)' | head -n1 | grep -q steps"

echo "== 7. Super Resolution (Phase 5)"
hook qidk.debug.SR --es engine bicubic --es backend CPU
sleep 4
step "bicubic SR returns 1024x1024" bash -c "$ADB logcat -d | grep -E 'QIDK.SR|QIDK.Hook.*SR' | grep -q 'out=1024x1024'"

echo "== 8. LinUCB router (Phase 6)"
hook qidk.debug.ROUTE
line="$(wait_log 'QIDK.Router: ' 20)"; echo "      ${line:-no router log line}"
step "router decided (or says exactly why it cannot)" bash -c "echo '$line' | grep -qE 'mode=|unusable'"
step "router: LinUCB scoring < 100 us" bash -c "echo '$line' | grep -oE 'policy_us=[0-9]+' | cut -d= -f2 | awk '{exit !(\$1 < 100)}'"
step "router: feature extraction < 1000 us" bash -c "echo '$line' | grep -oE 'feat_us=[0-9]+' | cut -d= -f2 | awk '{exit !(\$1 < 1000)}'"

echo "== 9. crashes"
step "no crash / ANR in the whole run" bash -c "! $ADB logcat -d -b crash | grep -q . && ! $ADB logcat -d | grep -E 'FATAL EXCEPTION|ANR in $PKG'"

echo
echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
