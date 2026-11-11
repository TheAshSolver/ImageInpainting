#!/system/bin/sh
# Runs ON the device:   adb push tools/sd_probe.sh /data/local/tmp/ && adb -s 8f27557f shell sh /data/local/tmp/sd_probe.sh
# Needs image.raw / mask.raw in /data/local/tmp/sd_runtime (the app writes them on every SD run: press Run once first).
# Shows what the runner really received (steps) and whether the output depends on the prompt (md5 prefix of sd_output.png).
cd /data/local/tmp/sd_runtime || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/sd_runtime:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/sd_runtime;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
RUNNER=${RUNNER:-./sd_qidk_runner_inpaint}
run() {
  rm -f sd_output.png
  "$@" 2>&1 | grep -E -i "steps|FATAL|\[ARGS\]" | head -2
  if [ -f sd_output.png ]; then echo "  output md5: $(md5sum sd_output.png | cut -c1-12)"; else echo "  no output written"; fi
}
echo "--- A  exactly how the old app calls it (negative prompt in argv[2]), prompt 1";  run $RUNNER "empty waters" ""
echo "--- A' same call, a completely different prompt";                                  run $RUNNER "dense green coral reef, colourful fish" ""
echo "--- B  steps in argv[2], prompt 1";                                                run $RUNNER "empty waters" 20
echo "--- C  steps in argv[2], prompt 2";                                                run $RUNNER "dense green coral reef, colourful fish" 20
echo "--- D  new CLI (only after the runner was rebuilt with sd_args.hpp)";             run $RUNNER --prompt "empty waters" --neg "boat" --steps 20 --cfg 2.0 --seed 42
echo "expected on the OLD runner: A and A' print the same md5 and a tiny step count (0/2); B and C differ from each other and show 20 steps."
