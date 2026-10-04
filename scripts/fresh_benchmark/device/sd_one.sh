#!/system/bin/sh
# usage: sd_one.sh STEM [PROMPT]   (image.raw / mask.raw already pushed into sd_runtime)
STEM=$1; PROMPT=${2:-cinematic photo restoration}
cd /data/local/tmp/sd_runtime
export LD_LIBRARY_PATH=/data/local/tmp/sd_runtime
export ADSP_LIBRARY_PATH='/data/local/tmp/sd_runtime;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'
CPUZ=""; GPUZ=""; NSPZ=""; ALLZ=""
for t in /sys/class/thermal/thermal_zone*; do
  ty=$(cat $t/type 2>/dev/null)
  case "$ty" in
    cpu-*|cpuss-*) CPUZ="$CPUZ $t/temp"; ALLZ="$ALLZ $t/temp";;
    gpuss-*)       GPUZ="$GPUZ $t/temp"; ALLZ="$ALLZ $t/temp";;
    nsphvx-*|nsphmx-*) NSPZ="$NSPZ $t/temp"; ALLZ="$ALLZ $t/temp";;
  esac
done
mx() { cat $@ 2>/dev/null | sort -n | tail -n 1; }
now_ms() { read u _ < /proc/uptime; echo $(( ${u%.*} * 1000 + 10 * ${u#*.} )); }
w0=$(now_ms)
while :; do
  h=$(mx $ALLZ); [ "$h" -le 45000 ] && break
  [ $(( ( $(now_ms) - w0 ) / 1000 )) -ge 120 ] && break
  sleep 3
done
sc=$(mx $CPUZ); sg=$(mx $GPUZ); sn=$(mx $NSPZ)
rm -f sd_output.png
t0=$(now_ms)
./sd_qidk_runner_inpaint "$PROMPT" > sd_stdout_$STEM.txt 2>&1
rc=$?
t1=$(now_ms)
ec=$(mx $CPUZ); eg=$(mx $GPUZ); en=$(mx $NSPZ)
echo "SDRESULT $STEM rc=$rc wall_ms=$(( t1 - t0 )) cooled_s=$(( ( t0 - w0 ) / 1000 )) start_mC cpu=$sc gpu=$sg nsp=$sn end_mC cpu=$ec gpu=$eg nsp=$en"
grep -E 'Profile|End-to-End' sd_stdout_$STEM.txt | sed "s/^/SDPROF $STEM /"
cp sd_output.png sd_out_$STEM.png
