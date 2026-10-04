#!/system/bin/sh
# usage: bench_cfg.sh TAG DLC LISTSUFFIX RUNTIME_FLAG [COOL_MC] [COOL_TIMEOUT_S]
# Measures cold (N=1) and warm (N=102) wall-clock of snpe-net-run, with compute-zone temps around every run.
TAG=$1; DLC=$2; LS=$3; RT=$4; COOL=${5:-42000}; CTO=${6:-240}
cd /data/local/tmp/lama
export LD_LIBRARY_PATH=/data/local/tmp/lama/lib:/data/local/tmp/lama:/data/local/tmp/sd_runtime:vendor/lib64:/system/lib64
export ADSP_LIBRARY_PATH='/data/local/tmp/lama/dsp/lib;/data/local/tmp/lama/dsp;/data/local/tmp/sd_runtime;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'

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
# monotonic clock (wall clock on this board steps); /proc/uptime has 10 ms resolution
now_ms() { read u _ < /proc/uptime; echo $(( ${u%.*} * 1000 + 10 * ${u#*.} )); }

# --- cooldown barrier on the hottest compute zone
t_wait0=$(now_ms)
while :; do
  h=$(mx $ALLZ)
  [ "$h" -le "$COOL" ] && break
  el=$(( ( $(now_ms) - t_wait0 ) / 1000 ))
  [ "$el" -ge "$CTO" ] && break
  sleep 3
done
echo "COOLDOWN $TAG waited_s=$(( ( $(now_ms) - t_wait0 ) / 1000 )) hottest_mC=$(mx $ALLZ) cpu=$(mx $CPUZ) gpu=$(mx $GPUZ) nsp=$(mx $NSPZ)"

run_once() {  # kind rep list outdir
  kind=$1; rep=$2; list=$3; od=$4
  rm -rf $od
  sc=$(mx $CPUZ); sg=$(mx $GPUZ); sn=$(mx $NSPZ)
  t0=$(now_ms)
  ./snpe-net-run --container $DLC --input_list $list --output_dir $od $RT --perf_profile burst >/data/local/tmp/lama/bench_stdout_$TAG.txt 2>&1
  rc=$?
  t1=$(now_ms)
  ec=$(mx $CPUZ); eg=$(mx $GPUZ); en=$(mx $NSPZ)
  nres=$(ls -d $od/Result_* 2>/dev/null | wc -l)
  echo "RESULT $TAG $kind rep=$rep rc=$rc ms=$(( t1 - t0 )) results=$nres start_mC cpu=$sc gpu=$sg nsp=$sn end_mC cpu=$ec gpu=$eg nsp=$en"
}

L1=list1_$LS.txt; LN=list_$LS.txt
# warm-up so that first-ever load of the .dlc from flash is not charged to cold run 1
run_once warmup 0 $L1 bench_out_${TAG}_warm >/dev/null
for r in 1 2 3; do run_once cold $r $L1 bench_out_${TAG}_cold; sleep 2; done
for r in 1 2 3; do run_once batch102 $r $LN bench_out_${TAG}_r$r; done
echo "DONE $TAG"
