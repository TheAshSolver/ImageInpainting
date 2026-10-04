#!/system/bin/sh
# Build SNPE HTP init-cache copies of the slow-to-launch DLCs ON THE DEVICE (one-time, ~8 s each).
#   adb push scripts/fresh_benchmark/make_init_cache.sh /data/local/tmp/ && adb shell sh /data/local/tmp/make_init_cache.sh
# Result: bench_cache_<name>.dlc next to the original. Cold launch drops from ~8 s to ~0.5 s for LaMa / AOT-GAN,
# outputs are bit-identical. Originals are left untouched. Point snpe-net-run --container at the cached copy
# (keep passing --enable_init_cache so SNPE uses the embedded cache).
cd /data/local/tmp/lama || exit 1
export LD_LIBRARY_PATH=/data/local/tmp/lama/lib:/data/local/tmp/lama:/data/local/tmp/sd_runtime:vendor/lib64:/system/lib64
export ADSP_LIBRARY_PATH='/data/local/tmp/lama/dsp/lib;/data/local/tmp/lama/dsp;/data/local/tmp/sd_runtime;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'
# needs one valid input list: image:=<raw image> mask:=<raw standard mask>
LIST=${1:-list1_standard.txt}
for M in lama_dilated aotgan; do
  cp $M.dlc bench_cache_$M.dlc
  rm -rf bench_cache_out
  ./snpe-net-run --container bench_cache_$M.dlc --input_list $LIST --output_dir bench_cache_out --use_dsp --perf_profile burst --enable_init_cache >/dev/null 2>&1
  echo "$M: cache built, $(stat -c %s bench_cache_$M.dlc) bytes (original $(stat -c %s $M.dlc))"
done
rm -rf bench_cache_out
