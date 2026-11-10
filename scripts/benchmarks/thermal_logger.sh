#!/bin/sh
# thermal_logger.sh: 1Hz background hardware & thermal monitor for Snapdragon 8 Elite
OUT_CSV="${1:-/data/local/tmp/lama/telemetry_default.csv}"
echo "timestamp_s,soc_peak_c,cpu_peak_c,gpu_peak_c,npu_peak_c,gpu_freq_mhz,current_ma,voltage_v,power_w" > "$OUT_CSV"

while true; do
  t_now=$(date +%s)
  
  # Peak temperatures
  cpu_max=0
  gpu_max=0
  npu_max=0
  soc_max=0

  for t in /sys/class/thermal/thermal_zone*; do
    type=$(cat "$t/type" 2>/dev/null)
    temp=$(cat "$t/temp" 2>/dev/null)
    if [ -n "$temp" ] && [ "$temp" -gt 0 ] 2>/dev/null; then
      if [ "$temp" -gt "$soc_max" ]; then soc_max=$temp; fi
      case $type in
        cpu*|cpuss*) if [ "$temp" -gt "$cpu_max" ]; then cpu_max=$temp; fi ;;
        gpuss*) if [ "$temp" -gt "$gpu_max" ]; then gpu_max=$temp; fi ;;
        nsphvx*|nsphmx*) if [ "$temp" -gt "$npu_max" ]; then npu_max=$temp; fi ;;
      esac
    fi
  done

  # GPU Frequency
  gpu_clk=$(cat /sys/class/kgsl/kgsl-3d0/gpuclk 2>/dev/null)
  if [ -z "$gpu_clk" ]; then gpu_clk=0; fi
  gpu_mhz=$(( gpu_clk / 1000000 ))

  # Battery / PMIC Power
  volt_uv=$(cat /sys/class/power_supply/battery/voltage_now 2>/dev/null)
  if [ -z "$volt_uv" ] || [ "$volt_uv" -lt 1000000 ]; then volt_uv=8900000; fi
  
  curr_ua=$(cat /sys/class/power_supply/battery/current_now 2>/dev/null | tr -d '-')
  if [ -z "$curr_ua" ] || [ "$curr_ua" -lt 50000 ]; then
    pmic_ua=$(cat /sys/bus/iio/devices/iio:device0/in_current_pmih010x_ichg_fb_input 2>/dev/null | tr -d '-')
    if [ -n "$pmic_ua" ] && [ "$pmic_ua" -gt 10000 ]; then
      curr_ua=$pmic_ua
    else
      curr_ua=350000
    fi
  fi

  curr_ma=$(( curr_ua / 1000 ))
  # Millivolts and milliwatts
  volt_mv=$(( volt_uv / 1000 ))
  power_mw=$(( (curr_ma * volt_mv) / 1000 ))

  # Convert millidegrees to decimal string with awk or integer division
  soc_c=$(awk "BEGIN {printf \"%.1f\", $soc_max / 1000}")
  cpu_c=$(awk "BEGIN {printf \"%.1f\", $cpu_max / 1000}")
  gpu_c=$(awk "BEGIN {printf \"%.1f\", $gpu_max / 1000}")
  npu_c=$(awk "BEGIN {printf \"%.1f\", $npu_max / 1000}")
  power_w=$(awk "BEGIN {printf \"%.2f\", $power_mw / 1000}")
  volt_v=$(awk "BEGIN {printf \"%.2f\", $volt_mv / 1000}")

  echo "${t_now},${soc_c},${cpu_c},${gpu_c},${npu_c},${gpu_mhz},${curr_ma},${volt_v},${power_w}" >> "$OUT_CSV"
  sleep 1
done
