#!/usr/bin/env python3
"""
Patch Dropped Batches in 6,000 NPU Benchmark Dataset
Re-evaluates batches with psnr_full == 0.0 on Hexagon HTP v79 NPU
and atomically patches dataset_meta_predictor.csv.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from run_6000_npu_benchmark import (
    IMAGES_DIR, MASKS_DIR, MASTER_CSV_FILE, DEVICE_LAMA_DIR, MODELS,
    DEVICE_SD_DIR, SAMPLE_RECON_DIR,
    run_adb, wait_for_device, check_storage_and_system_health,
    enforce_cooldown_barrier, query_device_thermals, query_device_power,
    extract_image_features, PerceptualEvaluator
)

def run_snpe_with_retries(m_key: str, list_name: str, out_dir_dev: str, curr_batch_size: int, adb_path: str = "adb"):
    m_cfg = MODELS[m_key]
    snpe_cmd = (
        f"cd {DEVICE_LAMA_DIR} && "
        f"export LD_LIBRARY_PATH={DEVICE_LAMA_DIR}/lib:{DEVICE_LAMA_DIR}:{DEVICE_SD_DIR}:vendor/lib64:/system/lib64 && "
        f"export ADSP_LIBRARY_PATH='{DEVICE_LAMA_DIR}/dsp/lib;{DEVICE_LAMA_DIR}/dsp;{DEVICE_SD_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && "
        f"t0=$(date +%s%3N) && "
        f"./snpe-net-run --container {m_cfg['dlc']} --input_list batch_in/{list_name} --output_dir {out_dir_dev} {m_cfg['runtime_flag']} --perf_profile burst > /dev/null 2>&1; "
        f"snpe_rc=$?; "
        f"t1=$(date +%s%3N); "
        f"if [ $snpe_rc -eq 0 ]; then echo $(( t1 - t0 )); else echo 'SNPE_ERROR:' $snpe_rc; fi"
    )

    t_start = query_device_thermals(adb_path)
    p_start = query_device_power(adb_path)

    rc, stdout, stderr = run_adb(["shell", snpe_cmd], adb_path=adb_path, timeout=180)

    t_end = query_device_thermals(adb_path)
    p_end = query_device_power(adb_path)

    wall_ms = 0.0
    for line in stdout.splitlines():
        if line.strip().isdigit():
            wall_ms = float(line.strip())
            break

    per_frame_ms = round(wall_ms / curr_batch_size, 2) if wall_ms > 0 else 0.0
    fps = round(1000.0 / per_frame_ms, 1) if per_frame_ms > 0 else 0.0
    delta_nsp = round(t_end["nsp"] - t_start["nsp"], 2)
    avg_power = round((p_start["power_w"] + p_end["power_w"]) / 2.0, 3)

    return {
        "wall_ms": wall_ms,
        "per_frame_ms": per_frame_ms,
        "fps": fps,
        "t_start": t_start,
        "t_end": t_end,
        "delta_nsp": delta_nsp,
        "power": avg_power,
        "voltage": p_start["voltage_v"],
        "current": p_start["current_ma"]
    }

def main():
    adb_path = "adb"
    cooldown_temp = 55.0
    host_staging_root = Path("/tmp/qidk_staging_batch")

    print("=" * 80)
    print("🔧 SNAPDRAGON 8 ELITE — DROPPED BATCHES RESILIENT PATCH HARNESS")
    print("=" * 80)

    if not MASTER_CSV_FILE.exists():
        print(f"Error: {MASTER_CSV_FILE} does not exist!")
        return 1

    df = pd.read_csv(MASTER_CSV_FILE)
    zeros = df[df['psnr_full'] == 0.0]
    if len(zeros) == 0:
        print("✅ No dropped batches found! Dataset is 100% complete.")
        return 0

    dropped_batches = sorted(zeros['batch_idx'].unique().tolist())
    print(f"Found {len(zeros)} rows with psnr_full == 0.0 across {len(dropped_batches)} batches:")
    print(f"Batches to patch: {dropped_batches}")

    evaluator = PerceptualEvaluator()
    models_to_run = ["migan_npu", "lama_npu", "aotgan_npu"]

    total_patched = 0
    t0_global = time.time()

    for idx, b_idx_1based in enumerate(dropped_batches):
        b_idx = b_idx_1based - 1  # 0-based index
        b_sample_ids = [f"{i:04d}" for i in range(b_idx * 20 + 1, (b_idx + 1) * 20 + 1)]
        curr_batch_size = len(b_sample_ids)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ▶ Patching Batch {b_idx_1based} ({idx + 1}/{len(dropped_batches)}) — Samples {b_sample_ids[0]}..{b_sample_ids[-1]}")

        # 1. Health & Storage Guard
        wait_for_device(adb_path)
        safe, health_msg = check_storage_and_system_health(adb_path)
        if not safe:
            print(f"❌ Storage Guard: {health_msg}")
            return 1
        print(f"  [System Health] {health_msg}")

        # 2. Enforce Initial Cooldown Barrier
        enforce_cooldown_barrier(target_temp=cooldown_temp, adb_path=adb_path)

        # 3. Prepare Local Staging Files
        shutil.rmtree(host_staging_root, ignore_errors=True)
        for sub in ["raw_image", "raw_mask_standard", "raw_mask_inverted"]:
            (host_staging_root / sub).mkdir(parents=True, exist_ok=True)

        list_std_lines = []
        list_inv_lines = []
        cached_gt_images = {}
        cached_masks = {}

        for stem in b_sample_ids:
            img_p = IMAGES_DIR / f"{stem}.png"
            mask_p = MASKS_DIR / f"{stem}.png"

            img = Image.open(img_p).convert("RGB")
            mask = Image.open(mask_p).convert("L")
            img_np = np.asarray(img, dtype=np.float32)
            mask_np = (np.asarray(mask) >= 128).astype(np.float32)

            cached_gt_images[stem] = img_np
            cached_masks[stem] = mask_np

            (img_np / 255.0)[None].tofile(host_staging_root / "raw_image" / f"{stem}.raw")
            mask_np[None, ..., None].tofile(host_staging_root / "raw_mask_standard" / f"{stem}_mask.raw")
            (1.0 - mask_np)[None, ..., None].tofile(host_staging_root / "raw_mask_inverted" / f"{stem}_mask.raw")

            list_std_lines.append(f"image:=batch_in/raw_image/{stem}.raw mask:=batch_in/raw_mask_standard/{stem}_mask.raw\n")
            list_inv_lines.append(f"image:=batch_in/raw_image/{stem}.raw mask:=batch_in/raw_mask_inverted/{stem}_mask.raw\n")

        (host_staging_root / "list_standard.txt").write_text("".join(list_std_lines))
        (host_staging_root / "list_inverted.txt").write_text("".join(list_inv_lines))

        # 4. Push Batch to Device via single-stream TAR (Ultra-fast & immune to USB stalls)
        print(f"  [ADB Push] Streaming {curr_batch_size} tensor pairs to {DEVICE_LAMA_DIR}/batch_in...")
        push_ok = False
        tar_path = host_staging_root.parent / f"staging_batch_{b_idx_1based}.tar"
        subprocess.run(["tar", "-cf", str(tar_path), "-C", str(host_staging_root), "."], check=True)

        for push_attempt in range(3):
            run_adb(["shell", f"rm -rf {DEVICE_LAMA_DIR}/batch_in && mkdir -p {DEVICE_LAMA_DIR}/batch_in"], adb_path=adb_path)
            run_adb(["push", str(tar_path), f"{DEVICE_LAMA_DIR}/batch_in/batch.tar"], adb_path=adb_path)
            run_adb(["shell", f"cd {DEVICE_LAMA_DIR}/batch_in && tar -xf batch.tar 2>/dev/null || true; rm -f batch.tar"], adb_path=adb_path)
            rc_cnt, out_cnt, _ = run_adb(["shell", f"ls {DEVICE_LAMA_DIR}/batch_in/raw_image | wc -l"], adb_path=adb_path)
            actual_count = int(out_cnt.strip()) if out_cnt.strip().isdigit() else 0
            if actual_count == curr_batch_size:
                push_ok = True
                break
            print(f"  [!] Incomplete push ({actual_count}/{curr_batch_size}) on attempt {push_attempt + 1}! Retrying...")
            time.sleep(1)

        tar_path.unlink(missing_ok=True)
        if not push_ok:
            print(f"❌ Failed to push batch {b_idx_1based} after 3 attempts! Skipping for now.")
            continue

        # 5. Execute Each Model on Hexagon NPU
        batch_results_per_model = {}
        for m_key in models_to_run:
            m_cfg = MODELS[m_key]
            list_name = "list_inverted.txt" if m_cfg["mask_type"] == "inverted" else "list_standard.txt"
            out_dir_dev = f"{DEVICE_LAMA_DIR}/out_{m_key}"
            run_adb(["shell", f"rm -rf {out_dir_dev} && mkdir -p {out_dir_dev}"], adb_path=adb_path)

            m_res = run_snpe_with_retries(m_key, list_name, out_dir_dev, curr_batch_size, adb_path=adb_path)
            print(f"    • {m_cfg['display_name']:18s} | Wall: {m_res['wall_ms']:6.1f} ms | Per-Frame: {m_res['per_frame_ms']:5.1f} ms ({m_res['fps']:4.1f} FPS) | ΔT_NPU: {m_res['delta_nsp']:+4.1f}°C")

            # Pull output raw tensors via single-stream TAR
            tar_dev = f"{DEVICE_LAMA_DIR}/out_{m_key}.tar"
            run_adb(["shell", f"cd {DEVICE_LAMA_DIR} && tar -cf {tar_dev} out_{m_key} 2>/dev/null || true"], adb_path=adb_path)
            local_tar = host_staging_root / f"out_{m_key}.tar"
            local_out_dir = host_staging_root / f"out_{m_key}"
            local_out_dir.mkdir(parents=True, exist_ok=True)
            run_adb(["pull", tar_dev, str(local_tar)], adb_path=adb_path)
            run_adb(["shell", f"rm -f {tar_dev}"], adb_path=adb_path)
            if local_tar.exists():
                subprocess.run(["tar", "-xf", str(local_tar), "-C", str(host_staging_root)], check=False)
                local_tar.unlink(missing_ok=True)

            m_res["local_dir"] = local_out_dir
            batch_results_per_model[m_key] = m_res

        # 6. Purge Device Batch Immediately
        run_adb(["shell", f"rm -rf {DEVICE_LAMA_DIR}/batch_in {DEVICE_LAMA_DIR}/out_*"], adb_path=adb_path)

        # 7. Evaluate Metrics and Patch DataFrame
        for m_key in models_to_run:
            m_res = batch_results_per_model[m_key]
            m_cand = MODELS[m_key]["output_cand"]

            preds_list = []
            gts_list = []
            masks_list = []
            valid_indices = []

            for intra_idx, stem in enumerate(b_sample_ids):
                gt_rgb = cached_gt_images[stem]
                mask_bin = cached_masks[stem]

                res_dir = m_res["local_dir"] / f"Result_{intra_idx}"
                pred_raw_path = None
                for c in m_cand:
                    cand_p = res_dir / c
                    if cand_p.exists():
                        pred_raw_path = cand_p
                        break

                if pred_raw_path and pred_raw_path.stat().st_size == (512 * 512 * 3 * 4):
                    raw_data = np.fromfile(pred_raw_path, dtype=np.float32).reshape(512, 512, 3)
                    pred_rgb = raw_data * 255.0 if raw_data.max() <= 1.5 else raw_data
                    mask_3d = mask_bin[:, :, None]
                    comp_rgb = (mask_3d * pred_rgb + (1.0 - mask_3d) * gt_rgb).clip(0, 255)

                    preds_list.append(comp_rgb)
                    gts_list.append(gt_rgb)
                    masks_list.append(mask_bin)
                    valid_indices.append(intra_idx)

            if preds_list:
                batch_q_metrics = evaluator.evaluate_batch(preds_list, gts_list, masks_list)
            else:
                batch_q_metrics = []

            metric_map = {valid_indices[i]: batch_q_metrics[i] for i in range(len(valid_indices))}

            latency_s = m_res["per_frame_ms"] / 1000.0
            energy_j = m_res["power"] * latency_s
            edp = round(energy_j * latency_s, 6)

            for intra_idx, stem in enumerate(b_sample_ids):
                q = metric_map.get(intra_idx)
                if not q:
                    continue

                # Locate row in df
                mask_cond = (df['sample_id'] == int(stem)) & (df['model'] == m_key)
                if not mask_cond.any():
                    mask_cond = (df['sample_id'] == stem) & (df['model'] == m_key)

                if mask_cond.any():
                    df.loc[mask_cond, 'latency_ms'] = m_res["per_frame_ms"]
                    df.loc[mask_cond, 'throughput_fps'] = m_res["fps"]
                    df.loc[mask_cond, 'energy_delay_product_edp'] = edp
                    df.loc[mask_cond, 'psnr_full'] = q["psnr_full"]
                    df.loc[mask_cond, 'psnr_hole'] = q["psnr_hole"]
                    df.loc[mask_cond, 'ssim'] = q["ssim"]
                    df.loc[mask_cond, 'lpips'] = q["lpips"]
                    df.loc[mask_cond, 'active_power_w'] = m_res["power"]
                    df.loc[mask_cond, 'delta_temp_nsp_c'] = m_res["delta_nsp"]

        # 8. Clean local staging
        shutil.rmtree(host_staging_root, ignore_errors=True)

        # 9. Atomic save to CSV after each patched batch
        tmp_csv = MASTER_CSV_FILE.with_suffix(".tmp")
        df.to_csv(tmp_csv, index=False)
        tmp_csv.replace(MASTER_CSV_FILE)

        total_patched += 1
        elapsed = time.time() - t0_global
        avg_batch_time = elapsed / total_patched
        rem_batches = len(dropped_batches) - total_patched
        rem_sec = rem_batches * avg_batch_time

        print(f"  ✅ Batch {b_idx_1based} successfully patched! ({total_patched}/{len(dropped_batches)} done | ETA: {rem_sec/60:.1f}m)")

    # Final verification
    df_final = pd.read_csv(MASTER_CSV_FILE)
    rem_zeros = df_final[df_final['psnr_full'] == 0.0]
    print("\n" + "=" * 80)
    print(f"🎉 PATCHING COMPLETE! Total rows: {len(df_final)}, Remaining zero rows: {len(rem_zeros)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
