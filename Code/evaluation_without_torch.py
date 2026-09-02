import argparse
import csv
import glob
import os
import re
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def load_image(path):
    """Loads an image as an RGB float32 array in range [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def extract_index(filename):
    """Extracts integer index from filenames like '1.png'."""
    match = re.search(r"\d+", os.path.splitext(os.path.basename(filename))[0])
    return int(match.group(0)) if match else None


def infer_model(pred_dir):
    """Infers model name from directory path or directory contents."""
    path_lower = os.path.abspath(pred_dir).lower()
    for m in ["migan", "aotgan", "lama"]:
        if m in path_lower:
            return m
    try:
        files = os.listdir(pred_dir)
        for m in ["migan", "aotgan", "lama"]:
            if any(f.lower().endswith(f"_{m}.png") for f in files):
                return m
    except Exception:
        pass
    return None


def find_pred_file(pred_dir, index_gt, model=None, offset=1):
    """Strictly finds matching prediction file for ground truth index N and model.

    Strict index matching:
      - 0-indexed: index_gt - offset (e.g., GT '1.png' matches '0_{model}.png' when offset=1)
      - 1-indexed fallback: index_gt (e.g., GT '1.png' matches '1_{model}.png')

    Strict model matching:
      - Only matches files with exact pattern '{idx}_{model}.<ext>'.
      - Stale files from other models or generic files (e.g. '0_output_0.png', '0_painted_image.png')
        are NEVER picked up.
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    cand_indices = [index_gt - offset]
    if offset != 0:
        cand_indices.append(index_gt)

    if model:
        model_lower = model.lower()
        for p_idx in cand_indices:
            for ext in valid_exts:
                target = os.path.join(pred_dir, f"{p_idx}_{model_lower}{ext}")
                if os.path.isfile(target):
                    return target
        return None
    else:
        # If model not explicitly provided, check for known model-named files first
        for p_idx in cand_indices:
            for m in ["migan", "aotgan", "lama"]:
                for ext in valid_exts:
                    target = os.path.join(pred_dir, f"{p_idx}_{m}{ext}")
                    if os.path.isfile(target):
                        return target

        # Exact index match (no model suffix, but exact index with extension)
        for p_idx in cand_indices:
            for ext in valid_exts:
                target = os.path.join(pred_dir, f"{p_idx}{ext}")
                if os.path.isfile(target):
                    return target

        return None


def run_batch_metrics(gt_dir, pred_dir, csv_output="metrics_summary.csv", model=None, offset=1):
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    gt_files = [
        os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.lower().endswith(valid_exts)
    ]
    gt_files.sort(key=lambda x: extract_index(x) or 0)

    if not gt_files:
        print(f"No image files found in ground truth directory: {gt_dir}")
        return

    # Auto-infer model if not specified
    if not model:
        inferred = infer_model(pred_dir)
        if inferred:
            model = inferred
            print(f"🔍 Inferred model from prediction path: '{model}'")

    print(f"==================================================")
    print(f"  Ground Truth Dir : {gt_dir}")
    print(f"  Prediction Dir   : {pred_dir}")
    print(f"  Target Model     : {model.upper() if model else 'Any (Strict Index)'}")
    print(f"  Index Offset     : {offset} (Pred = GT - {offset})")
    print(f"  CSV Report       : {csv_output}")
    print(f"==================================================")

    results = []
    psnr_values = []
    ssim_values = []

    print(f"Found {len(gt_files)} ground truth images. Computing metrics...\n")

    for gt_path in gt_files:
        idx = extract_index(gt_path)
        if idx is None:
            continue

        pred_path = find_pred_file(pred_dir, idx, model=model, offset=offset)
        if not pred_path:
            expected_name = f"{idx - offset}_{model}.png" if model else f"{idx - offset}.*"
            print(f"⚠️ [Skip] No matching output found for: {os.path.basename(gt_path)} (Expected: {expected_name})")
            continue

        try:
            gt_img = load_image(gt_path)
            pred_img = load_image(pred_path)

            # Auto-resize prediction to match ground-truth resolution if they differ
            if gt_img.shape != pred_img.shape:
                h, w = gt_img.shape[:2]
                pred_pil = Image.fromarray((pred_img * 255).astype(np.uint8))
                pred_pil = pred_pil.resize((w, h), Image.BICUBIC)
                pred_img = np.array(pred_pil).astype(np.float32) / 255.0

            psnr_val = float(compute_psnr(gt_img, pred_img, data_range=1.0))
            ssim_val = float(compute_ssim(gt_img, pred_img, data_range=1.0, channel_axis=2))

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)

            results.append({
                "Index": idx,
                "GT_File": os.path.basename(gt_path),
                "Pred_File": os.path.basename(pred_path),
                "Model": model or "unknown",
                "PSNR_dB": round(psnr_val, 4),
                "SSIM": round(ssim_val, 4),
            })

            print(f"Pair {idx:04d} -> GT: {os.path.basename(gt_path)} <--> Pred: {os.path.basename(pred_path)} | PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.4f}")

        except Exception as e:
            print(f"[Error] Failed on pair {idx}: {e}")

    if not results:
        print("\n❌ No valid image pairs were evaluated. Please verify filenames and --model flag.")
        return

    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))

    # Write per-pair metrics and append the final average row
    with open(csv_output, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["Index", "GT_File", "Pred_File", "Model", "PSNR_dB", "SSIM"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            "Index": "AVERAGE",
            "GT_File": f"{len(results)} pairs",
            "Pred_File": "-",
            "Model": model or "-",
            "PSNR_dB": round(avg_psnr, 4),
            "SSIM": round(avg_ssim, 4),
        })

    print("\n" + "=" * 50)
    print("                 EVALUATION SUMMARY               ")
    print("=" * 50)
    print(f"Target Model          : {model.upper() if model else 'N/A'}")
    print(f"Total Evaluated Pairs : {len(results)}")
    print(f"Average PSNR (dB)     : {avg_psnr:.4f} dB")
    print(f"Average SSIM          : {avg_ssim:.4f}")
    print(f"CSV Report Saved To   : {os.path.abspath(csv_output)}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch compare GT and prediction images using PSNR and SSIM with strict index and model matching."
    )
    parser.add_argument("--gt_dir", required=True, type=str, help="Folder containing ground truth images (e.g. 1.png)")
    parser.add_argument("--pred_dir", required=True, type=str, help="Folder containing model predictions (e.g. 0_migan.png)")
    parser.add_argument("--model", "-m", type=str, default=None, choices=["migan", "aotgan", "lama"], help="Model name to strictly match predictions (e.g. migan, aotgan, lama)")
    parser.add_argument("--offset", type=int, default=1, help="Index offset: pred_index = gt_index - offset (default: 1, e.g. GT 1 -> Pred 0)")
    parser.add_argument("--output_csv", type=str, default="metrics_summary.csv", help="Path to output CSV file")
    args = parser.parse_args()

    run_batch_metrics(args.gt_dir, args.pred_dir, args.output_csv, model=args.model, offset=args.offset)