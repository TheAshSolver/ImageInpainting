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


def find_pred_file(pred_dir, index_gt):
    """Finds matching prediction file for index N.

    Checks both 0-indexed (N-1) and 1-indexed (N) filenames.
    """
    patterns = [
        f"{index_gt - 1}_output_0.*",
        f"{index_gt - 1}.*",
        f"{index_gt}_output_0.*",
        f"{index_gt}.*",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(pred_dir, pat))
        if matches:
            return matches[0]
    return None


def run_batch_metrics(gt_dir, pred_dir, csv_output="metrics_summary.csv"):
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

    results = []
    psnr_values = []
    ssim_values = []

    print(f"Found {len(gt_files)} ground truth images. Computing metrics...\n")

    for gt_path in gt_files:
        idx = extract_index(gt_path)
        if idx is None:
            continue

        pred_path = find_pred_file(pred_dir, idx)
        if not pred_path:
            print(f"[Skip] No matching output found for: {os.path.basename(gt_path)} (Index {idx})")
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
                "PSNR_dB": round(psnr_val, 4),
                "SSIM": round(ssim_val, 4),
            })

            print(f"Pair {idx:04d} -> PSNR: {psnr_val:.4f} dB | SSIM: {ssim_val:.4f}")

        except Exception as e:
            print(f"[Error] Failed on pair {idx}: {e}")

    if not results:
        print("No valid image pairs were evaluated.")
        return

    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))

    # Write per-pair metrics and append the final average row
    with open(csv_output, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["Index", "GT_File", "Pred_File", "PSNR_dB", "SSIM"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            "Index": "AVERAGE",
            "GT_File": f"{len(results)} pairs",
            "Pred_File": "-",
            "PSNR_dB": round(avg_psnr, 4),
            "SSIM": round(avg_ssim, 4),
        })

    print("\n" + "=" * 45)
    print("                 EVALUATION SUMMARY          ")
    print("=" * 45)
    print(f"Total Evaluated Pairs : {len(results)}")
    print(f"Average PSNR (dB)     : {avg_psnr:.4f} dB")
    print(f"Average SSIM          : {avg_ssim:.4f}")
    print(f"CSV Report Saved To   : {os.path.abspath(csv_output)}")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch compare GT and prediction images using PSNR and SSIM.")
    parser.add_argument("--gt_dir", required=True, type=str, help="Folder containing ground truth images (e.g. 1.png)")
    parser.add_argument("--pred_dir", required=True, type=str, help="Folder containing model predictions (e.g. 0_output_0.png)")
    parser.add_argument("--output_csv", type=str, default="metrics_summary.csv", help="Path to output CSV file")
    args = parser.parse_args()

    run_batch_metrics(args.gt_dir, args.pred_dir, args.output_csv)