#!/usr/bin/env python3
"""
Regenerate Qualitative Inpainting Grids with Fixed Index Alignment and Accurate Domain Labels.

Fixes:
  1. Prevents off-by-one SD alignment: matches 1_sd.png, 2_sd.png, etc. accurately.
  2. Relabels domains accurately: Natural Scenes (Places) vs Structured Objects & Still Life.
  3. Annotates local Hole PSNR and LPIPS scores from CSV.
  4. Exports high-quality, lightweight JPGs (05_gan_domain_stress_6grid.jpg, 06_sd_vs_gans_5grid.jpg).
"""

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser("~/Desktop/college/ImageInpainting")
CSV_PATH = os.path.join(BASE_DIR, "Benchmark/output/all_pairs_detailed_metrics.csv")
IMG_DIR  = os.path.join(BASE_DIR, "Benchmark/input/image")
MASK_DIR = os.path.join(BASE_DIR, "Benchmark/input/mask")
OUT_DIR  = os.path.join(BASE_DIR, "Benchmark/output")
FIG_DIR  = os.path.join(OUT_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)

df = None
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"📖 Loaded {len(df)} records from {CSV_PATH}")


def make_corrupted(gt_img, mask_path):
    mask = Image.open(mask_path).convert("L").resize(gt_img.size, Image.Resampling.NEAREST)
    m_arr = np.array(mask) >= 128
    c_arr = np.array(gt_img).copy()
    c_arr[m_arr] = [255, 50, 50]  # Red mask overlay
    return Image.fromarray(c_arr)


def find_prediction_file(model, sample_idx):
    """Accurately maps 1-indexed benchmark sample to model prediction file."""
    res_dir = os.path.join(OUT_DIR, model, "results")
    if model == "sd":
        cand_names = [
            f"{sample_idx}_{model}.png",
            f"{sample_idx}.png",
            f"{sample_idx - 1}_{model}.png",
            f"{sample_idx - 1}.png"
        ]
    else:
        cand_names = [
            f"{sample_idx - 1}_{model}.png",
            f"{sample_idx - 1}.png",
            f"{sample_idx}_{model}.png",
            f"{sample_idx}.png"
        ]
    for cn in cand_names:
        p = os.path.join(res_dir, cn)
        if os.path.exists(p):
            return p
    return None


def get_metric_caption(model, sample_idx):
    """Retrieves local Hole PSNR and LPIPS score for the cell."""
    if df is None:
        return ""
    row = df[(df["model"] == model) & (df["sample_idx"] == sample_idx)]
    if len(row) == 0:
        return ""
    rec = row.iloc[0]
    return f"Hole: {rec['hole_psnr']:.1f} dB | LPIPS: {rec['lpips']:.3f}"


# -------------------------------------------------------------
# FIX GRID 2: Stable Diffusion vs GANs (Exact Sample 1 to 5 Alignment)
# -------------------------------------------------------------
print("\n🔧 Regenerating Grid 2 (SD vs GANs with exact alignment)...")
sd_samples = [1, 2, 3, 4, 5]
fig2, axes2 = plt.subplots(5, 6, figsize=(22, 18), dpi=200)
col_titles_2 = ["Ground Truth", "Corrupted Input", "MIGAN", "AOT-GAN", "LaMa Dilated", "Stable Diffusion 1.5"]

for r_idx, s_idx in enumerate(sd_samples):
    gt_p = os.path.join(IMG_DIR, f"{s_idx}.png")
    mk_p = os.path.join(MASK_DIR, f"{s_idx}.png")
    gt_pil = Image.open(gt_p).convert("RGB").resize((512, 512))
    cr_pil = make_corrupted(gt_pil, mk_p)
    
    pct_str = ""
    if df is not None:
        sub_pct = df[df["sample_idx"] == s_idx]["mask_area_pct"]
        if len(sub_pct) > 0:
            pct_str = f"\n({sub_pct.iloc[0]:.1f}% Mask)"

    axes2[r_idx, 0].imshow(gt_pil)
    axes2[r_idx, 0].set_ylabel(f"Sample #{s_idx}{pct_str}", fontsize=11, fontweight="bold")
    axes2[r_idx, 1].imshow(cr_pil)
    
    for c_idx, m in enumerate(["migan", "aotgan", "lama", "sd"], start=2):
        pred_p = find_prediction_file(m, s_idx)
        if pred_p and os.path.exists(pred_p):
            pr_pil = Image.open(pred_p).convert("RGB").resize((512, 512))
            axes2[r_idx, c_idx].imshow(pr_pil)
            caption = get_metric_caption(m, s_idx)
            if caption:
                axes2[r_idx, c_idx].set_xlabel(caption, fontsize=9, fontweight="bold", color="#1d3557", labelpad=4)
        else:
            axes2[r_idx, c_idx].set_facecolor("#e5e7eb")

for ax in axes2.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

for c_idx, title in enumerate(col_titles_2):
    axes2[0, c_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

plt.suptitle("Generative Latent Diffusion vs. Feed-Forward GANs: Qualcomm Hexagon HTP v79", fontsize=15, fontweight="bold", y=0.995)
plt.tight_layout()
out_fig2 = os.path.join(FIG_DIR, "06_sd_vs_gans_5grid.jpg")
plt.savefig(out_fig2, format="jpg", pil_kwargs={"quality": 92, "optimize": True})
plt.close()
print(f"✅ Fixed Grid 2 saved to: {out_fig2}")

# -------------------------------------------------------------
# FIX GRID 1: Relabel Subsets Accurately (Natural Scenes vs Structured Objects)
# -------------------------------------------------------------
print("\n🔧 Regenerating Grid 1 (Accurate Domain Labels)...")
sel_samples = [
    (155, "Places — Light (<15%)"),
    (1,   "Places — Medium (15–25%)"),
    (83,  "Places — Heavy (>25%)"),
    (40,  "Objects/Still Life — Light (<15%)"),
    (67,  "Objects/Still Life — Medium (15–25%)"),
    (190, "Objects/Still Life — Heavy (>25%)")
]

fig1, axes1 = plt.subplots(6, 5, figsize=(18, 22), dpi=200)
col_titles_1 = ["Ground Truth", "Corrupted Input", "MIGAN", "AOT-GAN", "LaMa Dilated"]

for r_idx, (s_idx, row_lbl) in enumerate(sel_samples):
    gt_p = os.path.join(IMG_DIR, f"{s_idx}.png")
    mk_p = os.path.join(MASK_DIR, f"{s_idx}.png")
    gt_pil = Image.open(gt_p).convert("RGB").resize((512, 512))
    cr_pil = make_corrupted(gt_pil, mk_p)
    
    axes1[r_idx, 0].imshow(gt_pil)
    axes1[r_idx, 0].set_ylabel(row_lbl, fontsize=10, fontweight="bold")
    axes1[r_idx, 1].imshow(cr_pil)
    
    for c_idx, m in enumerate(["migan", "aotgan", "lama"], start=2):
        pred_p = find_prediction_file(m, s_idx)
        if pred_p and os.path.exists(pred_p):
            pr_pil = Image.open(pred_p).convert("RGB").resize((512, 512))
            axes1[r_idx, c_idx].imshow(pr_pil)
            caption = get_metric_caption(m, s_idx)
            if caption:
                axes1[r_idx, c_idx].set_xlabel(caption, fontsize=9, fontweight="bold", color="#1d3557", labelpad=4)
        else:
            axes1[r_idx, c_idx].set_facecolor("#e5e7eb")

for ax in axes1.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

for c_idx, title in enumerate(col_titles_1):
    axes1[0, c_idx].set_title(title, fontsize=12, fontweight="bold", pad=8)

plt.suptitle("GAN Architectural Stress Benchmark: Snapdragon 8 Elite Hexagon NPU", fontsize=15, fontweight="bold", y=0.995)
plt.tight_layout()
out_fig1 = os.path.join(FIG_DIR, "05_gan_domain_stress_6grid.jpg")
plt.savefig(out_fig1, format="jpg", pil_kwargs={"quality": 92, "optimize": True})
plt.close()
print(f"✅ Fixed Grid 1 saved to: {out_fig1}")
