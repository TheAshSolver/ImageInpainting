#!/usr/bin/env python3
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

if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} not found!")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df['domain'] = np.where(df['sample_idx'] <= 100, 'Places', 'Faces')

def get_annotated_cell(df_sub, model, sample_idx):
    row = df_sub[(df_sub['model'] == model) & (df_sub['sample_idx'] == sample_idx)]
    if len(row) == 0:
        return None, "N/A"
    rec = row.iloc[0]
    img = Image.open(rec['pred_path']).convert("RGB").resize((512, 512))
    lbl = f"Hole: {rec['hole_psnr']:.1f} dB\nLPIPS: {rec['lpips']:.3f}"
    return img, lbl

def make_corrupted_input(gt_img, mask_path):
    mask = Image.open(mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
    m_arr = np.array(mask) >= 128
    c_arr = np.array(gt_img).copy()
    c_arr[m_arr] = [255, 50, 50]  # Red mask overlay
    return Image.fromarray(c_arr)

# -------------------------------------------------------------
# GRID 1: GAN Architectural & Domain Stress Grid (6 Rows x 5 Cols)
# -------------------------------------------------------------
print("🎨 Building Grid 1: GAN Domain Stress Grid (6 Rows)...")

# Select representative sample indices
sel_rows = []
for dom in ['Places', 'Faces']:
    d_df = df[df['domain'] == dom]
    # Light (<15%)
    r_light = d_df[d_df['mask_area_pct'] < 15.0].sort_values('mask_area_pct', ascending=False).iloc[0]
    # Med (15-25%)
    r_med   = d_df[(d_df['mask_area_pct'] >= 15.0) & (d_df['mask_area_pct'] <= 25.0)].iloc[0]
    # Heavy (>25%)
    r_heavy = d_df[d_df['mask_area_pct'] > 25.0].sort_values('mask_area_pct', ascending=False).iloc[0]
    
    sel_rows.append((r_light['sample_idx'], f"{dom} — Light ({r_light['mask_area_pct']:.1f}%)"))
    sel_rows.append((r_med['sample_idx'],   f"{dom} — Medium ({r_med['mask_area_pct']:.1f}%)"))
    sel_rows.append((r_heavy['sample_idx'], f"{dom} — Heavy ({r_heavy['mask_area_pct']:.1f}%)"))

fig1, axes1 = plt.subplots(6, 5, figsize=(18, 22), dpi=300)
col_titles_1 = ["Ground Truth", "Corrupted Input", "MIGAN", "AOT-GAN", "LaMa Dilated"]

for r_idx, (s_idx, row_label) in enumerate(sel_rows):
    gt_p = os.path.join(IMG_DIR, f"{s_idx}.png")
    mk_p = os.path.join(MASK_DIR, f"{s_idx}.png")
    gt_pil = Image.open(gt_p).convert("RGB").resize((512, 512))
    cr_pil = make_corrupted_input(gt_pil, mk_p)
    
    # Col 0: GT
    axes1[r_idx, 0].imshow(gt_pil)
    axes1[r_idx, 0].set_ylabel(row_label, fontsize=10, fontweight='bold')
    
    # Col 1: Corrupted
    axes1[r_idx, 1].imshow(cr_pil)
    
    # Cols 2, 3, 4: Models
    for c_idx, m in enumerate(['migan', 'aotgan', 'lama'], start=2):
        img, caption = get_annotated_cell(df, m, s_idx)
        if img:
            axes1[r_idx, c_idx].imshow(img)
            axes1[r_idx, c_idx].set_xlabel(caption, fontsize=9, fontweight='bold', color='#1d3557')
        else:
            axes1[r_idx, c_idx].set_facecolor("#e5e7eb")

for ax in axes1.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

for c_idx, title in enumerate(col_titles_1):
    axes1[0, c_idx].set_title(title, fontsize=12, fontweight='bold', pad=8)

plt.suptitle("GAN Architectural & Domain Stress Benchmark: Snapdragon 8 Elite Hexagon NPU", fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
out_fig1 = os.path.join(FIG_DIR, "05_gan_domain_stress_6grid.png")
plt.savefig(out_fig1, dpi=300)
plt.close()
print(f"✅ Saved Grid 1 to: {out_fig1}")

# -------------------------------------------------------------
# GRID 2: Generative Diffusion vs. GANs (5 Rows x 6 Cols)
# -------------------------------------------------------------
print("🎨 Building Grid 2: SD vs GANs Benchmark Grid (5 Rows)...")

sd_samples = sorted(df[df['model'] == 'sd']['sample_idx'].unique())[:5]
fig2, axes2 = plt.subplots(len(sd_samples), 6, figsize=(22, 18), dpi=300)
col_titles_2 = ["Ground Truth", "Corrupted Input", "MIGAN", "AOT-GAN", "LaMa Dilated", "Stable Diffusion 1.5"]

for r_idx, s_idx in enumerate(sd_samples):
    gt_p = os.path.join(IMG_DIR, f"{s_idx}.png")
    mk_p = os.path.join(MASK_DIR, f"{s_idx}.png")
    gt_pil = Image.open(gt_p).convert("RGB").resize((512, 512))
    cr_pil = make_corrupted_input(gt_pil, mk_p)
    
    pct = df[(df['sample_idx'] == s_idx)]['mask_area_pct'].iloc[0]
    
    axes2[r_idx, 0].imshow(gt_pil)
    axes2[r_idx, 0].set_ylabel(f"Sample #{s_idx}\n({pct:.1f}% Mask)", fontsize=10, fontweight='bold')
    axes2[r_idx, 1].imshow(cr_pil)
    
    for c_idx, m in enumerate(['migan', 'aotgan', 'lama', 'sd'], start=2):
        img, caption = get_annotated_cell(df, m, s_idx)
        if img:
            axes2[r_idx, c_idx].imshow(img)
            axes2[r_idx, c_idx].set_xlabel(caption, fontsize=9, fontweight='bold', color='#1d3557')
        else:
            axes2[r_idx, c_idx].set_facecolor("#e5e7eb")

for ax in axes2.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

for c_idx, title in enumerate(col_titles_2):
    axes2[0, c_idx].set_title(title, fontsize=12, fontweight='bold', pad=8)

plt.suptitle("Generative Latent Diffusion vs. Feed-Forward GANs: Snapdragon 8 Elite (HTP v79)", fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
out_fig2 = os.path.join(FIG_DIR, "06_sd_vs_gans_5grid.png")
plt.savefig(out_fig2, dpi=300)
plt.close()
print(f"✅ Saved Grid 2 to: {out_fig2}")
