#!/usr/bin/env python3
"""
Domain Performance Comparison: Places365 (Scenes) vs Objects & Faces on Snapdragon 8 Elite NPU.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = "Benchmark/output/all_pairs_detailed_metrics.csv"
if not os.path.exists(csv_path):
    print(f"CSV not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# Label first 100 as Places, remaining 100 as Objects / Faces
df['dataset'] = np.where(df['sample_idx'] <= 100, 'Places (Scenes)', 'Objects & Textures')

# Exclude SD for direct GAN domain comparison
df_gans = df[df['model'].isin(['migan', 'aotgan', 'lama'])].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

metrics = [
    ('hole_psnr', 'Hole-Only PSNR (dB) ↑', 'Higher is Better'),
    ('ssim', 'Structural Similarity (SSIM) ↑', 'Higher is Better'),
    ('lpips', 'Perceptual Error (LPIPS VGG) ↓', 'Lower is Better')
]

models = ['migan', 'aotgan', 'lama']
labels = ['MIGAN', 'AOT-GAN', 'LaMa']
width = 0.35
x = np.arange(len(models))

for idx, (met, title, subtitle) in enumerate(metrics):
    ax = axes[idx]
    
    places_vals = [df_gans[(df_gans['model']==m) & (df_gans['dataset']=='Places (Scenes)')][met].mean() for m in models]
    faces_vals = [df_gans[(df_gans['model']==m) & (df_gans['dataset']=='Objects & Textures')][met].mean() for m in models]
    
    rects1 = ax.bar(x - width/2, places_vals, width, label='Places (Scenes)', color='#3a86ff')
    rects2 = ax.bar(x + width/2, faces_vals, width, label='Objects & Textures', color='#ff006e')
    
    ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Value labels on top of bars
    for rect in rects1 + rects2:
        h = rect.get_height()
        fmt = f"{h:.2f}" if met != 'ssim' else f"{h:.3f}"
        ax.annotate(fmt, xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    if idx == 0:
        ax.legend(loc='lower right')

plt.suptitle("Domain Performance Comparison: Places365 (Scenes) vs. Objects & Textures (Snapdragon 8 Elite NPU)", fontsize=14, fontweight='bold')
plt.tight_layout()

out_path = "Benchmark/output/figures/07_domain_performance_comparison.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300)
plt.close()
print(f"✅ Saved domain performance comparison to: {out_path}")