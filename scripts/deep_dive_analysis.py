import os, glob
import pandas as pd
import numpy as np
import matplotlib
# Use default interactive GUI backend (TkAgg / Qt)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

csv_path = "Benchmark/output/all_pairs_detailed_metrics.csv"
if not os.path.exists(csv_path):
    print("CSV not found!")
    exit(1)

df = pd.read_csv(csv_path)

# Label first 100 as Places, remaining 100 as Faces
df['dataset'] = np.where(df['sample_idx'] <= 100, 'Places (Scenes)', 'Faces (Portraits)')

# Exclude SD for direct GAN domain comparison
df_gans = df[df['model'].isin(['migan', 'aotgan', 'lama'])].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.canvas.manager.set_window_title('Inpainting: Places vs. Faces Domain Comparison')

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
    faces_vals = [df_gans[(df_gans['model']==m) & (df_gans['dataset']=='Faces (Portraits)')][met].mean() for m in models]
    
    rects1 = ax.bar(x - width/2, places_vals, width, label='Places (Scenes)', color='#3a86ff')
    rects2 = ax.bar(x + width/2, faces_vals, width, label='Faces (Portraits)', color='#ff006e')
    
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

plt.suptitle("Domain Performance Comparison: Places365 (Scenes) vs. Faces on Snapdragon 8 Elite NPU", fontsize=14, fontweight='bold')
plt.tight_layout()
print("Opening interactive Matplotlib window...")
plt.show()
EOF