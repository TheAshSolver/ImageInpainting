#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.expanduser("~/Desktop/college/ImageInpainting")
CSV_PATH = os.path.join(BASE_DIR, "Benchmark/output/all_pairs_detailed_metrics.csv")
OUT_FIG  = os.path.join(BASE_DIR, "Benchmark/output/figures/scout_report_dashboard.png")

if not os.path.exists(CSV_PATH):
    print(f"Error: CSV not found at {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df['domain'] = np.where(df['sample_idx'] <= 100, 'Places (Scenes)', 'Faces (Portraits)')
df_g = df[df['model'].isin(['migan', 'aotgan', 'lama'])].copy()

BG_COLOR   = "#0b0f19"
PANEL_BG   = "#111827"
CARD_BG    = "#1f2937"
TEXT_COLOR = "#f3f4f6"
SUB_TEXT   = "#9ca3af"
GRID_COLOR = "#1f2937"

COLOR_MIGAN = "#00f5d4"
COLOR_AOT   = "#ff007f"
COLOR_LAMA  = "#3a86ff"

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': PANEL_BG,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': SUB_TEXT,
    'ytick.color': SUB_TEXT,
    'font.family': 'sans-serif',
})

fig = plt.figure(figsize=(19, 10.5), dpi=110)
fig.canvas.manager.set_window_title('Snapdragon 8 Elite NPU - Inpainting Architecture Scout Report')

# Top lowered to 0.77 to give 10% vertical buffer for polar labels
gs = fig.add_gridspec(2, 2, top=0.77, bottom=0.07, left=0.06, right=0.96, hspace=0.38, wspace=0.24)

# Header Block with clean vertical hierarchy
fig.text(0.06, 0.950, "QUALCOMM SNAPDRAGON 8 ELITE | HEXAGON HTP v79 NPU", fontsize=10, fontweight='bold', color=COLOR_MIGAN)
fig.text(0.06, 0.915, "Inpainting Architecture Scouting & Hardware Profiling", fontsize=17, fontweight='heavy', color=TEXT_COLOR)
fig.text(0.06, 0.880, "Stratified Evaluation: Places365 (Scenes) vs. Faces across 200 Irregular Masks (5%–35% Coverage)", fontsize=9.5, color=SUB_TEXT)

# PANEL 1: TACTICAL SCOUT RADAR
ax1 = fig.add_subplot(gs[0, 0], polar=True)
ax1.set_facecolor(PANEL_BG)

categories = ['Throughput\n(FPS)', 'Battery\n(1/Joules)', 'Thermal\nHeadroom', 'Scene Texture\n(Places)', 'Facial Coherence\n(Faces)', 'Perceptual\n(1/LPIPS)']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

stats_migan = [100, 95, 92, 74, 72, 76]
stats_aot   = [55,  46, 36, 100, 98, 100]
stats_lama  = [67,  62, 38, 97, 92, 85]

for stats_list, color, label in [
    (stats_migan, COLOR_MIGAN, 'MIGAN (Speed)'),
    (stats_aot,   COLOR_AOT,   'AOT-GAN (Quality)'),
    (stats_lama,  COLOR_LAMA,  'LaMa Dilated (Balanced)')
]:
    vals = stats_list + stats_list[:1]
    ax1.plot(angles, vals, color=color, linewidth=2.2, label=label)
    ax1.fill(angles, vals, color=color, alpha=0.15)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, fontsize=8, fontweight='bold', color=TEXT_COLOR)
ax1.tick_params(pad=10)
ax1.set_ylim(0, 100)
ax1.set_yticks([25, 50, 75, 100])
ax1.set_yticklabels(["25", "50", "75", "100"], color=SUB_TEXT, fontsize=7)
ax1.grid(color=GRID_COLOR, linestyle='--', linewidth=0.7)
ax1.spines['polar'].set_color(GRID_COLOR)

# Title placed neatly above polar tick labels
ax1.set_title("ARCHITECTURAL SCOUT RADAR", fontsize=10.5, fontweight='bold', pad=22, color=TEXT_COLOR)
ax1.legend(loc='upper right', bbox_to_anchor=(1.30, 1.05), frameon=True, facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=7.5)

# PANEL 2: OCCLUSION STRESS SCATTER
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL_BG)

x_dense = np.linspace(5, 36, 100)
for m, color in [('migan', COLOR_MIGAN), ('aotgan', COLOR_AOT), ('lama', COLOR_LAMA)]:
    for dom, marker, style, dom_lbl in [('Places (Scenes)', 'o', '-', 'Places'), ('Faces (Portraits)', '^', '--', 'Faces')]:
        sub = df_g[(df_g['model'] == m) & (df_g['domain'] == dom)]
        ax2.scatter(sub['mask_area_pct'], sub['hole_psnr'], color=color, marker=marker, alpha=0.30, s=20, edgecolors='none')
        slope, intercept, _, _, _ = stats.linregress(sub['mask_area_pct'], sub['hole_psnr'])
        ax2.plot(x_dense, slope * x_dense + intercept, color=color, linestyle=style, linewidth=1.8,
                 label=f"{m.upper()} [{dom_lbl}]: {slope:.2f} dB/%")

ax2.set_title("HOLE PSNR DEGRADATION vs. MASK AREA %", fontsize=10.5, fontweight='bold', pad=10)
ax2.set_xlabel("Mask Area (% of 512×512)", fontsize=9, fontweight='bold')
ax2.set_ylabel("Hole-Only PSNR (dB) ↑", fontsize=9, fontweight='bold')
ax2.set_xlim(4, 38)
ax2.set_ylim(10, 30)
ax2.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.7)
for spine in ax2.spines.values(): spine.set_color(GRID_COLOR)
ax2.legend(loc='upper right', framealpha=0.85, facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=7.5, ncol=2)

# PANEL 3: DOMAIN PENALTY (DUMBBELL / SLOPE CHART)
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(PANEL_BG)

# Calculate exact domain means and deltas
models_to_plot = [
    ('migan', COLOR_MIGAN, 'MIGAN', 2),
    ('lama', COLOR_LAMA, 'LaMa', 1),
    ('aotgan', COLOR_AOT, 'AOT-GAN', 0)
]

ax3.set_xlim(18.5, 22.0)
ax3.set_ylim(-0.6, 2.6)
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(['AOT-GAN', 'LaMa', 'MIGAN'], fontsize=9.5, fontweight='bold')

for m, color, lbl, y_idx in models_to_plot:
    p_mean = df_g[(df_g['model'] == m) & (df_g['domain'] == 'Places (Scenes)')]['hole_psnr'].mean()
    f_mean = df_g[(df_g['model'] == m) & (df_g['domain'] == 'Faces (Portraits)')]['hole_psnr'].mean()
    delta = p_mean - f_mean

    # Connecting track
    ax3.plot([f_mean, p_mean], [y_idx, y_idx], color=color, linewidth=3.5, alpha=0.4, zorder=2)
    
    # Points: Blue for Places, Pink for Faces
    ax3.scatter(p_mean, y_idx, color='#3a86ff', s=120, edgecolors='white', linewidth=1.2, zorder=3, label='Places (Scenes)' if y_idx==0 else "")
    ax3.scatter(f_mean, y_idx, color='#ff007f', s=120, edgecolors='white', linewidth=1.2, zorder=3, label='Faces (Portraits)' if y_idx==0 else "")
    
    # Text Annotations
    ax3.text(p_mean + 0.08, y_idx + 0.12, f"{p_mean:.2f} dB", color='#93c5fd', fontsize=8.5, fontweight='bold')
    ax3.text(f_mean - 0.08, y_idx + 0.12, f"{f_mean:.2f} dB", color='#f472b6', fontsize=8.5, fontweight='bold', ha='right')
    ax3.text((p_mean + f_mean)/2, y_idx - 0.22, f"Drop: -{delta:.2f} dB", color=TEXT_COLOR, fontsize=8, ha='center', style='italic')

ax3.set_title("DOMAIN PENALTY: PLACES vs. FACES REPAIR GAP", fontsize=10.5, fontweight='bold', pad=10)
ax3.set_xlabel("Hole-Only PSNR (dB) [Higher is Better]", fontsize=9, fontweight='bold')
ax3.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.7)
for spine in ax3.spines.values(): spine.set_color(GRID_COLOR)
ax3.legend(loc='lower left', framealpha=0.85, facecolor=PANEL_BG, edgecolor=GRID_COLOR, fontsize=7.5)

# PANEL 4: HARDWARE PROFILER CARDS
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL_BG)
ax4.axis('off')

ax4.text(0.0, 0.98, "HARDWARE SILICON TELEMETRY (QUALCOMM HTP v79)", fontsize=10.5, fontweight='bold', color=TEXT_COLOR, transform=ax4.transAxes)
ax4.text(0.0, 0.90, "Power (W) = (|I_uA|/10⁶)×(V_uV/10⁶)   |   Energy (J) = W × Sec   |   EDP = J × Sec", fontsize=8.2, fontfamily='monospace', color=COLOR_MIGAN, transform=ax4.transAxes)

cards = [
    {"name": "MIGAN", "color": COLOR_MIGAN, "lat": "0.216 s", "fps": "4.6 FPS", "pwr": "2.86 W", "j": "0.62 J", "edp": "0.133 J·s", "dt": "+9.6°C", "ram": "2.42 GB"},
    {"name": "LAMA DILATED", "color": COLOR_LAMA, "lat": "0.321 s", "fps": "3.1 FPS", "pwr": "3.10 W", "j": "0.99 J", "edp": "0.319 J·s", "dt": "+24.6°C", "ram": "3.00 GB"},
    {"name": "AOT-GAN", "color": COLOR_AOT, "lat": "0.389 s", "fps": "2.6 FPS", "pwr": "3.34 W", "j": "1.30 J", "edp": "0.505 J·s", "dt": "+25.4°C", "ram": "2.89 GB"}
]

card_y = 0.60
card_h = 0.22
for c in cards:
    rect = patches.FancyBboxPatch((0.0, card_y), 0.98, card_h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                  facecolor=CARD_BG, edgecolor=c['color'], linewidth=1.2, transform=ax4.transAxes)
    ax4.add_patch(rect)
    
    ax4.text(0.03, card_y + 0.13, c['name'], fontsize=9.5, fontweight='bold', color=c['color'], transform=ax4.transAxes)
    ax4.text(0.95, card_y + 0.13, f"EDP: {c['edp']}", fontsize=9, fontweight='bold', color=TEXT_COLOR, ha='right', transform=ax4.transAxes)
    
    row_text = f"Latency: {c['lat']} ({c['fps']})  |  Power: {c['pwr']}  |  Energy: {c['j']}  |  Thermal Rise: {c['dt']}  |  RAM: {c['ram']}"
    ax4.text(0.03, card_y + 0.04, row_text, fontsize=8, color=SUB_TEXT, transform=ax4.transAxes)
    
    card_y -= 0.27

ax4.text(0.0, 0.01, "* Neural networks compile to static tensors (1×3×512×512). Latency & power are O(1) constant across mask sizes.",
         fontsize=7.2, style='italic', color='#6b7280', transform=ax4.transAxes)

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
plt.savefig(OUT_FIG, dpi=300, facecolor=BG_COLOR)
print(f"✅ Saved clean 300 DPI dashboard to: {OUT_FIG}")
print("🚀 Opening window...")
plt.show()
