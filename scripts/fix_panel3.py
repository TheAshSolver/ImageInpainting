import re

path = "/home/tarsh/Desktop/college/ImageInpainting/scripts/plot_scout_dashboard.py"
with open(path, "r") as f:
    code = f.read()

# Replace Panel 3 block with a clear Dumbbell / Domain Delta chart
old_panel3_pattern = r"# PANEL 3: DOMAIN PENALTY QUADRANT.*?# PANEL 4:"
new_panel3 = """# PANEL 3: DOMAIN PENALTY (DUMBBELL / SLOPE CHART)
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

# PANEL 4:"""

code = re.sub(old_panel3_pattern, new_panel3, code, flags=re.DOTALL)

with open(path, "w") as f:
    f.write(code)

print("✅ Updated plot_scout_dashboard.py with clean Dumbbell Slope Chart!")
