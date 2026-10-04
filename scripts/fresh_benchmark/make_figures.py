#!/usr/bin/env python3
"""
Presentation figures for the measured QIDK benchmark (publication style, 300-dpi-ready).

Inputs (all produced by run_fresh_benchmark.py / the 2026-10-04 measurement session):
  --results  folder with summary.json and per_sample_csv/quality_*_lpips.csv, quality_sd_vs_gan_subset_lpips.csv
  --ds       Benchmark/input_102 (image/, mask/, ground_truth/)
  --raw      folder holding bench_out_{migan,lama,aotgan}_npu_r1/Result_N/*.raw  (for the per-sample sheets)
  --sd       folder with SD output PNGs named <stem>.png
  --out      figure output folder
Only measured quantities are plotted. There are NO energy/power panels: the board has no usable system-power sensor.
"""
import argparse, csv, json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True); ap.add_argument("--ds", required=True)
ap.add_argument("--raw", default=None); ap.add_argument("--sd", default=None)
ap.add_argument("--out", required=True); ap.add_argument("--dpi", type=int, default=200)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
S = json.load(open(os.path.join(A.results, "summary.json"), encoding="utf-8"))
CSV = os.path.join(A.results, "per_sample_csv")

C = {"migan": "#4361EE", "lama": "#2A9D8F", "aotgan": "#F4A261", "sd": "#E63946"}
NAME = {"migan": "MIGAN", "lama": "LaMa-Dilated", "aotgan": "AOT-GAN", "sd": "SD 1.5 (12-step)"}
BG, PANEL, EDGE, TXT = "white", "#F8F9FC", "#C9D1E0", "#111827"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": PANEL, "axes.edgecolor": EDGE, "figure.facecolor": BG,
                     "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.grid": True, "grid.alpha": .25, "grid.linestyle": "--"})


def rows(path):
    return [{k: (float(v) if k != "sample" and k != "model" else v) for k, v in r.items()} for r in csv.DictReader(open(path, encoding="utf-8"))]

def find_csv(tag):
    """per-sample quality CSV; works for the 2026-10-04 archive layout and for run_fresh_benchmark.py output (needs --lpips)."""
    for p in (f"per_sample_csv/quality_{tag}_lpips.csv", f"quality_{tag}.csv", f"per_sample_csv/quality_{tag}.csv"):
        fp = os.path.join(A.results, p)
        if os.path.exists(fp):
            r = rows(fp)
            if "lpips_vgg" in r[0]: return r
    raise SystemExit(f"no per-sample CSV with LPIPS for {tag} under {A.results} (run run_fresh_benchmark.py with --lpips)")

Q = {t: find_csv(t) for t in ("migan_npu", "lama_npu", "aotgan_npu", "migan_gpu", "lama_gpu", "aotgan_gpu")}
_sdcsv = os.path.join(CSV, "quality_sd_vs_gan_subset_lpips.csv")
if os.path.exists(_sdcsv):
    SDQ = [r for r in rows(_sdcsv) if r["model"] == "sd15_inpaint_12step"]
elif "sd" in S:
    SDQ = [dict(sample=k, **v["quality"]) for k, v in S["sd"]["samples"].items()]
else:
    raise SystemExit("no Stable Diffusion results found (run run_fresh_benchmark.py with --sd --lpips)")
K = ("migan", "lama", "aotgan")
col = lambda rs, k: np.array([r[k] for r in rs])
save = lambda fig, name: (fig.savefig(os.path.join(A.out, name), dpi=A.dpi, bbox_inches="tight", facecolor=fig.get_facecolor()), plt.close(fig), print("wrote", name))
FOOT = "Measured on QIDK (Snapdragon 8 Elite, HTP v79) | 102-sample object-removal set | energy not reported (no usable power sensor)"


# ------------------------------------------------------------------ 01 statistical distributions
def fig_box():
    fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))
    spec = [("psnr_full", "Global PSNR (dB) ↑"), ("psnr_hole", "Hole-Only PSNR (dB) ↑"), ("ssim", "Structural Similarity (SSIM) ↑"), ("lpips_vgg", "LPIPS-VGG ↓")]
    cats = [(k, Q[f"{k}_npu"], C[k], f"{NAME[k]}\n(NPU, n=102)") for k in K] + [("sd", SDQ, C["sd"], "SD 1.5 12-step\n(NPU, n=10)")]
    rng = np.random.default_rng(0)
    for ax, (m, title) in zip(axs.ravel(), spec):
        data = [col(r, m) for _, r, _, _ in cats]
        bp = ax.boxplot(data, widths=.5, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=2))
        for i, (patch, (k, r, c, lab)) in enumerate(zip(bp["boxes"], cats)):
            patch.set(facecolor=c, alpha=.75, edgecolor="black")
            ax.scatter(rng.normal(i + 1, .06, len(data[i])), data[i], s=14, color=c, alpha=.45, zorder=3, edgecolor="none")
            ax.scatter([i + 1], [data[i].mean()], s=55, color="white", edgecolor="black", zorder=5)
            top = ax.get_ylim()[1]
            ax.annotate(f"μ: {data[i].mean():.3f}\nmed: {np.median(data[i]):.3f}" if m in ("ssim", "lpips_vgg") else f"μ: {data[i].mean():.2f}\nmed: {np.median(data[i]):.2f}",
                        (i + 1, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=9,
                        fontweight="bold", color=c, bbox=dict(boxstyle="round,pad=.25", fc="white", ec=c))
        ax.set_xticks(range(1, len(cats) + 1)); ax.set_xticklabels([c[3] for c in cats], fontsize=9, fontweight="bold")
        ax.set_title(title); ax.margins(y=.18)
    fig.suptitle("Statistical Metric Distributions: GAN inpainters (NPU, 102 samples) vs. Stable Diffusion 1.5 (10 samples)", fontsize=15, fontweight="bold", y=.995)
    fig.text(.5, .005, FOOT, ha="center", fontsize=8, color="#555")
    save(fig, "01_metric_distributions_boxplots.png")


# ------------------------------------------------------------------ 02 latency vs quality (Pareto)
def fig_pareto():
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.5))
    rng = np.random.default_rng(1)
    pts = []
    off = {("migan", "npu"): (0, 70), ("lama", "npu"): (0, 100), ("aotgan", "npu"): (0, 130), ("migan", "gpu"): (0, -95), ("lama", "gpu"): (0, -125), ("aotgan", "gpu"): (0, -155)}
    for k in K:
        for rt, mk in (("npu", "o"), ("gpu", "s")):
            e = S["configs"][f"{k}_{rt}"] if "configs" in S else S[f"{k}_{rt}"]
            lat = e["steady_ms_per_img_mean"]; q = Q[f"{k}_{rt}"]
            ps = col(q, "psnr_full")
            a.scatter(lat * np.exp(rng.normal(0, .015, len(ps))), ps, s=12, color=C[k], alpha=.25, marker=mk)
            a.scatter([lat], [ps.mean()], s=150, color=C[k], edgecolor="black", lw=1.8, marker=mk, zorder=5)
            a.annotate(f"{NAME[k]} {rt.upper()}  {lat:.0f} ms", (lat, ps.mean()), xytext=off[(k, rt)], textcoords="offset points", fontsize=8, fontweight="bold", ha="center",
                       arrowprops=dict(arrowstyle="-", color=C[k], lw=1), bbox=dict(boxstyle="round,pad=.2", fc="white", ec=C[k], alpha=.9))
            lp = col(q, "lpips_vgg"); pts.append((lat, lp.mean(), lp.std() / np.sqrt(len(lp)), k, rt, mk))
    sd = S["sd"] if "sd" in S else None
    sdlat = (sd["pipeline_ms_mean"] if sd else 13309)
    a.scatter(sdlat * np.exp(rng.normal(0, .015, len(SDQ))), col(SDQ, "psnr_full"), s=14, color=C["sd"], alpha=.4)
    a.scatter([sdlat], [col(SDQ, "psnr_full").mean()], s=170, color=C["sd"], edgecolor="black", lw=1.8, zorder=5)
    a.annotate(f"SD 1.5 12-step (n=10)\n{sdlat/1000:.1f} s pipeline", (sdlat, col(SDQ, 'psnr_full').mean()), xytext=(-110, 60), textcoords="offset points", fontsize=8, fontweight="bold", color=C["sd"], ha="center",
               arrowprops=dict(arrowstyle="-", color=C["sd"], lw=1), bbox=dict(boxstyle="round,pad=.2", fc="white", ec=C["sd"]))
    a.set_xscale("log"); a.set_xlabel("Steady-state latency per image (ms, log scale) ↓"); a.set_ylabel("Global PSNR (dB) ↑")
    a.set_xticks([50, 100, 200, 500, 1000, 5000, 10000]); a.set_xticklabels(["50", "100", "200", "500", "1000", "5000", "10000"]); a.set_xlim(35, 25000)
    a.set_title("A. Latency vs. Reconstruction Fidelity (per-sample cloud + mean)")
    a.legend(handles=[Patch(color=C[k], label=NAME[k]) for k in K] + [plt.Line2D([], [], marker="o", color="gray", ls="", label="NPU"), plt.Line2D([], [], marker="s", color="gray", ls="", label="GPU")], fontsize=8, loc="lower left")
    # B: LPIPS vs latency with Pareto frontier
    boff = {("migan", "npu"): (0, 22), ("lama", "npu"): (0, -26), ("aotgan", "npu"): (0, 24), ("migan", "gpu"): (0, -28), ("lama", "gpu"): (0, -26), ("aotgan", "gpu"): (0, 24)}
    for lat, m, s, k, rt, mk in pts:
        b.errorbar(lat, m, yerr=s, fmt=mk, color=C[k], mec="black", ms=11, capsize=3, alpha=.9)
        b.annotate(f"{NAME[k]} {rt.upper()}", (lat, m), xytext=boff[(k, rt)], textcoords="offset points", fontsize=8, ha="center")
    fr, best = [], 9
    for lat, m, *_ in sorted(pts):
        if m < best: fr.append((lat, m)); best = m
    b.plot(*zip(*fr), color="#10B981", lw=2, ls="--", label="Pareto frontier (latency vs LPIPS)")
    b.set_xscale("log"); b.set_xlabel("Steady-state latency per image (ms, log scale) ↓"); b.set_ylabel("LPIPS-VGG (mean ± s.e.m.) ↓")
    b.set_xticks([50, 100, 200, 500, 1000]); b.set_xticklabels(["50", "100", "200", "500", "1000"]); b.set_xlim(35, 900)
    b.set_ylim(b.get_ylim()[0] - .004, b.get_ylim()[1] + .004)
    from matplotlib.ticker import NullFormatter
    for axx in (a, b): axx.xaxis.set_minor_formatter(NullFormatter())
    b.set_title("B. Latency vs. Perceptual Quality"); b.legend(fontsize=8)
    fig.suptitle("Qualcomm Snapdragon 8 Elite: Inpainting Pareto Frontier (NPU vs GPU, 102 samples)", fontsize=15, fontweight="bold")
    fig.text(.5, -.02, FOOT, ha="center", fontsize=8, color="#555")
    save(fig, "02_latency_vs_quality_pareto.png")


# ------------------------------------------------------------------ 03 radar
def fig_radar():
    cfg = lambda k: S["configs"][f"{k}_npu"] if "configs" in S else S[f"{k}_npu"]
    ax_names = ["Throughput\n(1 / steady ms)", "Cold start\n(1 / cold launch)", "Edge & Structure\n(SSIM)", "In-hole Detail\n(hole PSNR)",
                "Perceptual Realism\n(1 − LPIPS)", "Light-mask Fidelity\n(hole PSNR <15 %)", "Large-mask Robustness\n(hole PSNR >30 %)"]
    raw = {}
    for k in K:
        e = cfg(k); t = e["by_mask_tier"]
        raw[k] = [1 / e["steady_ms_per_img_mean"], 1 / e["cold_ms_mean"], e["ssim"], e["psnr_hole"], 1 - e["lpips_vgg"],
                  t["light(<15%)"]["psnr_hole"], t["heavy(>30%)"]["psnr_hole"]]
    best = np.max([raw[k] for k in K], axis=0)
    ang = np.linspace(0, 2 * np.pi, len(ax_names), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(10, 9)); ax = fig.add_subplot(111, polar=True)
    for k in K:
        v = (np.array(raw[k]) / best).tolist(); v += v[:1]
        ax.plot(ang, v, color=C[k], lw=2.5, label=NAME[k]); ax.fill(ang, v, color=C[k], alpha=.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(ax_names, fontsize=9, fontweight="bold"); ax.set_ylim(0, 1.05); ax.set_yticks([.25, .5, .75, 1.0]); ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.tick_params(axis="x", pad=18)
    fig.suptitle("Architectural Radar: GAN inpainters on Hexagon HTP v79 (value ÷ best of three)", fontsize=13, fontweight="bold", y=.97)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, frameon=False)
    fig.text(.5, .01, "Cold start uses the uncached DLC (LaMa/AOT-GAN drop to 0.49/0.57 s with the init cache). SD excluded (different subset). " + FOOT, ha="center", fontsize=6.5, color="#555")
    save(fig, "03_architectural_radar.png")


# ------------------------------------------------------------------ 04 telemetry bars
def fig_telemetry():
    g = lambda t: S["configs"][t] if "configs" in S else S[t]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(20, 6.2))
    x = np.arange(3); w = .36
    for j, (rt, hatch) in enumerate((("npu", ""), ("gpu", "//"))):
        vals = [g(f"{k}_{rt}")["steady_ms_per_img_mean"] for k in K]
        bars = a.bar(x + (j - .5) * w, vals, w, color=[C[k] for k in K], hatch=hatch, edgecolor="black", alpha=.9 if rt == "npu" else .55)
        for i, k in enumerate(K):
            reps = g(f"{k}_{rt}")["steady_ms_per_img"] if "steady_ms_per_img" in g(f"{k}_{rt}") else g(f"{k}_{rt}")["steady_ms_per_img_all"]
            a.scatter([x[i] + (j - .5) * w] * len(reps), reps, color="black", s=12, zorder=5)
            a.text(x[i] + (j - .5) * w, vals[i] * 1.03, f"{vals[i]:.0f} ms", ha="center", fontsize=8, fontweight="bold")
    for i, k in enumerate(K):
        a.text(x[i], max(g(f"{k}_gpu")["steady_ms_per_img_mean"], 1) * 1.12, f"NPU {g(f'{k}_gpu')['steady_ms_per_img_mean']/g(f'{k}_npu')['steady_ms_per_img_mean']:.1f}× faster", ha="center", fontsize=9, color="#047857", fontweight="bold")
    a.set_xticks(x); a.set_xticklabels([NAME[k] for k in K], fontweight="bold"); a.set_ylabel("Steady-state ms per image ↓"); a.set_title("A. Throughput: NPU vs GPU\n(dots = 3 repeats)", fontsize=11)
    a.legend(handles=[Patch(fc="gray", ec="black", label="NPU"), Patch(fc="gray", ec="black", hatch="//", alpha=.55, label="GPU")], fontsize=8); a.margins(y=.18)
    # B cold launch
    labels, vals, cols, hat = [], [], [], []
    for k in K:
        labels.append(f"{NAME[k]}\nNPU"); vals.append(g(f"{k}_npu")["cold_ms_mean"] / 1000); cols.append(C[k]); hat.append("")
        if k in ("lama", "aotgan"):
            cached = {"lama": 0.49, "aotgan": 0.57}[k]; labels.append(f"{NAME[k]}\nNPU+cache"); vals.append(cached); cols.append(C[k]); hat.append("xx")
        labels.append(f"{NAME[k]}\nGPU"); vals.append(g(f"{k}_gpu")["cold_ms_mean"] / 1000); cols.append(C[k]); hat.append("//")
    sdw = (S["sd"]["wall_ms_mean"] if "sd" in S else 25172) / 1000
    labels.append("SD 1.5\nNPU"); vals.append(sdw); cols.append(C["sd"]); hat.append("")
    xb = np.arange(len(vals)); bb = b.bar(xb, vals, color=cols, edgecolor="black", hatch=hat, alpha=.85)
    for i, v in enumerate(vals): b.text(i, v * 1.08, f"{v:.2f}s", ha="center", fontsize=7.5, fontweight="bold")
    b.set_yscale("log"); b.set_xticks(xb); b.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=7, rotation=55, ha="right"); b.set_ylabel("Cold single-launch time (s, log) ↓"); b.set_title("B. Cold start: one fresh process per image\n(what the app pays per tap)", fontsize=11)
    # C thermals
    cfgs = [f"{k}_{rt}" for rt in ("npu", "gpu") for k in K]; xc = np.arange(len(cfgs)); w3 = .26
    for j, (zone, zc) in enumerate((("cpu", "#EF4444"), ("gpu", "#F59E0B"), ("nsp", "#3B82F6"))):
        v = [g(t)["temps_c_cpu_gpu_nsp_end"][j] if "temps_c_cpu_gpu_nsp_end" in g(t) else [g(t)["end_temp_c_cpu"], g(t)["end_temp_c_gpu"], g(t)["end_temp_c_nsp"]][j] for t in cfgs]
        c.bar(xc + (j - 1) * w3, v, w3, color=zc, edgecolor="black", alpha=.85, label=f"{zone.upper()} zone (max)")
    for i, t in enumerate(cfgs):
        sl = g(t)["throttle_slowdown_pct"]
        c.text(i, 101, f"{sl:+.1f}%", ha="center", fontsize=8.5, fontweight="bold", color="#B91C1C" if sl > 3 else "#047857")
    c.axhline(85, color="#DC2626", ls="--", lw=1); c.text(-.45, 86, "85 °C", color="#DC2626", fontsize=8, ha="left")
    c.set_ylim(0, 112); c.set_xticks(xc); c.set_xticklabels([f"{NAME[k]}\n{rt.upper()}" for rt in ("npu", "gpu") for k in K], fontsize=7); c.set_ylabel("Compute-zone temp after 3×102 images (°C)")
    c.set_title("C. Thermals & throttling\n(label = batch-time change, rep 1→3)", fontsize=11); c.legend(fontsize=7.5, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.28), frameon=False)
    fig.suptitle("Snapdragon 8 Elite On-Device Telemetry: latency, cold start and thermals (measured, 3 repeats)", fontsize=15, fontweight="bold")
    fig.text(.5, -.16, FOOT, ha="center", fontsize=8, color="#555")
    save(fig, "04_telemetry_latency_coldstart_thermals.png")


# ------------------------------------------------------------------ 05 CDFs
def fig_cdf():
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, m, thr, lab in ((a, "psnr_full", 20.0, "Global PSNR (dB) ↑"), (b, "ssim", .70, "Structural Similarity Index (SSIM) ↑")):
        txt = []
        for k in K + ("sd",):
            d = np.sort(col(SDQ if k == "sd" else Q[f"{k}_npu"], m)); y = np.arange(1, len(d) + 1) / len(d) * 100
            ax.plot(d, y, lw=2.6, color=C[k], ls=":" if k == "sd" else "-", label=NAME[k] + (" (n=10)" if k == "sd" else " (NPU, n=102)"))
            txt.append(f"{NAME[k]}: {(d >= thr).mean()*100:.1f}%")
        ax.axvline(thr, color="#7C3AED", ls=":", lw=2, label=f"Threshold ({thr:g})")
        ax.text(.02, .97, "Pass rate (≥%g):\n" % thr + "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round", fc="white", ec=EDGE))
        ax.set_xlabel(lab); ax.set_ylabel("Cumulative % of samples"); ax.legend(fontsize=8, loc="lower right")
    a.set_title("A. Empirical CDF: Global PSNR"); b.set_title("B. Empirical CDF: SSIM")
    fig.suptitle("Quality Robustness CDF across the 102-sample benchmark (SD on its 10-sample subset)", fontsize=14, fontweight="bold")
    fig.text(.5, -.02, FOOT, ha="center", fontsize=8, color="#555")
    save(fig, "05_quality_robustness_cdf.png")


# ------------------------------------------------------------------ 06 mask-size sensitivity (regression)
def fig_regression():
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, m, lab in ((a, "psnr_hole", "Hole-only PSNR (dB) ↑"), (b, "lpips_vgg", "LPIPS-VGG ↓")):
        for k in K:
            q = Q[f"{k}_npu"]; x, y = col(q, "mask_pct"), col(q, m)
            ax.scatter(x, y, s=16, color=C[k], alpha=.35)
            sl, ic = np.polyfit(x, y, 1); r2 = np.corrcoef(x, y)[0, 1] ** 2
            xs = np.linspace(x.min(), x.max(), 50); ax.plot(xs, sl * xs + ic, color=C[k], lw=2.4, label=f"{NAME[k]}: slope {sl:+.3f}/%  R²={r2:.2f}")
        ax.set_xlabel("Mask coverage (% of image)"); ax.set_ylabel(lab); ax.legend(fontsize=8)
    a.set_title("A. Hole PSNR vs mask area (OLS)"); b.set_title("B. LPIPS vs mask area (OLS)")
    fig.suptitle("Sensitivity to Occlusion Size (NPU, 102 samples)", fontsize=14, fontweight="bold")
    fig.text(.5, -.02, FOOT, ha="center", fontsize=8, color="#555")
    save(fig, "06_mask_size_sensitivity_regression.png")


# ------------------------------------------------------------------ 07 tier bars
def fig_tiers():
    g = lambda t: S["configs"][t] if "configs" in S else S[t]
    tiers = ["light(<15%)", "medium(15-30%)", "heavy(>30%)"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.5)); x = np.arange(3); w = .26
    for ax, m, lab in ((a, "psnr_hole", "Hole-only PSNR (dB) ↑"), (b, "lpips_vgg", "LPIPS-VGG ↓")):
        for j, k in enumerate(K):
            t = g(f"{k}_npu")["by_mask_tier"]; v = [t[n][m] for n in tiers]
            ax.bar(x + (j - 1) * w, v, w, color=C[k], edgecolor="black", label=NAME[k])
            for i, vv in enumerate(v): ax.text(x[i] + (j - 1) * w, vv + (.1 if m == "psnr_hole" else .004), f"{vv:.2f}" if m == "psnr_hole" else f"{vv:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels([f"{n}\n(n={g('migan_npu')['by_mask_tier'][n]['n']})" for n in tiers], fontweight="bold"); ax.set_ylabel(lab); ax.legend(fontsize=8); ax.margins(y=.12)
    a.set_title("A. Hole PSNR by mask-size tier"); b.set_title("B. LPIPS by mask-size tier")
    fig.suptitle("Quality vs. Occlusion Tier (NPU): LaMa holds up best on large masks", fontsize=14, fontweight="bold")
    fig.text(.5, -.02, FOOT, ha="center", fontsize=8, color="#555")
    save(fig, "07_quality_by_mask_tier.png")


# ------------------------------------------------------------------ per-sample dark sheets
def raw_img(tag, stem):
    rd = os.path.join(A.raw, f"bench_out_{tag}_r1", f"Result_{int(stem)-1}")
    f = next(os.path.join(rd, n) for n in ("output_0.raw", "painted_image.raw") if os.path.exists(os.path.join(rd, n)))
    a = np.fromfile(f, dtype=np.float32)
    if a.max() <= 1.05: a = a * 255.0
    return np.clip(a, 0, 255).reshape(512, 512, 3)


def sheets():
    if not (A.raw and A.sd): print("skip per-sample sheets (no --raw/--sd)"); return
    stems = sorted({r["sample"] for r in SDQ}); dark = "#0F1115"
    lat = {k: S["configs"][f"{k}_npu"]["steady_ms_per_img_mean"] if "configs" in S else S[f"{k}_npu"]["steady_ms_per_img_mean"] for k in K}
    sdlat = (S["sd"]["pipeline_ms_mean"] if "sd" in S else 13309) / 1000
    for stem in stems:
        im = np.asarray(Image.open(os.path.join(A.ds, "image", f"{stem}.png")).convert("RGB"))
        gt = np.asarray(Image.open(os.path.join(A.ds, "ground_truth", f"{stem}.png")).convert("RGB"))
        mk = np.asarray(Image.open(os.path.join(A.ds, "mask", f"{stem}.png")).convert("L")) >= 128
        pct = mk.mean() * 100; tier = "LIGHT" if pct < 15 else "MEDIUM" if pct < 30 else "HEAVY"
        ov = im.copy().astype(np.float32); ov[mk] = ov[mk] * .45 + np.array([255, 70, 70]) * .55
        outs = {}
        for k in K:
            o = raw_img(f"{k}_npu", stem); outs[k] = np.where(mk[..., None], o, im).astype(np.uint8)
        outs["sd"] = np.where(mk[..., None], np.asarray(Image.open(os.path.join(A.sd, f"{stem}.png")).convert("RGB").resize((512, 512))), im).astype(np.uint8)
        met = {}
        for k in K: met[k] = next(r for r in Q[f"{k}_npu"] if r["sample"] == stem)
        met["sd"] = next(r for r in SDQ if r["sample"] == stem)
        win_l = min(met, key=lambda k: met[k]["lpips_vgg"]); win_p = max(met, key=lambda k: met[k]["psnr_hole"])
        fig = plt.figure(figsize=(16, 9.6), facecolor=dark); gs = GridSpec(3, 4, figure=fig, height_ratios=[1, .13, 1], hspace=.32, wspace=.05, left=.03, right=.97, top=.86, bottom=.07)
        fig.text(.03, .958, f"SAMPLE {stem}: OBJECT REMOVAL  |  {tier} OCCLUSION ({pct:.1f}% of image)", color="white", fontsize=16, fontweight="bold", va="center")
        fig.text(.03, .925, "Hexagon HTP v79 NPU  |  per-image latency: MIGAN %.0f ms, LaMa %.0f ms, AOT-GAN %.0f ms, SD 1.5 %.1f s (12 steps)" % (lat["migan"], lat["lama"], lat["aotgan"], sdlat), color="#9CA3AF", fontsize=10, va="center")
        def tile(r, c, img, title, edge, cap):
            ax = fig.add_subplot(gs[r, c]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values(): s.set_edgecolor(edge); s.set_linewidth(2.4)
            ax.set_title(title, color=edge, fontsize=11, fontweight="bold", pad=6); ax.set_xlabel(cap, color="#D1D5DB", fontsize=8.5, labelpad=6, fontweight="normal")
        tile(0, 0, ov.astype(np.uint8), "1. Input + object mask (red)", "#9CA3AF", "Object to remove")
        tile(0, 1, np.stack([mk * 255] * 3, -1).astype(np.uint8), "2. Mask", "#9CA3AF", f"Coverage {pct:.1f}%")
        tile(0, 2, gt, "3. Ground truth (object removed)", "#9CA3AF", "Reference scene")
        axc = fig.add_subplot(gs[0, 3]); axc.axis("off"); axc.set_facecolor(dark)
        SHORT = {"migan": "MIGAN", "lama": "LaMa", "aotgan": "AOT-GAN", "sd": "SD 1.5"}
        lines = [f"{'':9s}{'holePSNR':>9s}{'SSIM':>7s}{'LPIPS':>7s}"]
        for k in K + ("sd",):
            lines.append(f"{SHORT[k]:9s}{met[k]['psnr_hole']:9.2f}{met[k]['ssim']:7.3f}{met[k]['lpips_vgg']:7.3f}")
        axc.text(0, 1.0, "PER-SAMPLE METRICS", color="white", fontsize=12, fontweight="bold", va="top", transform=axc.transAxes)
        axc.text(0, .90, "\n".join(lines), color="#E5E7EB", fontsize=10, family="monospace", va="top", transform=axc.transAxes, linespacing=1.8)
        axc.text(0, .42, f"Best hole-PSNR:  {SHORT[win_p]}\nBest LPIPS:      {SHORT[win_l]}", color="#34D399", fontsize=11, fontweight="bold", va="top", transform=axc.transAxes, linespacing=1.7)
        axc.text(0, .12, "(single-sample metrics are noisy;\nsee box plots for distributions)", color="#6B7280", fontsize=8.5, va="top", transform=axc.transAxes)
        axm = fig.add_subplot(gs[1, :]); axm.axis("off")
        axm.text(0, .5, "DOWNSTREAM COMPOSITED INPAINTING     [ out = M·model + (1−M)·input ]", color="#6EE7B7", fontsize=10, fontweight="bold", va="center", transform=axm.transAxes)
        for j, k in enumerate(K + ("sd",)):
            ec = C[k] if k != "sd" else "#F87171"
            tile(2, j, outs[k], f"Inpaint: {NAME[k]}", ec, f"holePSNR {met[k]['psnr_hole']:.2f} dB | SSIM {met[k]['ssim']:.3f} | LPIPS {met[k]['lpips_vgg']:.3f}" + ("  ★ best LPIPS" if k == win_l else ""))
        fig.text(.5, .01, FOOT, color="#6B7280", fontsize=7.5, ha="center")
        fig.savefig(os.path.join(A.out, f"sample_{stem}_qualitative.png"), dpi=A.dpi - 50, facecolor=dark); plt.close(fig); print("wrote", f"sample_{stem}_qualitative.png")


if __name__ == "__main__":
    fig_box(); fig_pareto(); fig_radar(); fig_telemetry(); fig_cdf(); fig_regression(); fig_tiers(); sheets()
