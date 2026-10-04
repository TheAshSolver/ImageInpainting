#!/usr/bin/env python3
"""
Fresh, *measured* on-device benchmark for the QIDK (Snapdragon 8 Elite, HTP v79).

Everything reported is measured by this script - nothing comes from constants:
  * latency  : cold single launch (N=1, x3) and warm 102-image batch (x3) of snpe-net-run per model/runtime,
               steady-state per image = (batch - cold) / 101. Monotonic device clock (/proc/uptime).
  * thermals : hottest CPU / GPU / NSP thermal zone before & after every run, cooldown barrier between configs.
  * quality  : PSNR, hole-PSNR, SSIM (+ LPIPS-VGG with --lpips) of composited output vs ground truth.
               Missing / NaN / wrongly-sized outputs abort the run (no silent fallback to the input image).
  * SD 1.5   : (--sd) N samples spread over mask coverage, per-stage profile + wall-clock per launch.
Energy is NOT reported: the board exposes no usable system-power sensor (battery current stays ~0 under load).

Usage (repo root, device connected, adb on PATH or --adb):
    python scripts/fresh_benchmark/run_fresh_benchmark.py                      # 6 GAN configs, no LPIPS
    python scripts/fresh_benchmark/run_fresh_benchmark.py --lpips --sd         # + LPIPS + Stable Diffusion
    python scripts/fresh_benchmark/run_fresh_benchmark.py --configs migan_npu lama_npu --cached-dlc
Outputs go to Benchmark/output/fresh_benchmark/<timestamp>/ (summary.json, BENCHMARK_RESULTS.md, quality CSVs, raw logs).
Presentation figures: python scripts/fresh_benchmark/make_figures.py --results <that folder> --ds Benchmark/input_102 --out <figures folder>
"""
import argparse, csv, datetime, json, os, re, subprocess, sys, shutil, types
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DS = REPO / "Benchmark" / "input_102"
LAMA = "/data/local/tmp/lama"
SDDIR = "/data/local/tmp/sd_runtime"

CONFIGS = {  # tag: (dlc, mask list suffix, runtime flag, display name, runtime name)
    "migan_npu":  ("migan_htp_v79.dlc", "inverted", "--use_dsp", "MIGAN", "NPU"),
    "lama_npu":   ("lama_dilated.dlc",  "standard", "--use_dsp", "LaMa-Dilated", "NPU"),
    "aotgan_npu": ("aotgan.dlc",        "standard", "--use_dsp", "AOT-GAN", "NPU"),
    "migan_gpu":  ("migan.dlc",         "inverted", "--use_gpu", "MIGAN", "GPU"),
    "lama_gpu":   ("lama_dilated.dlc",  "standard", "--use_gpu", "LaMa-Dilated", "GPU"),
    "aotgan_gpu": ("aotgan.dlc",        "standard", "--use_gpu", "AOT-GAN", "GPU"),
}
CACHED = {"lama_dilated.dlc": "bench_cache_lama_dilated.dlc", "aotgan.dlc": "bench_cache_aotgan.dlc"}
ADB = "adb"


# --------------------------------------------------------------------------- adb helpers
def adb(*args, timeout=3600, check=True):
    p = subprocess.run([ADB, *args], capture_output=True, text=True, errors="replace", timeout=timeout)
    if check and p.returncode != 0:
        raise SystemExit(f"adb {' '.join(args)} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout


def shell(cmd, timeout=3600):
    return adb("shell", cmd, timeout=timeout, check=False)


# --------------------------------------------------------------------------- dataset / tensors
def stems():
    return [f"{i:03d}" for i in range(1, 103)]


def prepare_tensors(dest: Path):
    """NHWC float32 .raw tensors from the PNG triplets (same layout as scripts/prep_102_benchmark.py)."""
    for d in ("raw_image", "raw_mask_standard", "raw_mask_inverted"):
        (dest / d).mkdir(parents=True, exist_ok=True)
    for s in stems():
        img = Image.open(DS / "image" / f"{s}.png").convert("RGB")
        msk = Image.open(DS / "mask" / f"{s}.png").convert("L")
        assert img.size == (512, 512) and msk.size == (512, 512), s
        (np.asarray(img, dtype=np.float32) / 255.0)[None].tofile(dest / "raw_image" / f"{s}.raw")
        m = (np.asarray(msk) >= 128).astype(np.float32)
        m[None, ..., None].tofile(dest / "raw_mask_standard" / f"{s}_mask.raw")
        (1.0 - m)[None, ..., None].tofile(dest / "raw_mask_inverted" / f"{s}_mask.raw")
    for name, mdir in (("standard", "raw_mask_standard"), ("inverted", "raw_mask_inverted")):
        (dest / f"list_{name}.txt").write_text("".join(
            f"image:=bench_in/raw_image/{s}.raw mask:=bench_in/{mdir}/{s}_mask.raw\n" for s in stems()), newline="\n")
        (dest / f"list1_{name}.txt").write_text(
            f"image:=bench_in/raw_image/001.raw mask:=bench_in/{mdir}/001_mask.raw\n", newline="\n")


def push_inputs(tdir: Path):
    shell(f"rm -rf {LAMA}/bench_in; mkdir -p {LAMA}/bench_in")
    for d in ("raw_image", "raw_mask_standard", "raw_mask_inverted"):
        adb("push", str(tdir / d), f"{LAMA}/bench_in/")
    for f in tdir.glob("list*.txt"):
        adb("push", str(f), f"{LAMA}/{f.name}")
    for f in (HERE / "device").glob("*.sh"):
        adb("push", str(f), f"/data/local/tmp/{f.name}")


# --------------------------------------------------------------------------- quality
_lp = None


def get_lpips():
    global _lp
    if _lp is not None:
        return _lp
    import torch
    try:
        import lpips
        _lp = lpips.LPIPS(net="vgg", verbose=False).eval()
    except Exception as e:  # torchvision native ext unusable -> VGG16 from official weights (VGG16_WEIGHTS env or cache)
        import torch.nn as nn
        w = os.environ.get("VGG16_WEIGHTS", str(REPO / "cache" / "vgg16-397923af.pth"))
        if not os.path.exists(w):
            raise SystemExit(f"LPIPS unavailable ({e!r}) and no VGG16 weights at {w}; set VGG16_WEIGHTS or drop --lpips")
        def _vgg16(*a, **k):
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
            layers, c = [], 3
            for v in cfg:
                if v == 'M': layers.append(nn.MaxPool2d(2, 2))
                else: layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]; c = v
            f = nn.Sequential(*layers)
            sd = torch.load(w, map_location="cpu")
            f.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("features.")})
            return types.SimpleNamespace(features=f)
        tv = types.ModuleType("torchvision"); tv.models = types.ModuleType("torchvision.models"); tv.models.vgg16 = _vgg16
        sys.modules["torchvision"], sys.modules["torchvision.models"] = tv, tv.models
        for k in [k for k in sys.modules if k.startswith("lpips")]: del sys.modules[k]
        import lpips
        _lp = lpips.LPIPS(net="vgg", verbose=False).eval()
    return _lp


def score(stem, model_rgb, use_lpips):
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn, structural_similarity as ssim_fn
    img = np.asarray(Image.open(DS / "image" / f"{stem}.png").convert("RGB"), dtype=np.float32)
    gt = np.asarray(Image.open(DS / "ground_truth" / f"{stem}.png").convert("RGB"), dtype=np.float32)
    m = np.asarray(Image.open(DS / "mask" / f"{stem}.png").convert("L")) >= 128
    mk = m[..., None].astype(np.float32)
    comp8 = np.round(np.clip(img * (1 - mk) + model_rgb * mk, 0, 255)).astype(np.uint8)   # final = img*(1-M) + out*M
    g, p = gt / 255.0, comp8.astype(np.float32) / 255.0
    mse_h = float(np.mean((g - p)[m] ** 2)) if m.any() else 0.0
    row = dict(sample=stem, mask_pct=round(float(m.mean() * 100), 2),
               psnr_full=round(float(psnr_fn(g, p, data_range=1.0)), 4),
               psnr_hole=round(float(10 * np.log10(1.0 / mse_h)) if mse_h > 0 else 50.0, 4),
               ssim=round(float(ssim_fn(g, p, data_range=1.0, channel_axis=2)), 4))
    if use_lpips:
        import torch
        t = lambda x: torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)[None] * 2 - 1
        with torch.no_grad():
            row["lpips_vgg"] = round(float(get_lpips()(t(gt), t(comp8)).item()), 4)
    return row


def load_raw_output(res_dir: Path):
    f = next((res_dir / n for n in ("output_0.raw", "painted_image.raw") if (res_dir / n).exists()), None)
    if f is None:
        raise SystemExit(f"MISSING model output in {res_dir}")
    a = np.fromfile(f, dtype=np.float32)
    if a.size != 512 * 512 * 3 or not np.isfinite(a).all():
        raise SystemExit(f"BAD output {f} (size={a.size}, finite={bool(np.isfinite(a).all())})")
    if a.max() <= 1.05:
        a = a * 255.0 if a.min() >= -0.1 else (a + 1.0) * 127.5
    return np.clip(a, 0, 255).reshape(512, 512, 3)


def evaluate_gan(out_dir: Path, csv_path: Path, use_lpips):
    rows = [score(s, load_raw_output(out_dir / f"Result_{i}"), use_lpips) for i, s in enumerate(stems())]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


# --------------------------------------------------------------------------- device runs
RES_RE = re.compile(r"RESULT (\S+) (cold|batch102) rep=(\d) rc=(\d+) ms=(-?\d+) results=(\d+) start_mC cpu=(\d+) gpu=(\d+) nsp=(\d+) end_mC cpu=(\d+) gpu=(\d+) nsp=(\d+)")


def run_config(tag, dlc, ls, rt, out: Path, keep_raw, use_lpips, cool_mc, cool_timeout):
    print(f"\n=== {tag}", flush=True)
    # the runtime flag(s) must be ONE shell argument (e.g. '--use_dsp --enable_init_cache')
    log = shell(f"sh /data/local/tmp/bench_cfg.sh {tag} {dlc} {ls} '{rt}' {cool_mc} {cool_timeout}", timeout=3600)
    lines = [l for l in log.splitlines() if re.match(r"(COOLDOWN|RESULT|DONE)", l)]
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / f"{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    runs = {"cold": [], "batch102": []}
    for m in RES_RE.finditer("\n".join(lines)):
        t = [int(x) / 1000 for x in m.groups()[6:]]
        runs[m.group(2)].append(dict(rep=int(m.group(3)), rc=int(m.group(4)), ms=int(m.group(5)), n=int(m.group(6)), t_start=t[:3], t_end=t[3:]))
    if len(runs["cold"]) != 3 or len(runs["batch102"]) != 3:
        raise SystemExit(f"{tag}: incomplete timing results")
    if any(r["rc"] != 0 for r in runs["cold"] + runs["batch102"]) or any(r["n"] != 102 for r in runs["batch102"]):
        raise SystemExit(f"{tag}: snpe-net-run failed or produced < 102 results")
    raw = out / "raw_outputs"; raw.mkdir(exist_ok=True)
    adb("pull", f"{LAMA}/bench_out_{tag}_r1", str(raw) + os.sep)
    shell(f"cd {LAMA} && rm -rf bench_out_{tag}_r1 bench_out_{tag}_r2 bench_out_{tag}_r3 bench_out_{tag}_cold bench_out_{tag}_warm")
    q = evaluate_gan(raw / f"bench_out_{tag}_r1", out / f"quality_{tag}.csv", use_lpips)
    if not keep_raw:
        shutil.rmtree(raw / f"bench_out_{tag}_r1", ignore_errors=True)
    return runs, q


def summarize(runs, q):
    cold = [r["ms"] for r in runs["cold"]]; batch = [r["ms"] for r in runs["batch102"]]
    cold_m = float(np.mean(cold)); per = [(b - cold_m) / 101.0 for b in batch]
    e = dict(cold_ms_mean=round(cold_m, 1), cold_ms_all=cold, batch102_ms=batch,
             steady_ms_per_img_mean=round(float(np.mean(per)), 1), steady_ms_per_img_all=[round(x, 1) for x in per],
             steady_fps=round(1000 / float(np.mean(per)), 2), init_ms_est=round(cold_m - float(np.mean(per))),
             throttle_slowdown_pct=round((batch[-1] / batch[0] - 1) * 100, 1),
             temps_c_cpu_gpu_nsp_start=runs["batch102"][0]["t_start"], temps_c_cpu_gpu_nsp_end=runs["batch102"][-1]["t_end"], n_quality=len(q))
    for k in ("psnr_full", "psnr_hole", "ssim", "lpips_vgg"):
        if k in q[0]:
            v = np.array([r[k] for r in q]); e[k] = round(float(v.mean()), 4); e[k + "_std"] = round(float(v.std()), 4)
    tiers = {"light(<15%)": (0, 15), "medium(15-30%)": (15, 30), "heavy(>30%)": (30, 101)}
    e["by_mask_tier"] = {}
    for n, (lo, hi) in tiers.items():
        sel = [r for r in q if lo <= r["mask_pct"] < hi]
        if sel:
            d = dict(n=len(sel), psnr_hole=round(float(np.mean([r["psnr_hole"] for r in sel])), 2), ssim=round(float(np.mean([r["ssim"] for r in sel])), 4))
            if "lpips_vgg" in sel[0]: d["lpips_vgg"] = round(float(np.mean([r["lpips_vgg"] for r in sel])), 4)
            e["by_mask_tier"][n] = d
    return e


# --------------------------------------------------------------------------- Stable Diffusion
SDR = re.compile(r"SDRESULT (\d+) rc=(\d+) wall_ms=(\d+) cooled_s=(\d+) start_mC cpu=(\d+) gpu=(\d+) nsp=(\d+) end_mC cpu=(\d+) gpu=(\d+) nsp=(\d+)")


def run_sd(out: Path, tdir: Path, gan_quality, use_lpips, n=10):
    print("\n=== Stable Diffusion 1.5 (12-step DPM-Solver++)", flush=True)
    ref = gan_quality.get("migan_npu") or next(iter(gan_quality.values()))
    order = sorted(ref, key=lambda r: r["mask_pct"])
    pick = [order[i]["sample"] for i in np.linspace(0, len(order) - 1, n).round().astype(int)]
    (out / "sd").mkdir(exist_ok=True)
    per = {}
    for s in pick:
        adb("push", str(tdir / "raw_image" / f"{s}.raw"), f"{SDDIR}/image.raw")
        adb("push", str(tdir / "raw_mask_standard" / f"{s}_mask.raw"), f"{SDDIR}/mask.raw")
        o = shell(f"sh /data/local/tmp/sd_one.sh {s}", timeout=900)
        (out / "logs" / f"sd_{s}.txt").write_text(o)
        m = SDR.search(o)
        if not m or int(m.group(2)) != 0:
            raise SystemExit(f"SD run failed for {s}:\n{o}")
        d = dict(wall_ms=int(m.group(3)), t_start=[int(m.group(i)) / 1000 for i in (5, 6, 7)], t_end=[int(m.group(i)) / 1000 for i in (8, 9, 10)], prof={})
        for pm in re.finditer(r"SDPROF \d+ \[Profile\] (.+?): (\d+) ms", o): d["prof"][pm.group(1)] = int(pm.group(2))
        em = re.search(r"End-to-End Pipeline Latency: (\d+) ms", o); d["e2e_ms"] = int(em.group(1)) if em else None
        adb("pull", f"{SDDIR}/sd_out_{s}.png", str(out / "sd" / f"{s}.png"))
        im = Image.open(out / "sd" / f"{s}.png").convert("RGB").resize((512, 512))
        d["quality"] = score(s, np.asarray(im, dtype=np.float32), use_lpips)
        per[s] = d
        print(f"  {s}: wall {d['wall_ms']} ms, pipeline {d['e2e_ms']} ms", flush=True)
    shell(f"cd {SDDIR} && rm -f sd_out_*.png sd_stdout_*.txt")
    keys = ("psnr_full", "psnr_hole", "ssim") + (("lpips_vgg",) if use_lpips else ())
    comp = {"sd15_12step": {k: round(float(np.mean([v["quality"][k] for v in per.values()])), 4) for k in keys}}
    for tag, rows in gan_quality.items():
        if tag.endswith("_npu"):
            sel = [r for r in rows if r["sample"] in per]
            comp[tag] = {k: round(float(np.mean([r[k] for r in sel])), 4) for k in keys if k in sel[0]}
    prof = {k: round(float(np.mean([v["prof"][k] for v in per.values()]))) for k in next(iter(per.values()))["prof"]}
    return dict(samples=per, same_subset_comparison=comp, mean_stage_ms=prof,
                wall_ms_mean=round(float(np.mean([v["wall_ms"] for v in per.values()]))),
                pipeline_ms_mean=round(float(np.mean([v["e2e_ms"] for v in per.values()]))))


# --------------------------------------------------------------------------- report
def write_report(path: Path, summary):
    L = ["# Fresh on-device benchmark (measured)", "",
         f"Device: {summary['device']}  |  date: {summary['date']}  |  samples: 102 (Benchmark/input_102)", "",
         "| Model | Runtime | Per image (steady) | img/s | Cold launch | PSNR | Hole-PSNR | SSIM | LPIPS | batch slowdown rep1->3 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, (dlc, ls, rt, name, runtime) in CONFIGS.items():
        e = summary["configs"].get(tag)
        if not e: continue
        L.append(f"| {name}{' (cached DLC)' if e.get('cached_dlc') else ''} | {runtime} | {e['steady_ms_per_img_mean']} ms | {e['steady_fps']} | {e['cold_ms_mean']/1000:.2f} s | "
                 f"{e['psnr_full']:.2f} | {e['psnr_hole']:.2f} | {e['ssim']:.4f} | {('%.4f' % e['lpips_vgg']) if 'lpips_vgg' in e else '-'} | {e['throttle_slowdown_pct']} % |")
    L += ["", "Steady per image = (102-batch wall - mean cold launch)/101. Energy is not reported (no usable system power sensor on the board).", ""]
    for tag, e in summary["configs"].items():
        L.append(f"* `{tag}` by mask size: " + "; ".join(f"{k} n={v['n']} holePSNR={v['psnr_hole']} SSIM={v['ssim']}" + (f" LPIPS={v['lpips_vgg']}" if 'lpips_vgg' in v else "") for k, v in e["by_mask_tier"].items()))
    if "sd" in summary:
        s = summary["sd"]
        L += ["", "## Stable Diffusion 1.5 (12-step)", "",
              f"Pipeline-reported end-to-end: {s['pipeline_ms_mean']} ms; wall-clock per launch: {s['wall_ms_mean']} ms (model loading is outside the pipeline clock).",
              "Mean stage ms: " + ", ".join(f"{k} {v}" for k, v in s["mean_stage_ms"].items()), "",
              "Same-subset comparison:", "", "| Model | " + " | ".join(next(iter(s["same_subset_comparison"].values())).keys()) + " |", "|---|" + "---:|" * len(next(iter(s["same_subset_comparison"].values())))]
        for m, d in s["same_subset_comparison"].items():
            L.append(f"| {m} | " + " | ".join(str(v) for v in d.values()) + " |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    global ADB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adb", default=shutil.which("adb") or "adb")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--lpips", action="store_true", help="also compute LPIPS-VGG (slow on CPU: ~8 min/config)")
    ap.add_argument("--sd", action="store_true", help="also benchmark Stable Diffusion on --sd-samples samples")
    ap.add_argument("--sd-samples", type=int, default=10, help="number of SD samples (spread over mask coverage)")
    ap.add_argument("--cached-dlc", action="store_true", help="use bench_cache_*.dlc for LaMa/AOT-GAN (run make_init_cache.sh first)")
    ap.add_argument("--keep-raw", action="store_true", help="keep pulled raw outputs (~330 MB per config; needed for make_figures per-sample sheets)")
    ap.add_argument("--cool-mc", type=int, default=42000, help="cooldown target on hottest CPU/GPU/NSP zone, milli-degC")
    ap.add_argument("--cool-timeout", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ADB = a.adb

    dev = [l for l in adb("devices").splitlines()[1:] if l.strip().endswith("device")]
    if not dev:
        raise SystemExit("no authorized adb device")
    out = Path(a.out) if a.out else REPO / "Benchmark" / "output" / "fresh_benchmark" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    tdir = out / "tensors"
    print("preparing tensors ...", flush=True); prepare_tensors(tdir)
    print("pushing inputs ...", flush=True); push_inputs(tdir)

    summary = dict(device=dev[0].split()[0], date=datetime.date.today().isoformat(), configs={})
    quality = {}
    for tag in a.configs:
        dlc, ls, rt, name, runtime = CONFIGS[tag]
        cached = a.cached_dlc and dlc in CACHED and runtime == "NPU"
        if cached:
            rt_flag = rt + " --enable_init_cache"; dlc_use = CACHED[dlc]
            if "No such file" in shell(f"ls {LAMA}/{dlc_use} 2>&1"):
                raise SystemExit(f"{dlc_use} missing on device - run make_init_cache.sh first")
        else:
            rt_flag, dlc_use = rt, dlc
        runs, q = run_config(tag, dlc_use, ls, rt_flag, out, a.keep_raw, a.lpips, a.cool_mc, a.cool_timeout)
        quality[tag] = q
        summary["configs"][tag] = dict(summarize(runs, q), cached_dlc=cached)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if a.sd:
        summary["sd"] = run_sd(out, tdir, quality, a.lpips, n=a.sd_samples)
    shutil.rmtree(tdir, ignore_errors=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(out / "BENCHMARK_RESULTS.md", summary)
    print(f"\nDone. Results: {out}\n{(out / 'BENCHMARK_RESULTS.md').read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
