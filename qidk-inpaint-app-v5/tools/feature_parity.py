#!/usr/bin/env python3
"""feature_parity.py REF.csv APP.csv --features name1,name2,... [--key image] [--min-r 0.98] [--max-rel 0.03]

REF.csv  : the offline dataset (artifacts/dataset_100_features_6000.csv) - one row per image, one column per feature.
APP.csv  : the same rows recomputed by the app (debug hook qidk.debug.FEATURES writes it on the device; `adb pull` it).
Compares only the named features (the router's feature_names) on the images present in both files and reports, per feature,
Pearson r and the median relative error. Exit 1 if any feature is below --min-r or above --max-rel: in that case the
Kotlin feature does NOT reproduce the training definition and the router must stay disabled (train/serve skew).
"""
import argparse, csv, math, statistics, sys


def load(path, key):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r[key]: r for r in rows}


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return 1.0 if sx == 0 and sy == 0 else (0.0 if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref"); p.add_argument("app")
    p.add_argument("--features", required=True); p.add_argument("--key", default="image")
    p.add_argument("--min-r", type=float, default=0.98); p.add_argument("--max-rel", type=float, default=0.03)
    a = p.parse_args()
    ref, app = load(a.ref, a.key), load(a.app, a.key)
    common = sorted(set(ref) & set(app))
    print(f"{len(common)} images in common (ref {len(ref)}, app {len(app)})")
    if len(common) < 10:
        print("FAIL need at least 10 common images"); return 1
    bad = 0
    for name in a.features.split(","):
        if name not in next(iter(ref.values())) or name not in next(iter(app.values())):
            print(f"FAIL {name}: column missing in one of the files"); bad += 1; continue
        x = [float(ref[k][name]) for k in common]; y = [float(app[k][name]) for k in common]
        r = pearson(x, y)
        scale = max(1e-6, statistics.median(abs(v) for v in x))
        rel = statistics.median(abs(u - v) / max(abs(u), scale * 0.05) for u, v in zip(x, y))
        ok = r >= a.min_r and rel <= a.max_rel
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {name:28s} r={r:.4f}  median rel err={rel:.4f}")
    print("PARITY OK" if bad == 0 else f"{bad} feature(s) differ from the training definition - keep the heuristic router")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
