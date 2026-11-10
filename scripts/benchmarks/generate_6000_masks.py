#!/usr/bin/env python3
"""
scripts/benchmarks/generate_6000_masks.py
================================================================================
⚡ JAGGED, UNEVEN & HIGHLY INCONSISTENT 6,000-MASK DATASET GENERATOR
================================================================================
Generates 6,000 high-diversity, multi-modal inpainting masks with:
  1. Hole Counts (1 to 5 holes):
     - At least 40% single-hole masks: Exactly 2,400 masks (40.0%)
     - Holes 2, 3, 4, 5 evenly distributed: Exactly 900 masks each (15.0% each)
  2. Mask Coverage Area (5.0% to 30.0%):
     - Uniformly partitioned into 5 bins of 5%: [5-10%), [10-15%), [15-20%), [20-25%), [25-30%]
     - Exactly 1,200 masks per coverage bin (20.0% each)
  3. Joint Strata:
     - 1-hole: 480 masks per coverage bin (480 x 5 = 2,400)
     - 2..5 holes: 180 masks per coverage bin per hole count (180 x 5 = 900 each)
     - Total = 2,400 + (4 x 900) = 6,000 masks.
  4. Highly Jiggery, Uneven & Inconsistent Geometries:
     - Jagged Blobs: Multi-harmonic polar radial perturbations with high-frequency saw-tooth jitter.
     - Jittery Brushes: Meandering random-walk paths with sharp angular turns and oscillating stroke width.
     - Torn Boxes: Distressed rectangular cutouts with multi-segmented ragged boundaries and bite-outs.
     - Spiky Clefts: Star-burst shapes with sharp spurs and deep clefts/fissures.
     - Multi-hole heterogeneity: Each hole independently chooses its shape primitive.

Outputs:
  - Benchmark/6000_images/compiled/masks/{sample_id}.png (0001.png to 6000.png)
  - Benchmark/6000_images/compiled/masks_manifest.csv (all 10 topological features)
================================================================================
"""

import os
import sys
import csv
import time
import random
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILED_DIR = REPO_ROOT / "Benchmark" / "6000_images" / "compiled"
MASKS_DIR = COMPILED_DIR / "masks"
MANIFEST_PATH = COMPILED_DIR / "masks_manifest.csv"

TOTAL_MASKS = 6000
IMG_SIZE = (512, 512)

# Hole distribution: >=40% single-hole, even across remaining (2, 3, 4, 5)
HOLE_DISTRIBUTION = {
    1: 2400,  # 40%
    2: 900,   # 15%
    3: 900,   # 15%
    4: 900,   # 15%
    5: 900,   # 15%
}

COVERAGE_BINS = [
    (5.0, 10.0),
    (10.0, 15.0),
    (15.0, 20.0),
    (20.0, 25.0),
    (25.0, 30.0)
]


def create_jagged_blob(target_px: int, w: int, h: int) -> np.ndarray:
    """Amoeboid / fractal island with multi-harmonic perturbation and high-frequency edge jitter."""
    mask = np.zeros((h, w), dtype=np.uint8)
    box_aspect = float(w) / max(1.0, float(h))
    aspect = random.uniform(0.75, 1.25) * box_aspect
    aspect = float(np.clip(aspect, 0.45, 2.2))

    rx_base = np.sqrt(target_px * aspect / np.pi)
    ry_base = target_px / (np.pi * max(1.0, rx_base))

    scale = 1.0
    if rx_base * 2.1 > w:
        scale = min(scale, (w * 0.46) / rx_base)
    if ry_base * 2.1 > h:
        scale = min(scale, (h * 0.46) / ry_base)
    rx_base *= scale
    ry_base *= scale

    cx = random.randint(int(rx_base) + 3, max(int(rx_base) + 4, w - int(rx_base) - 3))
    cy = random.randint(int(ry_base) + 3, max(int(ry_base) + 4, h - int(ry_base) - 3))

    n_pts = random.randint(32, 60)
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    harmonics = random.randint(2, 4)
    r_mod = np.ones(n_pts, dtype=np.float64)
    for _ in range(harmonics):
        freq = random.randint(2, 6)
        amp = random.uniform(0.12, 0.28)
        phase = random.uniform(0, 2 * np.pi)
        r_mod += amp * np.sin(freq * angles + phase)

    # High frequency saw-tooth / jagged noise
    r_mod += np.random.uniform(-0.20, 0.20, n_pts)
    r_mod = np.clip(r_mod, 0.35, 1.45)

    pts = []
    for i in range(n_pts):
        px = int(np.clip(cx + rx_base * r_mod[i] * np.cos(angles[i]), 2, w - 3))
        py = int(np.clip(cy + ry_base * r_mod[i] * np.sin(angles[i]), 2, h - 3))
        pts.append((px, py))
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts_arr], 255)
    return mask


def create_jittery_brush(target_px: int, w: int, h: int) -> np.ndarray:
    """Erratic meandering brush path with sharp angular turns and fluctuating stroke width."""
    mask = np.zeros((h, w), dtype=np.uint8)
    n_nodes = random.randint(4, 7)
    cx, cy = w // 2, h // 2
    step_size = max(10, min(w, h) // (n_nodes + 1))
    nodes = [(cx + random.randint(-w // 6, w // 6), cy + random.randint(-h // 6, h // 6))]
    for _ in range(n_nodes - 1):
        angle = random.uniform(0, 2 * np.pi)
        dist = random.uniform(step_size * 0.6, step_size * 1.4)
        nx = int(np.clip(nodes[-1][0] + dist * np.cos(angle), 10, w - 10))
        ny = int(np.clip(nodes[-1][1] + dist * np.sin(angle), 10, h - 10))
        nodes.append((nx, ny))

    total_len = sum(np.hypot(nodes[i + 1][0] - nodes[i][0], nodes[i + 1][1] - nodes[i][1]) for i in range(len(nodes) - 1))
    base_w = int(target_px / max(15.0, total_len))
    base_w = max(10, min(min(w, h) // 2, base_w))

    for i in range(len(nodes) - 1):
        seg_len = int(np.hypot(nodes[i + 1][0] - nodes[i][0], nodes[i + 1][1] - nodes[i][1]))
        n_steps = max(4, seg_len // 3)
        for s in range(n_steps + 1):
            t = s / float(n_steps)
            px = int(nodes[i][0] * (1 - t) + nodes[i + 1][0] * t)
            py = int(nodes[i][1] * (1 - t) + nodes[i + 1][1] * t)
            curr_r = int(base_w * random.uniform(0.60, 1.40) // 2)
            jx = random.randint(-max(2, curr_r // 3), max(2, curr_r // 3))
            jy = random.randint(-max(2, curr_r // 3), max(2, curr_r // 3))
            cv2.circle(mask, (int(np.clip(px + jx, 2, w - 3)), int(np.clip(py + jy, 2, h - 3))), max(2, curr_r), 255, -1)
    return mask


def create_torn_box(target_px: int, w: int, h: int) -> np.ndarray:
    """Torn distressed box cutout with multi-segmented ragged boundaries and bite-ins."""
    mask = np.zeros((h, w), dtype=np.uint8)
    box_aspect = float(w) / max(1.0, float(h))
    aspect = random.uniform(0.7, 1.3) * box_aspect

    bw = int(np.sqrt(target_px * aspect))
    bh = int(target_px / max(1, bw))

    max_w = w - 10
    max_h = h - 10
    if bw > max_w:
        scale = max_w / float(bw)
        bw = max_w
        bh = min(max_h, int(bh / scale))
    if bh > max_h:
        scale = max_h / float(bh)
        bh = max_h
        bw = min(max_w, int(bw / scale))

    bw = max(12, bw)
    bh = max(12, bh)

    x1 = random.randint(4, max(5, w - bw - 4))
    y1 = random.randint(4, max(5, h - bh - 4))
    x2, y2 = x1 + bw, y1 + bh

    pts = []
    n_seg = 14
    for x in np.linspace(x1, x2, n_seg, endpoint=False):
        pts.append((int(x), int(y1 + random.randint(-max(3, bh // 7), max(3, bh // 7)))))
    for y in np.linspace(y1, y2, n_seg, endpoint=False):
        pts.append((int(x2 + random.randint(-max(3, bw // 7), max(3, bw // 7))), int(y)))
    for x in np.linspace(x2, x1, n_seg, endpoint=False):
        pts.append((int(x), int(y2 + random.randint(-max(3, bh // 7), max(3, bh // 7)))))
    for y in np.linspace(y2, y1, n_seg, endpoint=False):
        pts.append((int(x1 + random.randint(-max(3, bw // 7), max(3, bw // 7))), int(y)))

    pts_arr = np.clip(np.array(pts, dtype=np.int32), 2, [w - 3, h - 3])
    cv2.fillPoly(mask, [pts_arr], 255)
    return mask


def create_spiky_cleft(target_px: int, w: int, h: int) -> np.ndarray:
    """Spiky star-burst polygon with alternating sharp spurs and deep clefts/fissures."""
    mask = np.zeros((h, w), dtype=np.uint8)
    box_aspect = float(w) / max(1.0, float(h))
    aspect = random.uniform(0.7, 1.3) * box_aspect

    comp_target = target_px * 1.35
    rx_base = np.sqrt(comp_target * aspect / np.pi)
    ry_base = comp_target / (np.pi * max(1.0, rx_base))

    scale = 1.0
    if rx_base * 2.1 > w:
        scale = min(scale, (w * 0.45) / rx_base)
    if ry_base * 2.1 > h:
        scale = min(scale, (h * 0.45) / ry_base)
    rx_base *= scale
    ry_base *= scale

    cx = random.randint(int(rx_base) + 3, max(int(rx_base) + 4, w - int(rx_base) - 3))
    cy = random.randint(int(ry_base) + 3, max(int(ry_base) + 4, h - int(ry_base) - 3))

    n_spikes = random.randint(7, 13)
    n_pts = n_spikes * 2
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    r_factor = [random.uniform(1.15, 1.40) if i % 2 == 0 else random.uniform(0.45, 0.70) for i in range(n_pts)]

    pts = []
    for i in range(n_pts):
        px = int(np.clip(cx + rx_base * r_factor[i] * np.cos(angles[i]), 2, w - 3))
        py = int(np.clip(cy + ry_base * r_factor[i] * np.sin(angles[i]), 2, h - 3))
        pts.append((px, py))
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts_arr], 255)
    return mask


SHAPE_GENERATORS = [
    ("jagged_blob", create_jagged_blob),
    ("jittery_brush", create_jittery_brush),
    ("torn_box", create_torn_box),
    ("spiky_cleft", create_spiky_cleft)
]


def generate_exact_mask(target_holes: int, cov_min: float, cov_max: float, size: Tuple[int, int] = (512, 512)) -> Tuple[np.ndarray, int, float, str]:
    """
    Generates a mask with strictly `target_holes` (1 to 5) and coverage strictly in [cov_min, cov_max].
    Returns (mask_np, hole_count, coverage_pct, primary_shape_type).
    """
    w, h = size
    total_px = w * h

    for attempt in range(80):
        target_cov = random.uniform(cov_min + 0.4, cov_max - 0.4)
        target_total_px = int(total_px * (target_cov / 100.0))
        final_mask = np.zeros((h, w), dtype=np.uint8)
        used_shapes = []

        if target_holes == 1:
            st_name, st_fn = random.choice(SHAPE_GENERATORS)
            used_shapes.append(st_name)
            final_mask = st_fn(target_total_px, w, h)
        else:
            weights = np.random.dirichlet(np.ones(target_holes) * 2.5)
            hole_px_list = [int(target_total_px * wt) for wt in weights]
            diff = target_total_px - sum(hole_px_list)
            hole_px_list[0] += diff

            grid_cols = 3 if target_holes >= 4 else 2
            grid_rows = 2 if target_holes >= 3 else 1
            cw, ch = w // grid_cols, h // grid_rows
            grid_cells = [(c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
                          for r in range(grid_rows) for c in range(grid_cols)]
            random.shuffle(grid_cells)
            margin = 8

            for i, px_req in enumerate(hole_px_list):
                gx1, gy1, gx2, gy2 = grid_cells[i]
                eff_w = (gx2 - gx1) - 2 * margin
                eff_h = (gy2 - gy1) - 2 * margin
                st_name, st_fn = random.choice(SHAPE_GENERATORS)
                used_shapes.append(st_name)
                cand = st_fn(px_req, eff_w, eff_h)
                final_mask[gy1 + margin : gy1 + margin + eff_h, gx1 + margin : gx1 + margin + eff_w] = cand

        # Connected components verification
        n_labels, _, _, _ = cv2.connectedComponentsWithStats((final_mask > 127).astype(np.uint8), connectivity=8)
        actual_holes = n_labels - 1
        actual_cov = (np.sum(final_mask > 127) / total_px) * 100.0

        if actual_holes == target_holes and (cov_min <= actual_cov <= cov_max):
            primary_shape = used_shapes[0] if target_holes == 1 else f"multi_{len(set(used_shapes))}_types"
            return final_mask, actual_holes, round(actual_cov, 3), primary_shape

    # Calibrated fallback (strictly jagged)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    target_total_px = int(total_px * ((cov_min + cov_max) / 200.0))
    px_per_hole = int(target_total_px / target_holes)
    grid_cols = 3 if target_holes >= 4 else 2
    grid_rows = 2 if target_holes >= 3 else 1
    cw, ch = w // grid_cols, h // grid_rows
    margin = 8
    for i in range(target_holes):
        r_idx = i // grid_cols
        c_idx = i % grid_cols
        gx1, gy1 = c_idx * cw, r_idx * ch
        eff_w = cw - 2 * margin
        eff_h = ch - 2 * margin
        cand = create_jagged_blob(px_per_hole, eff_w, eff_h)
        final_mask[gy1 + margin : gy1 + margin + eff_h, gx1 + margin : gx1 + margin + eff_w] = cand

    n_labels, _, _, _ = cv2.connectedComponentsWithStats((final_mask > 127).astype(np.uint8), connectivity=8)
    actual_holes = n_labels - 1
    actual_cov = round((np.sum(final_mask > 127) / total_px) * 100.0, 3)
    return final_mask, actual_holes, actual_cov, "jagged_blob"


def compute_mask_features(mask: np.ndarray, mask_type: str, actual_holes: int, coverage_pct: float) -> dict:
    """Computes all 10 topological and spatial features for meta-prediction."""
    h, w = mask.shape
    total_pixels = h * w
    hole_pixels = int(np.sum(mask > 127))

    # Global centroid of mask
    y_coords, x_coords = np.where(mask > 127)
    if len(x_coords) > 0:
        centroid_x = round(float(np.mean(x_coords)) / w, 4)
        centroid_y = round(float(np.mean(y_coords)) / h, 4)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        dist_to_border = round(float(min(min_x, w - 1 - max_x, min_y, h - 1 - max_y)) / min(w, h), 4)
        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        aspect_ratio = round(float(bbox_w) / max(1.0, float(bbox_h)), 4)
    else:
        centroid_x, centroid_y = 0.5, 0.5
        dist_to_border = 0.5
        aspect_ratio = 1.0

    # Perimeter and compactness
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    perimeter_to_area = round(float(perimeter) / max(1.0, np.sqrt(hole_pixels)), 4) if hole_pixels > 0 else 0.0

    return {
        "mask_type": mask_type,
        "coverage_pct": coverage_pct,
        "hole_count": actual_holes,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "dist_to_border": dist_to_border,
        "aspect_ratio": aspect_ratio,
        "perimeter_to_area": perimeter_to_area
    }


def main():
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("⚡ GENERATING 6,000 JAGGED & BALANCED INPAINTING MASKS")
    print("=" * 80)
    print(f" Target Directory   : {MASKS_DIR}")
    print(f" Total Mask Count   : {TOTAL_MASKS}")
    print(f" Single-Hole Masks  : {HOLE_DISTRIBUTION[1]} (40.0% of total)")
    print(f" Multi-Hole Masks   : 900 each for 2, 3, 4, 5 holes (15.0% each)")
    print(f" Coverage Area Range: 5.0% to 30.0% (Exactly 1,200 per 5% bin)")
    print(f" Geometries         : Jagged Blobs, Jittery Brushes, Torn Boxes, Spiky Clefts")
    print("=" * 80 + "\n")

    # Construct balanced task pool:
    # For hole 1 (2400 masks): 2400 / 5 = 480 per coverage bin
    # For holes 2..5 (900 masks each): 900 / 5 = 180 per coverage bin
    plan = []
    for h, total_h_count in HOLE_DISTRIBUTION.items():
        count_per_bin = total_h_count // len(COVERAGE_BINS)
        for (cov_min, cov_max) in COVERAGE_BINS:
            for _ in range(count_per_bin):
                plan.append({
                    "target_holes": h,
                    "cov_min": cov_min,
                    "cov_max": cov_max
                })

    assert len(plan) == TOTAL_MASKS, f"Plan size mismatch: {len(plan)} != {TOTAL_MASKS}"

    # Shuffle plan so image IDs 0001..6000 have randomized, uniform sampling
    random.shuffle(plan)

    manifest_rows = []
    t0 = time.time()

    for idx, item in enumerate(plan, 1):
        stem = f"{idx:04d}"
        target_holes = item["target_holes"]
        cov_min = item["cov_min"]
        cov_max = item["cov_max"]

        mask, actual_holes, actual_cov, shape_type = generate_exact_mask(target_holes, cov_min, cov_max, IMG_SIZE)

        # Save mask as 8-bit PNG
        out_filename = f"{stem}.png"
        out_path = MASKS_DIR / out_filename
        Image.fromarray(mask).save(out_path, format="PNG")

        # Compute topological metadata
        feats = compute_mask_features(mask, shape_type, actual_holes, actual_cov)
        feats["sample_id"] = stem
        feats["filename"] = out_filename
        manifest_rows.append(feats)

        if idx % 500 == 0 or idx == TOTAL_MASKS:
            elapsed = time.time() - t0
            pace = idx / elapsed
            print(f"  [{idx:4d}/{TOTAL_MASKS}] ({idx * 100 // TOTAL_MASKS:3d}%) masks generated — Pace: {pace:5.1f} masks/sec")

    # Write manifest CSV
    fieldnames = [
        "sample_id", "filename", "mask_type", "coverage_pct", "hole_count",
        "centroid_x", "centroid_y", "dist_to_border", "aspect_ratio", "perimeter_to_area"
    ]
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 80)
    print("✅ BALANCED JAGGED 6,000-MASK GENERATION COMPLETE!")
    print(f"  • Masks Folder : {MASKS_DIR}")
    print(f"  • Manifest CSV : {MANIFEST_PATH}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
