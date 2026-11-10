#!/usr/bin/env python3
"""
scripts/benchmarks/compile_6000_dataset.py
================================================================================
Compiles and standardizes 6,000 balanced natural images into:
  Benchmark/6000_images/compiled/images/ (0001.png ... 6000.png)
Standardized to canonical 512x512 RGB PNG format with metadata manifest.
================================================================================
"""

import os
import sys
import csv
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "Benchmark" / "6000_images" / "natural_images"
DEST_DIR = REPO_ROOT / "Benchmark" / "6000_images" / "compiled"
IMAGES_DIR = DEST_DIR / "images"
MANIFEST_PATH = DEST_DIR / "dataset_manifest.csv"

TARGET_COUNT = 6000
CANONICAL_SIZE = (512, 512)

resample_filter = getattr(Image, "Resampling", Image).BICUBIC

def main():
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect images per class
    categories = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()])
    cat_files = {}
    for cat in categories:
        files = sorted(list((SOURCE_DIR / cat).glob("*.jpg")) + list((SOURCE_DIR / cat).glob("*.png")) + list((SOURCE_DIR / cat).glob("*.jpeg")))
        cat_files[cat] = files
        print(f"  • {cat:12s}: {len(files)} source images available")

    # Allocate target counts per category to hit exactly 6000
    # dog has 702, airplane has 727. Take all, distribute rest equally.
    allocations = {}
    remaining_slots = TARGET_COUNT
    for cat in ["dog", "airplane"]:
        take = min(len(cat_files[cat]), 750)
        allocations[cat] = take
        remaining_slots -= take

    other_cats = [c for c in categories if c not in allocations]
    per_other = remaining_slots // len(other_cats)
    extra = remaining_slots % len(other_cats)

    for idx, cat in enumerate(other_cats):
        take = per_other + (1 if idx < extra else 0)
        allocations[cat] = min(take, len(cat_files[cat]))

    print(f"\nFinal Class Allocation Plan for {TARGET_COUNT} images:")
    for cat, count in allocations.items():
        print(f"  → {cat:12s}: {count} images")

    # Compile and standardize
    sample_id = 1
    manifest_rows = []

    print(f"\nCompiling and resizing to {CANONICAL_SIZE[0]}x{CANONICAL_SIZE[1]} RGB PNGs...")
    for cat in sorted(allocations.keys()):
        count = allocations[cat]
        selected_files = cat_files[cat][:count]
        for img_path in selected_files:
            stem = f"{sample_id:04d}"
            out_filename = f"{stem}.png"
            out_path = IMAGES_DIR / out_filename

            try:
                with Image.open(img_path) as im:
                    orig_w, orig_h = im.size
                    im_rgb = im.convert("RGB")
                    if im_rgb.size != CANONICAL_SIZE:
                        im_rgb = im_rgb.resize(CANONICAL_SIZE, resample_filter)
                    im_rgb.save(out_path, format="PNG")

                manifest_rows.append({
                    "sample_id": stem,
                    "filename": out_filename,
                    "category": cat,
                    "original_filename": img_path.name,
                    "original_width": orig_w,
                    "original_height": orig_h,
                })

                if sample_id % 500 == 0 or sample_id == TARGET_COUNT:
                    print(f"  Processed {sample_id}/{TARGET_COUNT} ({sample_id*100//TARGET_COUNT}%)...")

                sample_id += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Write manifest CSV
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "filename", "category", "original_filename", "original_width", "original_height"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n✅ Dataset Compilation Complete!")
    print(f"  • Total images compiled : {len(manifest_rows)}")
    print(f"  • Destination folder    : {IMAGES_DIR}")
    print(f"  • Manifest file         : {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
