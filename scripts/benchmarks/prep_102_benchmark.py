#!/usr/bin/env python3
"""
scripts/prep_102_benchmark.py
Standardizes the 102-sample benchmark dataset (Benchmark/dataset_previous)
for Qualcomm Hexagon NPU / SNPE / QNN DLC inference.

Key Tasks:
1. Downscale ideal/ images to 512x512 RGB as official ground truths in Benchmark/input_102/ground_truth/
   using bicubic interpolation.
2. Standardize corrupted images/ into Benchmark/input_102/image/ (512x512 RGB).
3. Standardize masks/ into Benchmark/input_102/mask/ (512x512 uint8 binary, 255 = hole, 0 = keep).
4. Generate .raw binary tensors (NHWC float32 [0.0, 1.0]):
   - raw_image/: (1, 512, 512, 3) float32 (3,145,728 bytes)
   - raw_mask_standard/: (1, 512, 512, 1) float32 (1,048,576 bytes) [1.0 = hole, 0.0 = keep] for LaMa & AOT-GAN
   - raw_mask_inverted/: (1, 512, 512, 1) float32 (1,048,576 bytes) [0.0 = hole, 1.0 = keep] for MIGAN
   - raw_mask/: default copy of standard masks
5. Generate input_list_102.txt formatted for snpe-net-run on device:
   image:=input/raw_image/<stem>.raw mask:=input/raw_mask/<stem>_mask.raw
"""

import os
import sys
import glob
import argparse
import re
import numpy as np
from PIL import Image

TARGET_SIZE = (512, 512)
EXPECTED_IMAGE_RAW_BYTES = 1 * 512 * 512 * 3 * 4  # 3,145,728 bytes (float32)
EXPECTED_MASK_RAW_BYTES = 1 * 512 * 512 * 1 * 4   # 1,048,576 bytes (float32)


def find_matching_file(directory: str, stem: str, valid_extensions=(".png", ".jpg", ".jpeg", ".bmp")):
    """Locates an image file with the given stem in directory, handling anomalies like 025jpg.jpg."""
    for ext in valid_extensions:
        candidate = os.path.join(directory, f"{stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
    # Fallback to prefix match (e.g. 025jpg.jpg matches stem 025)
    matches = glob.glob(os.path.join(directory, f"{stem}*.*"))
    valid_matches = [m for m in matches if os.path.splitext(m)[1].lower() in valid_extensions]
    return valid_matches[0] if valid_matches else None


def prep_102_benchmark(
    src_dir="Benchmark/dataset_previous",
    dest_dir="Benchmark/input_102",
    output_input_list="input_list_102.txt",
    qidk_img_dir="input/raw_image",
    qidk_mask_dir="input/raw_mask",
):
    print("=" * 70)
    print("🚀 Standardizing 102-Sample Dataset for SNPE/QNN NPU Inference")
    print(f"  Source Directory : {src_dir}")
    print(f"  Target Directory : {dest_dir}")
    print(f"  Input List File  : {output_input_list}")
    print("=" * 70)

    src_ideal_dir = os.path.join(src_dir, "ideal")
    src_images_dir = os.path.join(src_dir, "images")
    src_masks_dir = os.path.join(src_dir, "masks")

    for d in [src_ideal_dir, src_images_dir, src_masks_dir]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Required source directory not found: {d}")

    # Standardize all 102 sample keys from 001 to 102
    stems = [f"{i:03d}" for i in range(1, 103)]
    print(f"📁 Processing {len(stems)} target samples ({stems[0]} ... {stems[-1]})")

    # Output subdirectories
    gt_dir = os.path.join(dest_dir, "ground_truth")
    img_dir = os.path.join(dest_dir, "image")
    mask_dir = os.path.join(dest_dir, "mask")
    raw_img_dir = os.path.join(dest_dir, "raw_image")
    raw_mask_std_dir = os.path.join(dest_dir, "raw_mask_standard")
    raw_mask_inv_dir = os.path.join(dest_dir, "raw_mask_inverted")
    raw_mask_default_dir = os.path.join(dest_dir, "raw_mask")

    for d in [gt_dir, img_dir, mask_dir, raw_img_dir, raw_mask_std_dir, raw_mask_inv_dir, raw_mask_default_dir]:
        os.makedirs(d, exist_ok=True)

    processed_count = 0
    valid_entries = []

    for idx, stem in enumerate(stems, 1):
        ideal_path = find_matching_file(src_ideal_dir, stem)
        img_path = find_matching_file(src_images_dir, stem)
        mask_path = find_matching_file(src_masks_dir, stem)

        if not ideal_path or not img_path or not mask_path:
            print(f"⚠️ Warning: Incomplete triplet for sample '{stem}'. Skipping. (ideal={ideal_path}, img={img_path}, mask={mask_path})")
            continue

        # -------------------------------------------------------------
        # 1. Ground Truth: Bicubic downscale from ideal/ to 512x512 RGB
        # -------------------------------------------------------------
        with Image.open(ideal_path) as im_ideal:
            im_gt = im_ideal.convert("RGB")
            if im_gt.size != TARGET_SIZE:
                im_gt = im_gt.resize(TARGET_SIZE, Image.Resampling.BICUBIC)
            out_gt_path = os.path.join(gt_dir, f"{stem}.png")
            im_gt.save(out_gt_path, format="PNG")

        # -------------------------------------------------------------
        # 2. Corrupted Image: Standardize to 512x512 RGB
        # -------------------------------------------------------------
        with Image.open(img_path) as im_raw:
            im_corrupt = im_raw.convert("RGB")
            if im_corrupt.size != TARGET_SIZE:
                im_corrupt = im_corrupt.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
            out_img_path = os.path.join(img_dir, f"{stem}.png")
            im_corrupt.save(out_img_path, format="PNG")

        # -------------------------------------------------------------
        # 3. Mask: Standardize to 512x512 uint8 binary (255=hole, 0=keep)
        # -------------------------------------------------------------
        with Image.open(mask_path) as im_m:
            im_mask_l = im_m.convert("L")
            if im_mask_l.size != TARGET_SIZE:
                im_mask_l = im_mask_l.resize(TARGET_SIZE, Image.Resampling.NEAREST)
            mask_np = np.array(im_mask_l, dtype=np.uint8)
            # Strict binarization: >= 128 -> 255 (hole), else 0 (keep)
            binary_mask_255 = np.where(mask_np >= 128, 255, 0).astype(np.uint8)
            im_binary = Image.fromarray(binary_mask_255, mode="L")
            out_mask_path = os.path.join(mask_dir, f"{stem}.png")
            im_binary.save(out_mask_path, format="PNG")

        # -------------------------------------------------------------
        # 4. Generate .raw Binary Tensors matching SNPE/QNN DLC Specs
        # -------------------------------------------------------------
        # Image raw: NHWC (1, 512, 512, 3) float32 normalized to [0.0, 1.0]
        img_arr = np.array(im_corrupt, dtype=np.float32) / 255.0
        img_nhwc = np.expand_dims(img_arr, axis=0).astype(np.float32)  # (1, 512, 512, 3)
        raw_img_path = os.path.join(raw_img_dir, f"{stem}.raw")
        img_nhwc.tofile(raw_img_path)

        # Standard Mask raw (for LaMa Dilated & AOT-GAN):
        # 1.0 = hole (area to inpaint), 0.0 = keep (background)
        # NHWC (1, 512, 512, 1) float32
        mask_std = (binary_mask_255 >= 128).astype(np.float32)
        mask_std_nhwc = np.expand_dims(np.expand_dims(mask_std, axis=-1), axis=0).astype(np.float32)
        raw_mask_std_path = os.path.join(raw_mask_std_dir, f"{stem}_mask.raw")
        mask_std_nhwc.tofile(raw_mask_std_path)

        # Inverted Mask raw (for MIGAN):
        # 0.0 = hole (area to inpaint), 1.0 = keep (background)
        # NHWC (1, 512, 512, 1) float32
        mask_inv = 1.0 - mask_std
        mask_inv_nhwc = np.expand_dims(np.expand_dims(mask_inv, axis=-1), axis=0).astype(np.float32)
        raw_mask_inv_path = os.path.join(raw_mask_inv_dir, f"{stem}_mask.raw")
        mask_inv_nhwc.tofile(raw_mask_inv_path)

        # Default raw_mask: copy standard mask
        raw_mask_default_path = os.path.join(raw_mask_default_dir, f"{stem}_mask.raw")
        mask_std_nhwc.tofile(raw_mask_default_path)

        valid_entries.append(stem)
        processed_count += 1

        if idx % 20 == 0 or idx == len(stems):
            print(f"  -> Processed {idx}/{len(stems)} pairs (latest: '{stem}')")

    # -----------------------------------------------------------------
    # 5. Generate input_list_102.txt for On-Device Batch SNPE Execution
    # -----------------------------------------------------------------
    # Standard input_list (pointing to qidk_image_dir and qidk_mask_dir)
    input_list_paths = [
        output_input_list,                                # e.g. root input_list_102.txt
        os.path.join(dest_dir, "input_list_102.txt"),     # inside Benchmark/input_102/
    ]

    for p in input_list_paths:
        with open(p, "w") as f:
            for stem in valid_entries:
                line = f"image:={qidk_img_dir}/{stem}.raw mask:={qidk_mask_dir}/{stem}_mask.raw\n"
                f.write(line)
        print(f"✅ Generated batch execution list: {p} ({len(valid_entries)} entries)")

    # Model-specific variant lists
    migan_list_path = os.path.join(dest_dir, "input_list_102_migan.txt")
    with open(migan_list_path, "w") as f:
        for stem in valid_entries:
            f.write(f"image:={qidk_img_dir}/{stem}.raw mask:=input/raw_mask_inverted/{stem}_mask.raw\n")
    print(f"✅ Generated MIGAN inverted mask list: {migan_list_path}")

    lama_list_path = os.path.join(dest_dir, "input_list_102_lama_aotgan.txt")
    with open(lama_list_path, "w") as f:
        for stem in valid_entries:
            f.write(f"image:={qidk_img_dir}/{stem}.raw mask:=input/raw_mask_standard/{stem}_mask.raw\n")
    print(f"✅ Generated LaMa/AOT-GAN standard mask list: {lama_list_path}")

    # Local paths list (for local offline testing / scripts)
    local_list_path = os.path.join(dest_dir, "input_list_102_local.txt")
    with open(local_list_path, "w") as f:
        for stem in valid_entries:
            f.write(f"image:={os.path.abspath(raw_img_dir)}/{stem}.raw mask:={os.path.abspath(raw_mask_default_dir)}/{stem}_mask.raw\n")
    print(f"✅ Generated local absolute paths list: {local_list_path}")

    # Verification checks
    sample_stem = valid_entries[0]
    s_img_raw = os.path.join(raw_img_dir, f"{sample_stem}.raw")
    s_mask_std = os.path.join(raw_mask_std_dir, f"{sample_stem}_mask.raw")
    s_mask_inv = os.path.join(raw_mask_inv_dir, f"{sample_stem}_mask.raw")

    print("\n🔍 Tensor Binary Validation:")
    print(f"  Sample ID: {sample_stem}")
    print(f"  Image Raw Size: {os.path.getsize(s_img_raw):,} bytes (Expected: {EXPECTED_IMAGE_RAW_BYTES:,})")
    print(f"  Std Mask Size : {os.path.getsize(s_mask_std):,} bytes (Expected: {EXPECTED_MASK_RAW_BYTES:,})")
    print(f"  Inv Mask Size : {os.path.getsize(s_mask_inv):,} bytes (Expected: {EXPECTED_MASK_RAW_BYTES:,})")

    assert os.path.getsize(s_img_raw) == EXPECTED_IMAGE_RAW_BYTES, "Image raw size mismatch!"
    assert os.path.getsize(s_mask_std) == EXPECTED_MASK_RAW_BYTES, "Standard mask raw size mismatch!"
    assert os.path.getsize(s_mask_inv) == EXPECTED_MASK_RAW_BYTES, "Inverted mask raw size mismatch!"

    # Verify polarities
    m_std_arr = np.fromfile(s_mask_std, dtype=np.float32)
    m_inv_arr = np.fromfile(s_mask_inv, dtype=np.float32)
    assert np.allclose(m_std_arr + m_inv_arr, 1.0), "Standard + Inverted mask must equal 1.0 everywhere!"

    print("\n🎉 Dataset Standardization Complete!")
    print(f"  Total Valid Pairs Processed: {processed_count} / {len(stems)}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and standardize 102-sample dataset for NPU inference.")
    parser.add_argument("--src_dir", default="Benchmark/dataset_previous", help="Source directory containing images, masks, ideal")
    parser.add_argument("--dest_dir", default="Benchmark/input_102", help="Target standardized benchmark directory")
    parser.add_argument("--input_list", default="input_list_102.txt", help="Output path for SNPE batch input list")
    args = parser.parse_args()

    prep_102_benchmark(
        src_dir=args.src_dir,
        dest_dir=args.dest_dir,
        output_input_list=args.input_list
    )
