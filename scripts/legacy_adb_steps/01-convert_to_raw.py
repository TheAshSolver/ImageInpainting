import os
import glob
import numpy as np
from PIL import Image
import argparse

# Model-specific presets for normalization and mask polarity
# In Benchmark/input/mask: 255 = Hole (area to inpaint), 0 = Keep (background)
MODEL_PRESETS = {
    "migan": {
        "normalise": True,
        "mask_reversed": True,  # MIGAN expects: 0 = hole, 1 = keep
        "description": "MIGAN: [0.0, 1.0] normalized, reversed mask (0=hole, 1=keep)",
    },
    "aotgan": {
        "normalise": True,
        "mask_reversed": False,  # AOT-GAN expects: 1 = hole, 0 = keep
        "description": "AOT-GAN: [0.0, 1.0] normalized, standard mask (1=hole, 0=keep)",
    },
    "lama": {
        "normalise": True,
        "mask_reversed": False,  # LaMa expects: 1 = hole, 0 = keep
        "description": "LaMa: [0.0, 1.0] normalized, standard mask (1=hole, 0=keep)",
    },
}


def convert_to_raw_resized(
    model=None,
    normalised=None,
    mask_reversed=None,
    base_dir="Benchmark/input",
    target_size=(512, 512),
):
    # Apply model preset if specified
    if model is not None:
        model_key = model.lower()
        if model_key not in MODEL_PRESETS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(MODEL_PRESETS.keys())}")
        preset = MODEL_PRESETS[model_key]
        if normalised is None:
            normalised = preset["normalise"]
        if mask_reversed is None:
            mask_reversed = preset["mask_reversed"]
        print(f"📌 Model: {model_key.upper()} -> {preset['description']}")
    else:
        if normalised is None:
            normalised = False
        if mask_reversed is None:
            mask_reversed = False

    print(f"⚙️ Configuration: normalise={normalised}, mask_reversed={mask_reversed}")

    img_dir = os.path.join(base_dir, "image")
    mask_dir = os.path.join(base_dir, "mask")
    out_img_dir = os.path.join(base_dir, "raw_image")
    out_mask_dir = os.path.join(base_dir, "raw_mask")

    # Create output directories if they don't exist
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Allowed image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    # Grab all images in the input/image directory
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))

    if not image_files:
        print(f"No images found in {img_dir}. Please check your paths.")
        return

    # Sort naturally by numerical index
    def get_index(path):
        base = os.path.splitext(os.path.basename(path))[0]
        return int(base) if base.isdigit() else base

    image_files.sort(key=get_index)

    success_count = 0
    for img_path in image_files:
        # Extract the base number from the filename
        filename = os.path.basename(img_path)
        number, _ = os.path.splitext(filename)

        # Search for the corresponding mask file
        mask_path = None
        for m_ext in valid_extensions:
            temp_path = os.path.join(mask_dir, f"{number}{m_ext}")
            if os.path.exists(temp_path):
                mask_path = temp_path
                break
            temp_mask_path = os.path.join(mask_dir, f"{number}_mask{m_ext}")
            if os.path.exists(temp_mask_path):
                mask_path = temp_mask_path
                break
        
        if not mask_path:
            print(f"⚠️ Warning: Mask for image {filename} not found in {mask_dir}. Skipping.")
            continue

        try:
            # --- Process Image ---
            img = Image.open(img_path).convert('RGB')
            # Resize image (Bilinear is standard for RGB images)
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            img_arr = np.array(img, dtype=np.float32)
            if normalised:
                img_arr = img_arr / 255.0
            # Expand dimensions to create NHWC format (1, 512, 512, 3)
            img_nhwc = np.expand_dims(img_arr, axis=0).astype(np.float32)

            # --- Process Mask ---
            mask = Image.open(mask_path).convert('L')
            # Resize mask using NEAREST to avoid interpolating/blending distinct mask labels
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
            mask_arr = np.array(mask, dtype=np.float32)

            # In Benchmark/input/mask: hole is 255 (to inpaint), background is 0
            # Threshold at 128 to get clean binary mask: 1.0 = hole, 0.0 = background
            hole_binary = (mask_arr >= 128.0).astype(np.float32)

            if mask_reversed:
                # Polarity inversion: hole becomes 0.0, background/keep becomes 1.0
                binary_mask = 1.0 - hole_binary
            else:
                # Standard polarity: hole is 1.0, background/keep is 0.0
                binary_mask = hole_binary

            if normalised:
                mask_arr = binary_mask
            else:
                mask_arr = binary_mask * 255.0
            
            # Add Channel dimension -> (512, 512, 1)
            mask_arr = np.expand_dims(mask_arr, axis=-1) 
            
            # Expand dimensions to create NHWC format (1, 512, 512, 1)
            mask_nhwc = np.expand_dims(mask_arr, axis=0).astype(np.float32)

            # --- Save to .raw ---
            out_img_path = os.path.join(out_img_dir, f"{number}.raw")
            out_mask_path = os.path.join(out_mask_dir, f"{number}_mask.raw")

            img_nhwc.tofile(out_img_path)
            mask_nhwc.tofile(out_mask_path)

            success_count += 1
            print(f"✅ Processed & Resized to {target_size[0]}x{target_size[1]}: {number}")

        except Exception as e:
            print(f"❌ Error processing {number}: {e}")

    print(f"\n✨ Done! Successfully processed {success_count}/{len(image_files)} pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert and resize images/masks to raw binaries with model-aware normalization and polarity."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["migan", "aotgan", "lama"],
        default=None,
        help="Target model preset: migan (norm, reversed mask), aotgan (norm, standard mask), lama (norm, standard mask)"
    )
    parser.add_argument(
        "-n", "--normalise",
        action="store_true",
        default=None,
        help="Normalize pixel values to [0.0, 1.0] instead of [0.0, 255.0] (overrides preset)"
    )
    parser.add_argument(
        "--no-normalise",
        action="store_false",
        dest="normalise",
        help="Keep pixel values in [0.0, 255.0]"
    )
    parser.add_argument(
        "-r", "--reverse",
        action="store_true",
        default=None,
        help="Invert/reverse mask values (0=hole, 1=keep) (overrides preset)"
    )
    parser.add_argument(
        "--no-reverse",
        action="store_false",
        dest="reverse",
        help="Keep standard mask values (1=hole, 0=keep)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="Benchmark/input",
        help="Base directory containing image/ and mask/ subfolders (default: Benchmark/input)"
    )

    args = parser.parse_args()

    print("Starting conversion and resizing...")
    convert_to_raw_resized(
        model=args.model,
        normalised=args.normalise,
        mask_reversed=args.reverse,
        base_dir=args.base_dir
    )
    print("Conversion complete!")