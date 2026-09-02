import os
import glob
import numpy as np
from PIL import Image
import argparse

def convert_to_raw_resized(normalised = False, mask_reversed = False):
    # Define directory paths
    base_dir = "Benchmark/input"
    img_dir = os.path.join(base_dir, "image")
    mask_dir = os.path.join(base_dir, "mask")
    out_img_dir = os.path.join(base_dir, "raw_image")
    out_mask_dir = os.path.join(base_dir, "raw_mask")

    # Target resolution
    target_size = (512, 512)

    # Create output directories if they don't exist
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Allowed image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    # Grab all images in the input/image directory
    image_files = []
    for ext in valid_extensions:
        image_files.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))

    if not image_files:
        print(f"No images found in {img_dir}. Please check your paths.")
        return

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
        
        if not mask_path:
            print(f"⚠️ Warning: Mask for image {filename} not found in {mask_dir}. Skipping.")
            continue

        try:
            # --- Process Image ---
            img = Image.open(img_path).convert('RGB')
            # Resize image (Bilinear is good for standard images)
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            if(normalised):
                img_arr = np.array(img, dtype=np.float32) / 255.0
            else:
                img_arr = np.array(img, dtype=np.float32)
            # Expand dimensions to create NHWC format (1, 512, 512, 3)
            img_nhwc = np.expand_dims(img_arr, axis=0)

            # --- Process Mask ---
            mask = Image.open(mask_path).convert('L')
            # Resize mask using NEAREST to avoid interpolating/blending distinct mask labels
            mask = mask.resize(target_size, Image.Resampling.NEAREST)
            mask_arr = np.array(mask, dtype=np.float32)
            if (normalised):
                mask_arr = np.array(mask, dtype=np.float32) / 255.0
                if(mask_reversed):
                    mask_arr = 1-mask_arr
                mask_arr = np.where(mask_arr > 0.5, 1.0, 0.0).astype(np.float32)
            else:
                if(mask_reversed):
                    mask_arr = 255-mask_arr
                mask_arr = np.array(mask, dtype=np.float32)
            
            # Add Channel dimension -> (512, 512, 1)
            mask_arr = np.expand_dims(mask_arr, axis=-1) 
            
            # Expand dimensions to create NHWC format (1, 512, 512, 1)
            mask_nhwc = np.expand_dims(mask_arr, axis=0)

            # --- Save to .raw ---
            out_img_path = os.path.join(out_img_dir, f"{int(number)-1}.raw")
            out_mask_path = os.path.join(out_mask_dir, f"{int(number)-1}_mask.raw")

            img_nhwc.tofile(out_img_path)
            mask_nhwc.tofile(out_mask_path)

            print(f"✅ Processed & Resized to 512x512: {number}")

        except Exception as e:
            print(f"❌ Error processing {number}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and resize images/masks to raw binaries.")
    
    parser.add_argument(
        "-n", "--normalise",
        action="store_true",
        help="Normalize pixel values to [0.0, 1.0] instead of [0.0, 255.0]"
    )
    parser.add_argument(
        "-r", "--reverse",
        action="store_true",
        help="Invert/reverse mask values (255 - mask)"
    )

    args = parser.parse_args()

    print("Starting conversion and resizing...")
    convert_to_raw_resized(normalised=args.normalise, mask_reversed=args.reverse)
    print("Conversion complete!")