import sys
from pathlib import Path
import numpy as np
from PIL import Image


def prepare_inputs(image_path: str, mask_path: str) -> None:
    # Load and resize
    img = Image.open(image_path).convert("RGB").resize((512, 512))
    mask = Image.open(mask_path).convert("L").resize((512, 512))

    # Normalize image to [0, 1]
    img_np = np.array(img, dtype=np.float32)/255   # (512, 512, 3)

    # Threshold mask at 200 to handle grey areas
    mask_np = (np.array(mask) >= 200).astype(np.float32)  # (512, 512)

    # Convert to CHW format for NPU
    # img_np = np.transpose(img_np)  # (3, 512, 512)
    mask_np = mask_np[..., np.newaxis]  # (512,512,1)

    # Add batch dimension
    img_np = img_np[np.newaxis, ...]  # (1, 3, 512, 512)
    mask_np = mask_np[np.newaxis, ...]  # (1,  512, 512, 1)

    # Save as raw binary
    img_np.tofile("image.raw")
    mask_np.tofile("mask.raw")
    print("Successfully generated image.raw and mask.raw")


if __name__ == "__main__":
    # Uses command line arguments if provided, otherwise defaults to local files
    img_file = sys.argv[1] if len(sys.argv) > 1 else "1.png"
    mask_file = sys.argv[2] if len(sys.argv) > 2 else "1_mask.png"

    prepare_inputs(img_file, mask_file)