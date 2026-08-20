import sys
import numpy as np
from PIL import Image


def prepare_inputs(image_path: str, mask_path: str) -> None:
    # 1. Load and resize
    img = Image.open(image_path).convert("RGB").resize((512, 512))
    mask = Image.open(mask_path).convert("L").resize((512, 512))

    # 2. Binarize & invert mask (0 = hole, 1 = keep)
    mask_np = (np.array(mask) >= 200).astype(np.float32)  # (512, 512)
    mask_np = 1.0 - mask_np

    # 3. Normalize image to [-1, 1]
    img_np = (
        np.array(img, dtype=np.float32) * 2.0 / 255.0 - 1.0
    )  # (512, 512, 3)

    # 4. Zero out the hole area
    img_np = img_np * mask_np[..., np.newaxis]

    # 5. Convert to NCHW format
    img_np = np.transpose(img_np, (2, 0, 1))  # (3, 512, 512)
    mask_np = mask_np[np.newaxis, ...]  # (1, 512, 512)

    # 6. Add batch dimension
    img_np = img_np[np.newaxis, ...].astype(np.float32)  # (1, 3, 512, 512)
    mask_np = mask_np[np.newaxis, ...].astype(np.float32)  # (1, 1, 512, 512)

    # 7. Save raw binary
    img_np.tofile("image.raw")
    mask_np.tofile("mask.raw")
    print("Saved 'image.raw' and 'mask.raw'")


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else "1.png"
    mask_arg = sys.argv[2] if len(sys.argv) > 2 else "1_mask.png"

    prepare_inputs(img_arg, mask_arg)