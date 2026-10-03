from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


def postprocess_output(
    output_raw_path: Union[str, Path],
    original_image_path: Union[str, Path],
    mask_path: Union[str, Path],
    output_path: Union[str, Path] = "./output_blended.png",
    target_size: tuple[int, int] = (512, 512),
) -> None:
    """Loads a raw model tensor, loads original image and mask from file paths,

    denormalizes the output, blends them, and saves the result.
    """
    # 1. Resolve output destination (handle directories safely)
    destination = Path(output_path)
    if destination.is_dir():
        destination = destination / "output_blended.png"
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 2. Load & reshape raw model output (CHW -> HWC)
    # Size 196,608 = 3 * 256 * 256
    output = np.fromfile(output_raw_path, dtype=np.float32)
    output = output.reshape(1, target_size[1], target_size[0],3)[0]
    # output = np.transpose(output, (1, 2, 0))

    # Clean any NaN/Inf values if present
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

    # Denormalize from [-1, 1] to [0, 255]
    output = (output ) * 255.0
    output = np.clip(output, 0.0, 255.0)
    # 3. Load & resize original image
    original_img = (
        Image.open(original_image_path).convert("RGB").resize(target_size)
    )
    original_np = np.array(original_img, dtype=np.float32)
    

    # 4. Load mask from string path, convert to grayscale, and normalize to [0.0, 1.0]
    mask_img = Image.open(mask_path).convert("L").resize(target_size)
    mask_np = np.array(mask_img, dtype=np.float32) / 255.0
    mask_3ch = np.stack([mask_np] * 3, axis=-1)  # Broadcast to (H, W, 3)

    # 5. Blend: original * mask + output * (1 - mask)
    blended = original_np *(1- mask_3ch) + output * (mask_3ch)
    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

    # 6. Save image
    Image.fromarray(blended_uint8, mode='RGB' ).save(destination)
    print(f"Saved blended image to: {destination}")


def main():
    postprocess_output(
        output_raw_path="./output/Result_0/output_0.raw",
        original_image_path="./1.png",
        mask_path="./1_mask.png",
        output_path="./output_result.png",
    )


if __name__ == "__main__":
    main()