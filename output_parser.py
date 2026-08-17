import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union


def postprocess_output(
    output_raw_path: Union[str, Path],
    # original_image_path: Union[str, Path],
    # mask_np: np.ndarray,
    output_path: Union[str, Path],
) -> None:
    """Post-processes raw model output tensor, un-normalizes pixel values,

    and blends the result with the original image using a mask.
    """
    # 1. Load raw binary output (1, 3, 512, 512) in [-1, 1]
    output = np.fromfile(output_raw_path, dtype=np.float32)
    print(output)
    output = output.reshape(1,  512, 512, 3)[0]  # Shape: (3, 512, 512)
    
    # output = np.transpose(output, (1, 2, 0))  # Shape: (512, 512, 3)

    # 2. Denormalize from [-1, 1] -> [0, 255]
    output = (output) * 255.0
    
    output = np.clip(output, 0, 255).astype(np.uint8)
    print(output)
    # # 3. Load & resize original image (enforcing 3-channel RGB)
    # original_img = (
    #     Image.open(original_image_path).convert("RGB").resize((512, 512))
    # )
    # original = np.array(original_img, dtype=np.float32)

    # # 4. Prepare mask (ensure 3 channels and proper broadcasting)
    # if mask_np.ndim == 2:
    #     mask_3ch = np.stack([mask_np] * 3, axis=-1).astype(np.float32)
    # else:
    #     mask_3ch = mask_np.astype(np.float32)

    # # 5. Blend: original * mask + output * (1 - mask)
    # blended = original * mask_3ch + output * (1.0 - mask_3ch)
    # blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

    # 6. Save final composite image
    Image.fromarray(output, mode='RGB').save(output_path)



def main():
    postprocess_output("./output/Result_0/painted_image.raw", "./output.png")

main()