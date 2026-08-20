from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


def postprocess_output(
    output_raw_path: Union[str, Path],
    output_path: Union[str, Path] = "./output_result.png",
) -> None:
    # 1. Load raw binary and reshape to (3, 512, 512)
    output = np.fromfile(output_raw_path, dtype=np.uint8).reshape(3, 512, 512)

    # 2. Transpose CHW -> HWC (512, 512, 3)
    output = np.transpose(output, (1, 2, 0))

    # 3. Multiply by 255 and cast to uint8
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    # 4. Save image
    Image.fromarray(output).save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    postprocess_output(
        output_raw_path="./output/Result_0/output_0.raw",
        output_path="./output_result.png",
    )