import os
import glob
import numpy as np
from PIL import Image
import argparse

# Primary and fallback raw output filenames produced by models
MODEL_RAW_CANDIDATES = {
    "lama": ["painted_image.raw", "output_0.raw"],
    "migan": ["output_0.raw", "painted_image.raw"],
    "aotgan": ["output_0.raw", "painted_image.raw"],
}


def convert_raw_to_images(
    model="migan",
    base_dir=None,
    output_results_dir=None,
    H=512,
    W=512
):
    model_key = model.lower() if model else "migan"

    # Default base_dir resolves to Benchmark/output/<model>, or Benchmark/output if not present
    if base_dir is None:
        model_dir = os.path.join("Benchmark/output", model_key)
        if os.path.exists(model_dir):
            base_dir = model_dir
        else:
            base_dir = "Benchmark/output"

    if output_results_dir is None:
        output_results_dir = os.path.join(base_dir, "results")
    
    os.makedirs(output_results_dir, exist_ok=True)

    # Find all Result_* directories (e.g., Result_0, Result_1)
    result_dirs = glob.glob(os.path.join(base_dir, "Result_*"))
    
    if not result_dirs:
        print(f"No Result_* directories found in {base_dir}")
        return

    # Sort numerically: Result_0, Result_1, ..., Result_16
    def get_result_index(path):
        dir_name = os.path.basename(path)
        try:
            return int(dir_name.split("_")[1])
        except (IndexError, ValueError):
            return 999999

    result_dirs.sort(key=get_result_index)

    print(f"\n--- Converting RAW Outputs to Images ---")
    print(f"  Model             : {model_key.upper()}")
    print(f"  Source Directory  : {base_dir}")
    print(f"  Output Directory  : {output_results_dir}")
    print(f"  Found Results     : {len(result_dirs)}")
    print(f"----------------------------------------")

    preferred_files = MODEL_RAW_CANDIDATES.get(model_key, ["output_0.raw", "painted_image.raw"])

    converted_count = 0
    for res_dir in result_dirs:
        dir_name = os.path.basename(res_dir)
        try:
            index = dir_name.split("_")[1]
        except IndexError:
            continue

        # Search for the relevant raw file for this model
        raw_path = None
        for candidate in preferred_files:
            p = os.path.join(res_dir, candidate)
            if os.path.isfile(p):
                raw_path = p
                break

        if not raw_path:
            print(f"⚠️ Warning: None of {preferred_files} found in {dir_name}. Skipping.")
            continue

        try:
            # 1. Read raw binary data as float32
            arr = np.fromfile(raw_path, dtype=np.float32)

            # 2. Determine number of channels
            pixels = H * W
            if len(arr) % pixels != 0:
                print(f"⚠️ Warning: {raw_path} element count ({len(arr)}) doesn't match {H}x{W}. Skipping.")
                continue
            
            channels = len(arr) // pixels
            raw_min = float(arr.min()) if len(arr) > 0 else 0.0
            raw_max = float(arr.max()) if len(arr) > 0 else 0.0

            # 3. Auto-scale [0.0, 1.0] range to [0.0, 255.0]
            # When raw_max <= 1.05, the tensor is in normalized range [0.0, 1.0] or [-1.0, 1.0]
            if raw_max <= 1.05:
                if raw_min < -0.1:
                    # [-1.0, 1.0] -> [0.0, 255.0]
                    arr = (arr + 1.0) * 0.5 * 255.0
                else:
                    # [0.0, 1.0] -> [0.0, 255.0]
                    arr = arr * 255.0

            # Clip to valid [0, 255] and cast to uint8
            arr = np.clip(arr, 0, 255).astype(np.uint8)

            # 4. Reshape and convert to PIL Image
            if channels == 3:
                # RGB format (512, 512, 3)
                arr = arr.reshape((H, W, 3))
                img = Image.fromarray(arr, 'RGB')
            elif channels == 1:
                # Grayscale format (512, 512)
                arr = arr.reshape((H, W))
                img = Image.fromarray(arr, 'L')
            else:
                print(f"⚠️ Unexpected channel count ({channels}) in {raw_path}. Skipping.")
                continue

            # 5. Save named file as {idx}_{model}.png
            out_name = f"{index}_{model_key}.png"
            out_path = os.path.join(output_results_dir, out_name)
            
            img.save(out_path)
            converted_count += 1
            scaled_info = "(auto-scaled *255)" if raw_max <= 1.05 else "(already [0, 255])"
            print(f"✅ Saved: {out_name} from {dir_name}/{os.path.basename(raw_path)} [range: {raw_min:.2f}..{raw_max:.2f}] {scaled_info}")

        except Exception as e:
            print(f"❌ Error processing {raw_path}: {e}")

    print(f"\n🎉 Successfully converted {converted_count} images to {output_results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert output .raw files to auto-scaled images named {idx}_{model}.png."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["migan", "aotgan", "lama"],
        default="migan",
        help="Target model name for file naming ({idx}_{model}.png). Default: migan"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Base directory containing Result_* folders (default: Benchmark/output/<model>)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to save images (default: <base_dir>/results)"
    )

    args = parser.parse_args()

    print("Converting output .raw files to images...")
    convert_raw_to_images(
        model=args.model,
        base_dir=args.base_dir,
        output_results_dir=args.output_dir
    )
    print("Conversion complete!")