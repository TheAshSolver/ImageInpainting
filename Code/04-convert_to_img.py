import os
import glob
import numpy as np
from PIL import Image

def convert_raw_to_images():
    base_dir = "Benchmark/output/"
    output_results_dir = os.path.join(base_dir, "results")
    
    # Create the final results directory
    os.makedirs(output_results_dir, exist_ok=True)
    
    # Spatial dimensions used during inference
    H, W = 512, 512

    # Find all Result_* directories (e.g., Result_0, Result_1)
    result_dirs = glob.glob(os.path.join(base_dir, "Result_*"))
    
    if not result_dirs:
        print(f"No Result directories found in {base_dir}")
        return

    for res_dir in result_dirs:
        # Extract the index number from the folder name (e.g., Result_0 -> 0)
        dir_name = os.path.basename(res_dir)
        try:
            index = dir_name.split("_")[1]
        except IndexError:
            continue

        # Target files inside each Result folder
        files_to_convert = ["output_0.raw", "painted_image.raw"]

        for file_name in files_to_convert:
            raw_path = os.path.join(res_dir, file_name)
            
            if not os.path.exists(raw_path):
                print(f"⚠️ Warning: {file_name} not found in {dir_name}. Skipping.")
                continue
            
            try:
                # 1. Read the raw binary data as uint8 (No Normalization)
                arr = np.fromfile(raw_path, dtype=np.float32)

                # 2. Determine number of channels by dividing total bytes by (H * W)
                pixels = H * W
                if len(arr) % pixels != 0:
                    print(f"⚠️ Warning: {raw_path} byte length ({len(arr)}) doesn't match {H}x{W} dimensions. Skipping.")
                    continue
                
                channels = len(arr) // pixels
                
                # 3. Reshape and convert to PIL Image
                if channels == 3:
                    # RGB format (512, 512, 3)
                    arr = arr.reshape((H, W, 3))
                    arr= np.clip((arr*255), 0, 255).astype(np.uint8)
                    img = Image.fromarray(arr, 'RGB')
                elif channels == 1:
                    # Grayscale format (512, 512)
                    arr = arr.reshape((H, W))
                    img = Image.fromarray(arr, 'L')
                else:
                    print(f"⚠️ Unexpected channel count ({channels}) in {raw_path}. Skipping.")
                    continue

                # 4. Save the image to the results folder
                # Output format: Benchmark/output/results/0_output0.png
                base_name = file_name.replace('.raw', '.png')
                out_name = f"{index}_{base_name}"
                out_path = os.path.join(output_results_dir, out_name)
                
                img.save(out_path)
                print(f"✅ Processed and saved: {out_path}")

            except Exception as e:
                print(f"❌ Error processing {raw_path}: {e}")

if __name__ == "__main__":
    print("Converting output .raw files to images...")
    convert_raw_to_images()
    print("Conversion complete!")