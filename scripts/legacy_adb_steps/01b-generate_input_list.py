import os
import glob

def generate_qidk_input_list(
    image_dir="Benchmark/input/raw_image", 
    mask_dir="Benchmark/input/raw_mask", 
    output_txt="input_list.txt",

    qidk_image_dir = "input/raw_image",
    qidk_mask_dir = "input/raw_mask",
):
    """
    Generates an input_list.txt file for QIDK.
    Args:
        image_dir (str): Relative path where the image .raw files are located.
        mask_dir (str): Relative path where the mask .raw files are located.
        output_txt (str): Path to save the generated input_list.txt.
    """
    
    # We will search for .raw files in the actual filesystem.
    # Adjust this search path if your script is running from a different root 
    # than where 'input/image/' is located (e.g., if it's inside Benchmark/).
    search_path = os.path.join(image_dir, "*.raw")
    image_files = glob.glob(search_path)
    
    if not image_files:
        print(f"⚠️ No .raw files found in {image_dir}.")
        return

    # Extract numbers from filenames, assuming format is '1.raw', '2.raw', etc.
    # This allows us to sort them numerically so the list is in logical order.
    valid_numbers = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        number, _ = os.path.splitext(filename)
        valid_numbers.append(number)
    
    # Sort naturally (e.g., 1, 2, 10 instead of 1, 10, 2)
    valid_numbers.sort(key=lambda x: int(x) if x.isdigit() else x)

    # Write to input_list.txt
    with open(output_txt, 'w') as f:
        for number in valid_numbers:
            # Construct the exact relative paths required by the tool
            img_path = f"{qidk_image_dir}/{number}.raw"
            mask_path = f"{qidk_mask_dir}/{number}_mask.raw"
            
            # Format required by QNN/SNPE: input_name:=path input_name2:=path2
            line = f"image:={img_path} mask:={mask_path}\n"
            f.write(line)
            
    print(f"✅ Successfully generated {output_txt} with {len(valid_numbers)} entries.")

if __name__ == "__main__":
    # If your .raw files were saved to "Benchmark/input/raw_image" in the previous script,
    # but you want the text file to read "input/image/...", you can run it like this:
    # (Just ensure the tool runs from the directory where 'input/image' exists)
    
    generate_qidk_input_list(
        image_dir="Benchmark/input/raw_image", 
        mask_dir="Benchmark/input/raw_mask",
        output_txt="input_list.txt"
    )