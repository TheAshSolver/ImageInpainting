import os
import numpy as np
from PIL import Image

def check_and_save_raw(filename, expected_shape, mode):
    print(f"--- Checking {filename} ---")
    
    if not os.path.exists(filename):
        print(f"❌ Error: {filename} not found.")
        return

    # 1. Verify File Size
    # expected_shape * 1 byte (since uint8)
    expected_size = np.prod(expected_shape)
    actual_size = os.path.getsize(filename)
    
    print(f"File size: {actual_size} bytes (Expected: {expected_size} bytes)")
    if actual_size != expected_size:
        print(f"❌ Warning: Size mismatch! This means the file is either not uint8 or not the exact shape {expected_shape}.")
        print("-" * 30 + "\n")
        return

    # 2. Read and Reshape
    raw_array = np.fromfile(filename, dtype=np.uint8).reshape(expected_shape)
    
    # 3. Print Statistics
    print(f"Shape: {raw_array.shape}")
    print(f"Data Type: {raw_array.dtype}")
    print(f"Min Value: {raw_array.min()} (Should be >= 0)")
    print(f"Max Value: {raw_array.max()} (Should be <= 255)")
    print(f"Mean Value: {raw_array.mean():.2f}")

    # 4. Convert to Image and Save
    # Strip batch dimension
    img_data = raw_array[0] 
    
    if mode == 'RGB':
        # Convert from CHW (3, 512, 512) to HWC (512, 512, 3)
        img_data = np.transpose(img_data, (1, 2, 0))
        img = Image.fromarray(img_data, mode='RGB')
    elif mode == 'L':
        # For mask, it's (1, 512, 512), stripping batch made it (1, 512, 512)
        # We need it to be 2D (512, 512) for grayscale
        img_data = img_data[0]
        img = Image.fromarray(img_data, mode='L')
        
    save_name = f"check_{filename.split('.')[0]}.png"
    img.save(save_name)
    print(f"✅ Saved visualization to: {save_name}\n")


# Check the image
check_and_save_raw("image.raw", expected_shape=(1, 3, 512, 512), mode='RGB')

# Check the mask
check_and_save_raw("mask.raw", expected_shape=(1, 1, 512, 512), mode='L')

print("Check complete! Open the generated .png files to visually verify your inputs.")