import qai_hub as hub
import numpy as np
from PIL import Image
# 1. Load and prepare your inputs (same as before)
image_input = np.fromfile("image.raw", dtype=np.uint8).reshape(1, 3, 512, 512)
mask_input = np.fromfile("mask.raw", dtype=np.uint8).reshape(1, 1, 512, 512)
mask_input = 255 - mask_input

client = hub.Client()

# 2. Select a cloud-hosted hardware device
# You can choose various Snapdragon tiers (e.g., Galaxy S23, S24, or Snapdragon X Elite)
device = hub.Device("Samsung Galaxy S24 (Family)")

# 3. Submit the Inference Job
print(f"Uploading migan.dlc and inputs to {device.name}...")
inference_job = client.submit_inference_job(
    model="migan.dlc",  # The path to your compiled DLC file
    device=device,
    inputs={
        "image": [image_input],
        "mask": [mask_input]
    },
    name="Migan_Cloud_Inference",
    options="--compute_unit cpu",
)

# 4. Wait for completion and download the output
print("Running inference on real hardware...")
output_data = inference_job.download_output_data()

# The results are returned as a dictionary of numpy arrays
# Extract the output using the model's output node name (usually something like 'output_0')
output_name = list(output_data.keys())[0]
inpainted_tensor = output_data[output_name][0]

print("Inference successful! Output shape:", inpainted_tensor.shape)

# (You can then process and save 'inpainted_tensor' using the PIL code from earlier)
output_array = inpainted_tensor[0]

# Convert from Channels-First (CHW) back to Channels-Last (HWC) -> [512, 512, 3]
output_array = np.transpose(output_array, (1, 2, 0))

# Safely handle the output depending on whether the model outputs float or uint8
if output_array.dtype in [np.float32, np.float16, np.float64]:
    # Denormalize floats [0.0, 1.0] to [0, 255]
    final_image_array = np.clip(output_array * 255.0, 0, 255).astype(np.uint8)
else:
    # Already uint8, just ensure bounds and type
    final_image_array = np.clip(output_array, 0, 255).astype(np.uint8)

# 6. Save the Result
image = Image.fromarray(final_image_array, mode='RGB')
image.save("result_inpainted.png")

print("Successfully saved the inpainted image to result_inpainted.png!")