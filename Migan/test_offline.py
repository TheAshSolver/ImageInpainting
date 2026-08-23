import numpy as np
import onnxruntime as ort
from PIL import Image

# 1. Load the raw binary files directly into their NCHW shapes
# Based on earlier logs, Migan expects uint8 input types.
image_input = np.fromfile("image.raw", dtype=np.uint8).reshape(1, 3, 512, 512)
mask_input = np.fromfile("mask.raw", dtype=np.uint8).reshape(1, 1, 512, 512)

# 2. Invert the mask
# For a uint8 array (0-255), we invert it by subtracting it from 255.
mask_input = 255 - mask_input

# 3. Load the Model
# Note: ONNX Runtime will automatically find and load the 'migan_512_core.data' 
# weights file as long as it is kept in the same directory as the .onnx file.
print("Loading model...")
session = ort.InferenceSession("migan_512_core.onnx")

# 4. Run Inference
print("Running inference...")
inputs = {
    "image": image_input,
    "mask": mask_input
}
output_name = session.get_outputs()[0].name
outputs = session.run([output_name], inputs)
inpainted_tensor = outputs[0]

# 5. Post-process the Output
# Strip the batch dimension -> shape becomes [3, 512, 512]
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