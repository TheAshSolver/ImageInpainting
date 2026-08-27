"""
generate_onnx.py

Exports a QNN/DLC-compatible ONNX model for MIGAN at a fixed resolution
(default 512x512).

Why this is different from the original export in create_onnx_pipeline.py:
-----------------------------------------------------------------------
The original MIGAN_Pipeline.forward() computed the inpainting bounding box
*from mask pixel values at runtime* (get_masked_bbox), then dynamically
sliced the image/mask to that box before running the model. When exported
to ONNX, this becomes a `Range` node whose limit is a *data-dependent*
tensor (not a constant, not even a symbolic shape dim) plus a data-dependent
slice. Regular ONNX Runtime can execute that fine, but the QNN converter
compiles to a fixed graph on the NPU and explicitly rejects data-dependent
Range bounds:

    ValueError: Dynamic value for tensor name: /Cast_4_output_0,
    is not supported.

The fix is to NOT export the bbox-finding / cropping / pasting-back logic
at all. Only the fixed-size "core" (resize-to-resolution -> normalize ->
MIGAN -> blend) is exported, with static shapes (batch=1, H=W=resolution).
The bbox/crop/paste logic is run in plain Python/NumPy on the host, outside
the ONNX/DLC graph -- see `migan_host_inference_example` at the bottom of
this file for how to wire that back together at inference time.

NHWC I/O (added):
------------------
The QNN HTP backend runs convolutions natively in NHWC (channel-last).
When the graph's declared I/O is NCHW (PyTorch/ONNX default), the DLC
converter inserts boundary `Transpose` ops to bridge NCHW <-> NHWC, and
in some SDK/graph configs those boundary transposes fail to compile/run
on HTP.

To avoid depending on the converter's automatic layout-conversion pass,
this graph now declares its ONNX inputs/outputs as NHWC directly, and
does the permute to NCHW (for the conv stack) and back to NHWC (for the
output) *inside* the graph using ordinary `permute` ops. Because the
declared I/O layout now already matches what HTP wants, the converter's
layout-propagation pass should not need to add any extra boundary
transpose nodes -- and if it does, they're now operating on data that's
already in the layout it wants, rather than fighting the graph's native
NCHW declaration.

Usage:
    python generate_onnx.py \
        --model-path Migan_512.pt \
        --resolution 512 \
        --output-dir ./onnx_out
"""

import argparse
import math
import numbers
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from PIL import Image

from lib.model_zoo.migan_inference import Generator as MIGAN


# ---------------------------------------------------------------------------
# GaussianSmoothing (unchanged, from
# https://github.com/yuval-alaluf/Attend-and-Excite/blob/main/utils/gaussian_smoothing.py)
# ---------------------------------------------------------------------------
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super().__init__()
        self.padding = (kernel_size - 1) // 2

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(f"Only 1, 2, 3 dims supported. Got {dim}.")

    def forward(self, input):
        input = F.pad(
            input,
            (self.padding, self.padding, self.padding, self.padding),
            mode="reflect",
        )
        return self.conv(
            input, weight=self.weight.to(input.dtype), groups=self.groups, padding="valid"
        )


# ---------------------------------------------------------------------------
# MIGANCore: everything EXCEPT the dynamic bbox / crop / paste-back logic.
# Fixed-size, static-shape only -> safe for QNN/DLC.
#
# I/O layout: NHWC in, NHWC out. Internally permutes to NCHW because the
# conv stack (MIGAN itself, GaussianSmoothing) expects channel-first.
# ---------------------------------------------------------------------------
class MIGANCore(nn.Module):
    """
    Takes an already-cropped image/mask pair (any size >= resolution is fine,
    since we resize to a fixed `resolution` internally) and returns the
    inpainted, blended crop at the SAME size as the input crop.

    image: (1, H, W, 3) uint8/float32, NHWC
    mask:  (1, H, W, 1) uint8/float32, NHWC   (255 = keep, 0 = hole, matching read_mask())
    result: (1, H, W, 3) float32, NHWC
    """

    def __init__(self, model_path, resolution, device="cpu"):
        super().__init__()
        self.model = MIGAN(resolution=resolution)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.gaussian_blur = GaussianSmoothing(
            channels=1, kernel_size=5, sigma=1.0, dim=2
        ).to(device)
        self.res = resolution

    def preprocess(self, image, mask):
        # image, mask are NCHW here (already permuted in forward()).
        image_r = tvF.resize(image, (self.res, self.res), interpolation=Image.BILINEAR)
        mask_r = tvF.resize(mask, (self.res, self.res), interpolation=Image.NEAREST)
        image_r = image_r.to(torch.float32) * 2 / 255 - 1
        mask_r = mask_r.to(torch.float32) / 255
        model_input = torch.cat([mask_r - 0.5, image_r * mask_r], dim=1)
        return model_input

    def postprocess(self, image, mask, model_output):
        model_output = ((model_output * 0.5 + 0.5) * 255).clamp(0, 255)
        model_output = tvF.resize(
            model_output, (self.res, self.res), interpolation=Image.BILINEAR
        )
        image = image.to(torch.float32)
        mask = mask.to(torch.float32)
        mask = F.max_pool2d(mask, 3, stride=1, padding=1)
        mask = self.gaussian_blur(mask)
        mask = mask / 255.0
        composed_img = image * mask + model_output * (1 - mask)
        return composed_img.clamp(0, 255).to(torch.float32)

    def forward(self, image_nhwc: torch.Tensor, mask_nhwc: torch.Tensor) -> torch.Tensor:
        # NHWC -> NCHW at the graph boundary. This becomes a plain
        # `Transpose` node in the exported ONNX graph, same as any other
        # op -- but now it's declared as part of the model's own compute
        # rather than something the DLC converter has to infer and inject.
        image = image_nhwc.permute(0, 3, 1, 2).contiguous()
        mask = mask_nhwc.permute(0, 3, 1, 2).contiguous()

        model_input = self.preprocess(image, mask)
        model_output = self.model(model_input)
        result_nchw = self.postprocess(image, mask, model_output)

        # NCHW -> NHWC before returning, so the graph's declared output
        # layout also matches what HTP prefers.
        result_nhwc = result_nchw.permute(0, 2, 3, 1).contiguous()
        return result_nhwc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/migan_512_places2.pt",
                         help="Path to the trained .pt checkpoint.")
    parser.add_argument("--resolution", type=int, default=512,
                         help="Model resolution (256 or 512).")
    parser.add_argument("--output-dir", type=Path, default=Path("./models"),
                         help="Where to write the .onnx file.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MIGAN core (resolution={args.resolution}) from {args.model_path} ...")
    core = MIGANCore(model_path=args.model_path, resolution=args.resolution, device=args.device)

    res = args.resolution
    # NHWC dummy tensors now: (1, H, W, C)
    dummy_image = torch.ones(1, res, res, 3, device=args.device, dtype=torch.float32) * 255
    dummy_mask = torch.ones(1, res, res, 1, device=args.device, dtype=torch.float32) * 255
    dummy_mask[:, res // 4, res // 4, :] = 0
    dummy_mask[:, res // 3, res // 2, :] = 0

    onnx_export_path = args.output_dir / f"migan_{res}_core_nhwc.onnx"

    print("Exporting ONNX model (fully static shapes: batch=1, "
          f"H=W={res}, NHWC I/O) ...")
    torch.onnx.export(
        core,
        (dummy_image, dummy_mask),
        str(onnx_export_path),
        verbose=False,
        export_params=True,
        # NOTE: no dynamic_axes on purpose. QNN/DLC wants a fixed compute
        # graph; since the crop/resize now happens on the host before the
        # image ever reaches this graph, batch/H/W can all be static.
        input_names=["image", "mask"],
        output_names=["result"],
        do_constant_folding=True,
        opset_version=17,
    )
    print(f"ONNX model exported to {onnx_export_path}")

    print("Verifying with onnxruntime ...")
    ort_sess = ort.InferenceSession(str(onnx_export_path))
    out = ort_sess.run(
        None,
        {
            "image": dummy_image.numpy(),
            "mask": dummy_mask.numpy(),
        },
    )
    print("OK - output shape:", out[0].shape, "dtype:", out[0].dtype)
    print("\nNext step: compile this .onnx to QNN DLC, declaring image/mask/result")
    print("as NHWC to the converter (see convert_to_dlc.sh / README notes).")


# ---------------------------------------------------------------------------
# Reference for how to use the exported model at inference time, now that
# bbox-finding / cropping / pasting-back live in plain Python instead of
# inside the ONNX graph, AND the graph's I/O is NHWC. Not called by main()
# -- included for reference.
# ---------------------------------------------------------------------------
def migan_host_inference_example(image_np, mask_np, ort_sess, resolution, padding=128):
    """
    image_np: (H, W, 3) uint8 numpy array
    mask_np:  (H, W) uint8 numpy array, 255=keep / 0=hole
    ort_sess: onnxruntime.InferenceSession for migan_{res}_core_nhwc.onnx
    """
    h, w = mask_np.shape
    ys, xs = np.where(mask_np < 255)
    if len(xs) == 0:
        return image_np  # nothing to inpaint

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cnt_x, cnt_y = (x_min + x_max) // 2, (y_min + y_max) // 2
    crop_size = max(x_max - x_min, y_max - y_min) + padding * 2
    crop_size = max(crop_size, resolution)
    offset = crop_size // 2

    x_min = max(cnt_x - offset, 0)
    x_max = min(cnt_x + offset, w)
    y_min = max(cnt_y - offset, 0)
    y_max = min(cnt_y + offset, h)

    # push back against the image edges so the crop stays crop_size x crop_size
    x_excess = max(crop_size - (x_max - x_min), 0)
    y_excess = max(crop_size - (y_max - y_min), 0)
    x_min = max(x_min - x_excess, 0)
    x_max = min(x_max + x_excess, w)
    y_min = max(y_min - y_excess, 0)
    y_max = min(y_max + y_excess, h)

    crop_img = image_np[y_min:y_max, x_min:x_max]        # (h, w, 3) -- already HWC
    crop_mask = mask_np[y_min:y_max, x_min:x_max]         # (h, w)

    # No more transpose(2, 0, 1): the graph now wants NHWC directly.
    inp_img = crop_img[None].astype(np.float32)                 # (1, h, w, 3)
    inp_mask = crop_mask[None, :, :, None].astype(np.float32)   # (1, h, w, 1)

    result = ort_sess.run(None, {"image": inp_img, "mask": inp_mask})[0]  # (1, h, w, 3) NHWC
    result = result[0]  # (h, w, 3) -- no transpose needed, already HWC

    out = image_np.copy()
    out[y_min:y_max, x_min:x_max] = result
    return out


if __name__ == "__main__":
    main()