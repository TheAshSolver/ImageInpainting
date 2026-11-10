import os
import torch
import numpy as np
from mobile_sam import sam_model_registry

def export_mobilesam():
    model_type = "vit_t"
    checkpoint = "models/mobile_sam.pt"
    os.makedirs("models", exist_ok=True)
    
    print(f"Loading MobileSAM checkpoint: {checkpoint}...")
    mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
    mobile_sam.eval()

    # ----------------------------------------------------------------------
    # 1. Export Image Encoder
    # ----------------------------------------------------------------------
    class ImageEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, x):
            return self.encoder(x)

    encoder_wrapper = ImageEncoderWrapper(mobile_sam.image_encoder)
    dummy_img = torch.randn(1, 3, 1024, 1024)

    encoder_path = "models/mobilesam_image_encoder.onnx"
    print(f"Exporting image encoder to {encoder_path}...")
    torch.onnx.export(
        encoder_wrapper,
        dummy_img,
        encoder_path,
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=17,
        do_constant_folding=True
    )
    print("Encoder export complete.")

    # ----------------------------------------------------------------------
    # 2. Export Interactive Mask Decoder (Supports Point Prompts & Whole-Object Multimask)
    # ----------------------------------------------------------------------
    class InteractivePromptDecoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.prompt_encoder = model.prompt_encoder
            self.mask_decoder = model.mask_decoder

        def forward(self, image_embeddings, point_coords, point_labels):
            # point_coords: [1, N, 2], point_labels: [1, N]
            points = (point_coords, point_labels)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            return low_res_masks, iou_predictions

    decoder_wrapper = InteractivePromptDecoderWrapper(mobile_sam)
    dummy_embeddings = torch.randn(1, 256, 64, 64)
    dummy_points = torch.tensor([[[256.0 * 2.0, 250.0 * 2.0]]], dtype=torch.float32)
    dummy_labels = torch.tensor([[1]], dtype=torch.int32)

    decoder_path = "models/mobilesam_circle_decoder.onnx"
    print(f"Exporting prompt decoder to {decoder_path}...")
    torch.onnx.export(
        decoder_wrapper,
        (dummy_embeddings, dummy_points, dummy_labels),
        decoder_path,
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["low_res_masks", "iou_predictions"],
        opset_version=17,
        do_constant_folding=True
    )
    print("Decoder export complete.")
    print("ONNX export complete: decoupled encoder and circle-prompt decoder ready.")

if __name__ == "__main__":
    export_mobilesam()
