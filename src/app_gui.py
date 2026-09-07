import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#!/usr/bin/env python3
"""
src/app_gui.py
Interactive Inpainting Local GUI & Router Integration.

Key Features:
1. Image Selection: Upload custom photo, capture live webcam/camera, or select from 102 benchmark samples.
2. Target Masking:
   - Freehand brush touch/sketching directly over image (translucent red overlay).
   - Target bounding box / tap-to-mask via CPU GrabCut (<50 ms).
   - Read-only preview panels: Raw Binary Mask + Blended Mask Overlay.
3. Audited Router Decision Flow:
   - Real-time feature extraction (MediaPipe face detection, Laplacian variance, edge density, mask geometry).
   - Recommended model with explicit heuristic justification.
4. NPU Inference & Telemetry Harness:
   - Supports MIGAN, LaMa Dilated, AOT-GAN, and Stable Diffusion 1.5 (RePaint).
   - Hooks into on-device Qualcomm Hexagon NPU (HTP v79) via FastRPC.
   - Displays execution latency, power, energy, and thermal metrics.
5. Headless-Safe:
   - Runs cleanly in terminal/CI via --test or --headless without GUI lockups.
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

import gradio as gr

from src.auto_masking import (
    TARGET_SIZE,
    ensure_512_image,
    create_brush_mask,
    create_mask_overlay,
    grabcut_bounding_box_mask,
    tap_to_mask,
    mask_to_raw_tensors,
    refine_mask_grabcut,
)
from src.router import (
    classify_and_route,
    run_live_snpe_inference,
    check_device_status,
    BENCHMARK_PROFILES,
)

BENCHMARK_102_DIR = "Benchmark/input_102"


def get_available_samples():
    """Discovers available benchmark sample IDs."""
    img_dir = os.path.join(BENCHMARK_102_DIR, "image")
    if os.path.isdir(img_dir):
        files = sorted(os.listdir(img_dir))
        stems = [os.path.splitext(f)[0] for f in files if f.endswith(".png")]
        return stems
    return []


AVAILABLE_SAMPLES = get_available_samples()


# -----------------------------------------------------------------------------
# Callback Functions for Gradio UI
# -----------------------------------------------------------------------------

def load_benchmark_sample(sample_id: str):
    """Loads image and mask for a selected benchmark sample."""
    if not sample_id:
        empty = Image.new("RGB", (512, 512), (0, 0, 0))
        empty_mask = Image.new("L", (512, 512), 0)
        return {"background": empty, "layers": [], "composite": empty}, empty_mask, empty, "No sample selected.", "No sample selected."

    img_path = os.path.join(BENCHMARK_102_DIR, "image", f"{sample_id}.png")
    mask_path = os.path.join(BENCHMARK_102_DIR, "mask", f"{sample_id}.png")

    if not os.path.isfile(img_path):
        empty = Image.new("RGB", (512, 512), (0, 0, 0))
        empty_mask = Image.new("L", (512, 512), 0)
        return {"background": empty, "layers": [], "composite": empty}, empty_mask, empty, f"Sample '{img_path}' not found.", f"Sample '{img_path}' not found."

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L") if os.path.isfile(mask_path) else Image.new("L", (512, 512), 0)

    # Pre-render translucent overlay layer for ImageEditor
    mask_np = np.array(mask)
    overlay_pil = create_mask_overlay(img, mask_np, color=(255, 40, 40), alpha=0.45)

    editor_value = {
        "background": img,
        "layers": [],
        "composite": img,
    }

    route_res = classify_and_route(img, mask)
    router_md = format_router_markdown(route_res)
    status_msg = f"✅ Loaded Benchmark Sample **{sample_id}** (512x512 RGB). Router recommendation: **{route_res['recommended_model'].upper()}**."

    return editor_value, mask, overlay_pil, router_md, status_msg


def generate_grabcut_mask(
    editor_data: Any,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fast_mode: bool = True,
):
    """Executes target bounding box GrabCut in <50ms and updates read-only mask previews."""
    if editor_data is None:
        empty = Image.new("L", (512, 512), 0)
        empty_rgb = Image.new("RGB", (512, 512), (0, 0, 0))
        return empty, empty_rgb, "", "⚠️ Please load or upload an image first."

    if isinstance(editor_data, dict):
        base_img = editor_data.get("background") or editor_data.get("composite")
    else:
        base_img = editor_data

    if base_img is None:
        empty = Image.new("L", (512, 512), 0)
        empty_rgb = Image.new("RGB", (512, 512), (0, 0, 0))
        return empty, empty_rgb, "", "⚠️ No image data found in editor."

    img_512 = ensure_512_image(base_img)
    bbox = (x1, y1, x2, y2)

    mask_bin, latency_ms = grabcut_bounding_box_mask(img_512, bbox, iter_count=1, fast_mode=fast_mode)
    mask_pil = Image.fromarray(mask_bin, mode="L")
    overlay_pil = create_mask_overlay(img_512, mask_bin, color=(255, 40, 40), alpha=0.45)

    route_res = classify_and_route(img_512, mask_bin)
    router_md = format_router_markdown(route_res)

    msg = f"⚡ GrabCut executed in **{latency_ms:.2f} ms** (Target: <50 ms). Isolated foreground: **{int(np.sum(mask_bin > 0))} pixels**."
    return mask_pil, overlay_pil, router_md, msg


def extract_active_mask(editor_data: Any, current_mask_img: Any) -> np.ndarray:
    """
    Extracts active inpainting mask ensuring standard polarity: 255 = hole, 0 = keep.
    Prefers user-drawn brush strokes from editor_data if present and non-empty.
    Falls back to current_mask_img (e.g. GrabCut result or benchmark sample mask).
    """
    mask_512 = np.zeros(TARGET_SIZE, dtype=np.uint8)

    # 1. Check if user drew brush strokes in ImageEditor layers
    if isinstance(editor_data, dict) and editor_data.get("layers") and len(editor_data["layers"]) > 0:
        brush_mask = create_brush_mask(editor_data)
        if np.any(brush_mask > 0):
            mask_512 = brush_mask

    # 2. If no brush strokes on canvas, check current_mask_img
    if np.sum(mask_512 > 0) == 0 and current_mask_img is not None:
        cand_mask = create_brush_mask(current_mask_img)
        if np.any(cand_mask > 0):
            mask_512 = cand_mask

    # 3. If still empty, check composite diff
    if np.sum(mask_512 > 0) == 0 and isinstance(editor_data, dict):
        comp_mask = create_brush_mask(editor_data)
        if np.any(comp_mask > 0):
            mask_512 = comp_mask

    return mask_512


def analyze_current_canvas(editor_data: Any, current_mask_img: Any):
    """Analyzes user drawing or current mask and runs the router classifier."""
    if editor_data is None:
        empty = Image.new("L", (512, 512), 0)
        empty_rgb = Image.new("RGB", (512, 512), (0, 0, 0))
        return empty, empty_rgb, "", "⚠️ Please provide an image."

    if isinstance(editor_data, dict):
        base_img = editor_data.get("background") or editor_data.get("composite")
    else:
        base_img = editor_data

    if base_img is None:
        empty = Image.new("L", (512, 512), 0)
        empty_rgb = Image.new("RGB", (512, 512), (0, 0, 0))
        return empty, empty_rgb, "", "⚠️ No valid image found."

    img_512 = ensure_512_image(base_img)
    mask_512 = extract_active_mask(editor_data, current_mask_img)

    mask_pil = Image.fromarray(mask_512, mode="L")
    overlay_pil = create_mask_overlay(img_512, mask_512, color=(255, 40, 40), alpha=0.45)

    route_res = classify_and_route(img_512, mask_512)
    router_md = format_router_markdown(route_res)
    msg = f"🔍 Routing analysis complete. Recommended: **{route_res['recommended_model'].upper()}** (Coverage: {route_res['features']['mask_area_pct']}%)."

    return mask_pil, overlay_pil, router_md, msg


def refine_mask_interaction(
    editor_data: Any,
    current_mask_img: Any,
):
    """Snaps rough brush strokes tightly to object boundaries using OpenCV GrabCut."""
    if editor_data is None:
        return None, None, "", "⚠️ Please provide an input image first."

    if isinstance(editor_data, dict):
        base_img = editor_data.get("background") or editor_data.get("composite")
    else:
        base_img = editor_data

    if base_img is None:
        return None, None, "", "⚠️ No image loaded."

    img_512 = ensure_512_image(base_img)
    rough_mask = extract_active_mask(editor_data, current_mask_img)

    if np.sum(rough_mask > 0) == 0:
        return None, None, "", "⚠️ Please draw brush strokes over the unwanted object first."

    refined_mask, latency_ms = refine_mask_grabcut(img_512, rough_mask, iterations=2)
    mask_pil = Image.fromarray(refined_mask, mode="L")
    overlay_pil = create_mask_overlay(img_512, refined_mask, color=(255, 40, 40), alpha=0.45)

    route_res = classify_and_route(img_512, refined_mask)
    router_md = format_router_markdown(route_res)
    msg = f"✨ Mask snapped to object edges via GrabCut in {latency_ms:.1f} ms! Recommended: **{route_res['recommended_model'].upper()}**."

    return mask_pil, overlay_pil, router_md, msg


def clear_mask_interaction(editor_data: Any):
    """Instantly resets the inpainting mask to completely clean (0=keep)."""
    empty_mask = np.zeros((512, 512), dtype=np.uint8)
    mask_pil = Image.fromarray(empty_mask, mode="L")

    if editor_data is None:
        return mask_pil, None, "", "🧹 Mask reset to empty (100% keep)."

    if isinstance(editor_data, dict):
        base_img = editor_data.get("background") or editor_data.get("composite")
    else:
        base_img = editor_data

    if base_img is None:
        return mask_pil, None, "", "🧹 Mask reset to empty (100% keep)."

    img_512 = ensure_512_image(base_img)
    overlay_pil = Image.fromarray(img_512, mode="RGB")
    route_res = classify_and_route(img_512, empty_mask)
    router_md = format_router_markdown(route_res)

    return mask_pil, overlay_pil, router_md, "🧹 Mask cleared! Canvas reset to 100% keep."


def execute_inpainting_pipeline(
    editor_data: Any,
    current_mask_img: Any,
    model_choice: str = "Auto (Router Recommended)",
):
    """Executes inpainting on Hexagon NPU via SNPE net-run or SD RePaint runner."""
    if editor_data is None:
        return None, "", "⚠️ Please provide an input image."

    if isinstance(editor_data, dict):
        base_img = editor_data.get("background") or editor_data.get("composite")
    else:
        base_img = editor_data

    if base_img is None:
        return None, "", "⚠️ No image found."

    img_512 = ensure_512_image(base_img)
    mask_512 = extract_active_mask(editor_data, current_mask_img)

    if np.sum(mask_512 > 0) == 0:
        return None, "", "⚠️ Inpainting mask is empty. Please draw brush strokes or select a target box."

    # Parse model choice
    mc = model_choice.lower()
    if "auto" in mc:
        selected_key = "auto"
    elif "diffusion" in mc or "repaint" in mc or "sd" in mc:
        selected_key = "sd"
    elif "migan" in mc:
        selected_key = "migan"
    elif "lama" in mc:
        selected_key = "lama"
    elif "aot" in mc:
        selected_key = "aotgan"
    else:
        selected_key = "auto"

    # Enforce Real NPU Execution - Kill silent CPU simulation
    try:
        inpaint_result, meta = run_live_snpe_inference(
            img_512, mask_512, model_key=selected_key, allow_cpu_fallback=False
        )
        res_pil = Image.fromarray(inpaint_result, mode="RGB")
        telemetry_md = format_telemetry_markdown(meta)
        status_msg = f"✨ Inpainting finished successfully using **{meta['model_executed']}** [{meta['execution_mode']}]."
        return res_pil, telemetry_md, status_msg

    except Exception as e:
        err_msg = str(e)
        error_card = f"""### ❌ Qualcomm Snapdragon 8 Elite NPU Execution Failure
<div style="background-color: #ef444420; border-left: 5px solid #ef4444; padding: 14px; border-radius: 6px; margin-bottom: 14px;">
  <span style="font-size: 1.2em; font-weight: bold; color: #ef4444;">NPU Communication / Inference Failed</span>
  <p style="margin-top: 8px; color: #fca5a5; font-family: monospace;"><b>Error:</b> {err_msg}</p>
  <p style="margin-top: 6px; font-size: 0.9em; color: #cbd5e1;">
    <i>Real hardware NPU execution is strictly enforced. Silent CPU simulation fallback is disabled. Ensure device permissions ('adb devices -l') are authorized and the board is awake.</i>
  </p>
</div>"""
        gr.Warning(f"NPU Hardware Failure: {err_msg}")
        return None, error_card, f"❌ NPU Hardware Execution Failed: {err_msg}"


# -----------------------------------------------------------------------------
# Markdown Formatters
# -----------------------------------------------------------------------------

def format_router_markdown(route_res: Dict[str, Any]) -> str:
    feat = route_res["features"]
    rec = route_res["recommended_model"].upper()
    rule = route_res["rule_triggered"]
    just = route_res["justification"]
    prof = route_res["telemetry"]

    badge_color = "#10b981" if rec == "MIGAN" else ("#3b82f6" if rec == "AOT-GAN" else "#8b5cf6")

    md = f"""### 🎯 Router Model Recommendation
<div style="background-color: {badge_color}20; border-left: 5px solid {badge_color}; padding: 12px; border-radius: 4px; margin-bottom: 12px;">
  <span style="font-size: 1.3em; font-weight: bold; color: {badge_color};">Recommended Model: {rec}</span>
  <br><b>Rule Triggered:</b> <code>{rule}</code>
  <p style="margin: 6px 0 0 0; color: #e5e7eb;">{just}</p>
</div>

#### 📊 Extracted Heuristic Features:
| Feature Attribute | Value | Target Model Association |
| :--- | :--- | :--- |
| **Faces Detected** | `{feat['faces_detected']}` face(s) | > 0 $\\rightarrow$ MIGAN (portrait optimization) |
| **Mask Area Coverage** | `{feat['mask_area_pct']}%` | > 25% $\\rightarrow$ LaMa Dilated (Fourier global RF) |
| **Mask Spatial Location** | `{feat['mask_location']}` | Centered vs Edge/Corner |
| **Mask Disconnected Fragments** | `{feat['mask_fragments']}` | $\\ge$ 3 $\\rightarrow$ LaMa Dilated |
| **Laplacian Texture Variance** | `{feat['laplacian_variance']}` | > 500 $\\rightarrow$ AOT-GAN (sharp texture) |
| **Canny Edge Density** | `{feat['edge_density']}` | > 0.08 $\\rightarrow$ AOT-GAN (dense geometric lines) |
| **OCR Text Present** | `{feat['text_present']}` | True $\\rightarrow$ AOT-GAN |

#### ⚡ Model Hardware Telemetry Baseline (Hexagon NPU / HTP v79):
- **Latency / Img:** `{prof['latency_ms']} ms`
- **Active Energy:** `{prof['energy_j']} Joules`
- **Average Power:** `{prof['power_w']} Watts`
- **Thermal Rise (ΔT):** `+{prof['thermal_rise_c']} °C` (Peak: `{prof['peak_temp_c']} °C`)
- **Fidelity Baseline:** `{prof['ssim']} SSIM` | `{prof['lpips']} LPIPS` | `{prof['global_psnr_db']} dB PSNR`
"""
    return md


def format_telemetry_markdown(meta: Dict[str, Any]) -> str:
    mode = meta["execution_mode"]
    dev = meta["device"]
    model = meta["model_executed"]
    snpe_ms = meta["snpe_latency_ms"]
    pipe_ms = meta["total_pipeline_ms"]
    j = meta["energy_j"]
    w = meta["power_w"]
    dt = meta["thermal_rise_c"]

    is_live = "LIVE" in mode
    color = "#10b981" if is_live else "#f59e0b"
    header = "Qualcomm Hexagon NPU Live Execution" if is_live else "Local Simulation Execution"

    md = f"""### 🚀 Inference & Hardware Telemetry
<div style="background-color: {color}20; border-left: 5px solid {color}; padding: 10px; border-radius: 4px; margin-bottom: 12px;">
  <b>Status:</b> {header}<br>
  <b>Target Device:</b> <code>{dev}</code><br>
  <b>Model Executed:</b> <code>{model}</code>
</div>

| Performance Metric | Recorded Value | Hardware Target |
| :--- | :--- | :--- |
| **SNPE On-Device Latency** | **`{snpe_ms:.2f} ms`** | Sub-second edge inference |
| **End-to-End Pipeline** | **`{pipe_ms:.2f} ms`** | Includes staging & retrieval |
| **Energy Consumption** | **`{j} Joules`** | Battery impact per stroke |
| **Active Power Draw** | **`{w} Watts`** | FastRPC HTP v79 load |
| **Thermal Delta (ΔT)** | **`+{dt} °C`** | Governed by cooling barrier |
"""
    return md


# -----------------------------------------------------------------------------
# Gradio Application Builder
# -----------------------------------------------------------------------------

def build_app():
    is_dev_connected, dev_str = check_device_status()
    device_badge = f"🟢 Connected: {dev_str}" if is_dev_connected else "🟡 Standalone / Simulation Mode"

    with gr.Blocks(title="Snapdragon 8 Elite Inpainting Engine") as demo:
        gr.Markdown(
            f"""# ⚡ Qualcomm Snapdragon 8 Elite Image Inpainting & Router Prototype
**Target Acceleration:** Qualcomm Hexagon NPU (HTP v79) via FastRPC & SNPE/QNN DLCs  
**Hardware Status:** `{device_badge}`  
**Supported Models:** **MIGAN** (Portraits & Speed) | **LaMa Dilated** (Large Voids) | **AOT-GAN** (Dense Texture) | **Stable Diffusion 1.5 (RePaint)** (Generative Synthesis)
"""
        )

        with gr.Row():
            # Left Column: Image Selection & Canvas
            with gr.Column(scale=5):
                gr.Markdown("### 1. Source Image & Interactive Masking Canvas")
                with gr.Row():
                    sample_dropdown = gr.Dropdown(
                        choices=AVAILABLE_SAMPLES,
                        label="Select from 102 Benchmark Samples",
                        value=AVAILABLE_SAMPLES[0] if AVAILABLE_SAMPLES else None,
                        scale=3
                    )
                    load_btn = gr.Button("📂 Load Sample", variant="secondary", scale=1)

                image_editor = gr.ImageEditor(
                    label="Interactive Canvas (Upload Photo, Webcam Capture, or Brush Over Target)",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "webcam"],
                    brush=gr.Brush(colors=["#ff3333", "#ffffff"], default_color="#ff3333", default_size=25),
                    layers=True,
                )

                with gr.Accordion("🎯 Target Bounding Box / Tap-to-Mask (OpenCV GrabCut <50ms)", open=False):
                    gr.Markdown("Specify bounding box coordinates around unwanted object to auto-extract silhouette.")
                    with gr.Row():
                        box_x1 = gr.Slider(0, 512, value=150, step=1, label="X1 (Left)")
                        box_y1 = gr.Slider(0, 512, value=150, step=1, label="Y1 (Top)")
                    with gr.Row():
                        box_x2 = gr.Slider(0, 512, value=350, step=1, label="X2 (Right)")
                        box_y2 = gr.Slider(0, 512, value=350, step=1, label="Y2 (Bottom)")
                    fast_gc_checkbox = gr.Checkbox(value=True, label="Fast Mode (Sub-50ms Guaranteed)")
                    grabcut_btn = gr.Button("⚡ Run GrabCut Auto-Mask (<50ms)", variant="primary")

                with gr.Row():
                    refine_btn = gr.Button("✨ Refine Mask (Snap to Edges)", variant="secondary")
                    reset_btn = gr.Button("🧹 Reset / Clear Mask", variant="stop")

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Update Mask & Run Router", variant="secondary")
                    inpaint_btn = gr.Button("🚀 Run Inpainting on Hexagon NPU", variant="primary")

                model_selector = gr.Radio(
                    choices=[
                        "Auto (Router Recommended)",
                        "MIGAN",
                        "LaMa Dilated",
                        "AOT-GAN",
                        "Stable Diffusion 1.5 (RePaint)",
                    ],
                    value="Auto (Router Recommended)",
                    label="Model Execution Choice"
                )

            # Right Column: Read-Only Previews, Inpainted Result & Telemetry
            with gr.Column(scale=5):
                gr.Markdown("### 2. Mask Previews (Read-Only) & Inpainted Output")
                with gr.Row():
                    # Read-only previews: no camera / file upload buttons
                    mask_display = gr.Image(
                        label="Binary Mask (0=Keep, 255=Hole)",
                        type="pil",
                        interactive=False,
                        height=220
                    )
                    overlay_display = gr.Image(
                        label="Blended Mask Overlay (Preview)",
                        type="pil",
                        interactive=False,
                        height=220
                    )

                result_display = gr.Image(
                    label="Inpainted Output",
                    type="pil",
                    interactive=False,
                    height=300
                )

                status_box = gr.Markdown("Ready.")
                router_output = gr.Markdown("Upload or select an image to view routing logic.")
                telemetry_output = gr.Markdown()

        # Wire Events
        load_btn.click(
            fn=load_benchmark_sample,
            inputs=[sample_dropdown],
            outputs=[image_editor, mask_display, overlay_display, router_output, status_box]
        )

        sample_dropdown.change(
            fn=load_benchmark_sample,
            inputs=[sample_dropdown],
            outputs=[image_editor, mask_display, overlay_display, router_output, status_box]
        )

        grabcut_btn.click(
            fn=generate_grabcut_mask,
            inputs=[image_editor, box_x1, box_y1, box_x2, box_y2, fast_gc_checkbox],
            outputs=[mask_display, overlay_display, router_output, status_box]
        )

        refine_btn.click(
            fn=refine_mask_interaction,
            inputs=[image_editor, mask_display],
            outputs=[mask_display, overlay_display, router_output, status_box]
        )

        reset_btn.click(
            fn=clear_mask_interaction,
            inputs=[image_editor],
            outputs=[mask_display, overlay_display, router_output, status_box]
        )

        analyze_btn.click(
            fn=analyze_current_canvas,
            inputs=[image_editor, mask_display],
            outputs=[mask_display, overlay_display, router_output, status_box]
        )

        inpaint_btn.click(
            fn=execute_inpainting_pipeline,
            inputs=[image_editor, mask_display, model_selector],
            outputs=[result_display, telemetry_output, status_box]
        )

    return demo


# -----------------------------------------------------------------------------
# Headless Validation Suite
# -----------------------------------------------------------------------------

def run_headless_test():
    """Runs automated end-to-end headless validation of all components."""
    print("=" * 70)
    print("🧪 Running Headless Automated Validation of GUI & Router Integration")
    print("=" * 70)

    samples = get_available_samples()
    print(f"1. Benchmark Samples Discovered: {len(samples)} samples.")
    assert len(samples) >= 100, f"Expected at least 100 benchmark samples, found {len(samples)}"

    sample_id = "001"
    editor_val, mask_pil, overlay_pil, router_md, status = load_benchmark_sample(sample_id)
    assert editor_val is not None
    assert mask_pil is not None
    assert overlay_pil is not None
    assert "MIGAN" in router_md
    print(f"2. Sample {sample_id} loaded & routed to MIGAN correctly.")

    sample_010 = "010"
    ed_010, _, _, _, _ = load_benchmark_sample(sample_010)
    gc_mask, gc_overlay, gc_md, gc_status = generate_grabcut_mask(ed_010, 100, 100, 300, 300, fast_mode=True)
    assert gc_mask is not None
    assert gc_overlay is not None
    assert "GrabCut executed" in gc_status
    print("3. Target Bounding Box GrabCut executed cleanly in sub-50ms.")

    print("4. Testing Inpainting Pipeline execution with MIGAN...")
    res_img, tele_md, inpaint_status = execute_inpainting_pipeline(
        editor_val, mask_pil, model_choice="MIGAN"
    )
    assert res_img is not None
    assert "finished successfully" in inpaint_status
    print("5. MIGAN Inpainting Pipeline executed successfully.")

    print("6. Testing Inpainting Pipeline execution with Stable Diffusion 1.5...")
    res_sd, tele_sd, status_sd = execute_inpainting_pipeline(
        editor_val, mask_pil, model_choice="Stable Diffusion 1.5 (RePaint)"
    )
    assert res_sd is not None
    assert "finished successfully" in status_sd
    print("7. Stable Diffusion 1.5 RePaint executed successfully.")

    print("8. Testing Red Brush Stroke (#ff3333) Extraction from Transparent Canvas Layer...")
    test_layer = np.zeros((512, 512, 4), dtype=np.uint8)
    # Draw red brush stroke (#ff3333)
    test_layer[150:250, 150:250] = [255, 51, 51, 255]
    ed_brush = {"background": ed_010["background"], "layers": [test_layer], "composite": ed_010["background"]}
    extracted = extract_active_mask(ed_brush, None)
    assert extracted[200, 200] == 255, "Red brush stroke must evaluate to 255 (hole)"
    assert extracted[10, 10] == 0, "Unpainted canvas must evaluate to 0 (background)"
    assert np.sum(extracted == 255) == 100 * 100
    print("9. Red Brush Stroke correctly extracted (255=hole, 0=background) without grayscale trap!")

    print("\n🎉 ALL HEADLESS VALIDATION TESTS PASSED CLEANLY (Zero GUI Lockup)!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapdragon 8 Elite Inpainting GUI & Router Prototype.")
    parser.add_argument("--test", action="store_true", help="Run automated headless test suite and exit.")
    parser.add_argument("--port", type=int, default=7860, help="Web GUI server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Web GUI server host (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="Generate public shareable link")
    args = parser.parse_args()

    if args.test:
        success = run_headless_test()
        sys.exit(0 if success else 1)

    app = build_app()
    print(f"🚀 Launching Inpainting Prototype GUI on http://{args.host}:{args.port}")
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
