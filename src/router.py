import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
src/router.py
Decision flow router and SNPE/QNN/SD execution harness.

Audited Decision Logic:
- Facial geometry detected -> MIGAN (facial optimization, 216ms, 0.62J)
- Large mask void (>25% area) or high fragmentation (>=3 holes) -> LaMa Dilated (FFC global receptive field)
- High edge density (>0.08) or Laplacian texture (>500.0) -> AOT-GAN (optimal contextual sharpness, SSIM 0.921)
- Smooth / low texture baseline -> MIGAN (low energy edge default)
- Explicit selection: Stable Diffusion 1.5 (RePaint) (generative hallucination, 50.9s, 135J)
"""

import os
import sys
import time
import subprocess
import tempfile
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, Union

from src.auto_masking import ensure_512_image, mask_to_raw_tensors, composite_inpaint_result

# Model benchmark telemetry baseline on Snapdragon 8 Elite (Hexagon NPU / HTP v79)
BENCHMARK_PROFILES = {
    "migan": {
        "name": "MIGAN",
        "dlc": "migan.dlc",
        "runtime": "dsp",
        "runtime_flag": "--use_dsp",
        "latency_ms": 216.0,
        "energy_j": 0.62,
        "power_w": 2.86,
        "peak_temp_c": 48.4,
        "thermal_rise_c": 9.6,
        "global_psnr_db": 27.17,
        "hole_psnr_db": 19.65,
        "ssim": 0.9028,
        "lpips": 0.1245,
        "ram_gb": 2.42,
        "architecture": "Multi-scale depthwise separable GAN (facial & portrait optimization)"
    },
    "lama": {
        "name": "LaMa Dilated",
        "dlc": "lama_dilated.dlc",
        "runtime": "gpu",
        "runtime_flag": "--use_gpu",
        "latency_ms": 321.0,
        "energy_j": 0.99,
        "power_w": 3.10,
        "peak_temp_c": 70.3,
        "thermal_rise_c": 24.6,
        "global_psnr_db": 28.22,
        "hole_psnr_db": 20.73,
        "ssim": 0.9193,
        "lpips": 0.1195,
        "ram_gb": 3.00,
        "architecture": "Fast Fourier Convolutions (FFC) with global receptive field"
    },
    "aotgan": {
        "name": "AOT-GAN",
        "dlc": "aotgan.dlc",
        "runtime": "gpu",
        "runtime_flag": "--use_gpu",
        "latency_ms": 390.0,
        "energy_j": 1.30,
        "power_w": 3.34,
        "peak_temp_c": 68.0,
        "thermal_rise_c": 25.4,
        "global_psnr_db": 28.34,
        "hole_psnr_db": 20.86,
        "ssim": 0.9214,
        "lpips": 0.1058,
        "ram_gb": 2.89,
        "architecture": "Aggregated Contextual Transformations (dense texture & sharp edges)"
    },
    "sd": {
        "name": "Stable Diffusion 1.5 (RePaint)",
        "dlc": "sd_runtime",
        "runtime": "htp_v79",
        "runtime_flag": "--use_htp",
        "latency_ms": 50930.0,
        "energy_j": 135.02,
        "power_w": 2.65,
        "peak_temp_c": 74.9,
        "thermal_rise_c": 30.0,
        "global_psnr_db": 10.13,
        "hole_psnr_db": 9.66,
        "ssim": 0.3183,
        "lpips": 0.7216,
        "ram_gb": 4.20,
        "architecture": "Diffusion Generative Model (20-step stochastic RePaint schedule)"
    }
}


# -----------------------------------------------------------------------------
# Feature Extraction Functions
# -----------------------------------------------------------------------------

def detect_faces(img_rgb: np.ndarray) -> int:
    """Detects number of human faces using MediaPipe or OpenCV Haar Cascade fallback."""
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
                res = fd.process(img_rgb)
                if res and res.detections:
                    return len(res.detections)
                return 0
    except Exception:
        pass

    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        return len(faces)
    except Exception:
        return 0


def compute_laplacian_variance(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def compute_edge_density(img_gray: np.ndarray) -> float:
    edges = cv2.Canny(img_gray, 100, 200)
    return float(np.sum(edges > 0) / edges.size)


def detect_text(img_rgb: np.ndarray) -> bool:
    try:
        import pytesseract
        txt = pytesseract.image_to_string(img_rgb)
        return len(txt.strip()) > 1
    except Exception:
        return False


def analyze_mask_geometry(mask: np.ndarray) -> Dict[str, Any]:
    mask_bin = (mask >= 128).astype(np.uint8)
    h, w = mask_bin.shape[:2]
    total_pix = float(mask_bin.size)
    hole_pix = float(np.sum(mask_bin == 1))
    area_ratio = hole_pix / total_pix if total_pix > 0 else 0.0

    ys, xs = np.where(mask_bin == 1)
    if len(xs) == 0:
        return {
            "mask_area_ratio": 0.0,
            "mask_location": "none",
            "mask_fragments": 0,
            "mask_bbox": None,
        }

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    if 0.33 * w < cx < 0.66 * w and 0.33 * h < cy < 0.66 * h:
        location = "center"
    else:
        location = "edge_or_corner"

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    fragments = max(0, num_labels - 1)

    return {
        "mask_area_ratio": area_ratio,
        "mask_location": location,
        "mask_fragments": fragments,
        "mask_bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
    }


# -----------------------------------------------------------------------------
# Decision Router Engine
# -----------------------------------------------------------------------------

def classify_and_route(
    image: Union[np.ndarray, Image.Image],
    mask: Optional[Union[np.ndarray, Image.Image]] = None,
) -> Dict[str, Any]:
    """
    Extracts heuristic features and executes the audited routing decision tree.
    """
    img_rgb = ensure_512_image(image)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    if mask is None:
        mask_np = np.zeros((512, 512), dtype=np.uint8)
    else:
        if isinstance(mask, Image.Image):
            mask_np = np.array(mask.convert("L").resize((512, 512), Image.Resampling.NEAREST))
        else:
            mask_np = mask
            if mask_np.shape[:2] != (512, 512):
                mask_np = cv2.resize(mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_np = np.where(mask_np >= 128, 255, 0).astype(np.uint8)

    faces = detect_faces(img_rgb)
    lap_var = compute_laplacian_variance(img_gray)
    edge_dens = compute_edge_density(img_gray)
    text_found = detect_text(img_rgb)
    mask_stats = analyze_mask_geometry(mask_np)

    features = {
        "faces_detected": faces,
        "laplacian_variance": round(lap_var, 2),
        "edge_density": round(edge_dens, 4),
        "text_present": text_found,
        "brightness_mean": round(float(np.mean(img_gray)), 2),
        "saturation_mean": round(float(np.mean(hsv[:, :, 1])), 2),
        "mask_area_ratio": round(mask_stats["mask_area_ratio"], 4),
        "mask_area_pct": round(mask_stats["mask_area_ratio"] * 100.0, 2),
        "mask_location": mask_stats["mask_location"],
        "mask_fragments": mask_stats["mask_fragments"],
        "mask_bbox": mask_stats["mask_bbox"],
    }

    area_ratio = mask_stats["mask_area_ratio"]
    fragments = mask_stats["mask_fragments"]

    # Rule 1: Facial Structure Priority
    if faces > 0 and area_ratio <= 0.25:
        model = "migan"
        reason = (
            f"High facial geometry detected ({faces} face{'s' if faces > 1 else ''}) with localized mask "
            f"({features['mask_area_pct']}% area). Routed to MIGAN: specialized depthwise facial GAN architecture, "
            f"sub-second interactive latency (216 ms), lowest active energy (0.62 Joules)."
        )
        rule_triggered = "RULE_FACE_PORTRAIT"

    # Rule 2: Large Corruption Void / Extreme Fragmentation
    elif area_ratio > 0.25 or fragments >= 3 or (mask_stats["mask_location"] == "edge_or_corner" and area_ratio > 0.15):
        model = "lama"
        reason = (
            f"Large corruption void ({features['mask_area_pct']}% > 25.0%) or high fragmentation ({fragments} components). "
            f"Routed to LaMa Dilated: Fast Fourier Convolutions (FFC) provide global receptive field, preventing structural "
            f"collapse across wide missing regions (Hole PSNR: 14.70 dB in Tier 3 stress tests)."
        )
        rule_triggered = "RULE_LARGE_VOID_FFC"

    # Rule 3: High Texture / Geometric Edges / OCR Text
    elif edge_dens > 0.08 or lap_var > 500.0 or text_found:
        model = "aotgan"
        reason = (
            f"Dense geometric boundaries / high texture detected (edge density: {edge_dens:.4f}, Laplacian variance: {lap_var:.1f}). "
            f"Routed to AOT-GAN: Aggregated Contextual Transformations maintain superior boundary continuity and fine pattern "
            f"synthesis (benchmark highest SSIM 0.9214, lowest LPIPS 0.1058)."
        )
        rule_triggered = "RULE_HIGH_TEXTURE_AOT"

    # Rule 4: Baseline / Smooth Surface Default
    else:
        model = "migan"
        reason = (
            f"Low-frequency surface / smooth textures (edge density: {edge_dens:.4f}, Laplacian: {lap_var:.1f}) with moderate mask "
            f"({features['mask_area_pct']}%). Routed to MIGAN: optimal edge power efficiency (0.62 Joules/image, 216 ms latency, "
            f"minimal thermal rise +9.6°C)."
        )
        rule_triggered = "RULE_SMOOTH_DEFAULT"

    return {
        "recommended_model": model,
        "rule_triggered": rule_triggered,
        "justification": reason,
        "features": features,
        "telemetry": BENCHMARK_PROFILES[model],
        "all_profiles": BENCHMARK_PROFILES,
    }


# -----------------------------------------------------------------------------
# On-Device SNPE & SD Execution Harness
# -----------------------------------------------------------------------------

def check_device_status() -> Tuple[bool, str]:
    try:
        res = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=3)
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        device_lines = [l for l in lines[1:] if "device" in l and not l.startswith("*")]
        if device_lines:
            serial = device_lines[0].split()[0]
            prop = subprocess.run(["adb", "-s", serial, "shell", "getprop", "ro.product.model"], capture_output=True, text=True, timeout=3)
            model = prop.stdout.strip() or "Qualcomm Device"
            return True, f"{model} ({serial})"
        return False, "No ADB device connected"
    except Exception as e:
        return False, f"ADB Error: {e}"


def run_live_snpe_inference(
    image: Union[np.ndarray, Image.Image],
    mask: np.ndarray,
    model_key: str = "auto",
    device_base: str = "/data/local/tmp/lama",
    sd_base: str = "/data/local/tmp/sd_runtime",
    prompt: str = "A high quality detailed cinematic photo",
    allow_cpu_fallback: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Executes live inference on Snapdragon 8 Elite NPU via SNPE net-run or SD RePaint runner.
    """
    img_512 = ensure_512_image(image)
    standard_mask = np.where(mask >= 128, 255, 0).astype(np.uint8)

    route_info = classify_and_route(img_512, standard_mask)
    selected_model = route_info["recommended_model"] if model_key == "auto" else model_key.lower()
    
    # Normalize model key
    if "sd" in selected_model or "diffusion" in selected_model or "repaint" in selected_model:
        selected_model = "sd"
    elif "migan" in selected_model:
        selected_model = "migan"
    elif "lama" in selected_model:
        selected_model = "lama"
    elif "aot" in selected_model:
        selected_model = "aotgan"

    profile = BENCHMARK_PROFILES.get(selected_model, BENCHMARK_PROFILES["migan"])
    is_connected, dev_name = check_device_status()
    t_start = time.perf_counter()

    if not is_connected:
        if not allow_cpu_fallback:
            raise RuntimeError(f"Qualcomm Snapdragon Device Not Connected via ADB ({dev_name}). Ensure board is plugged in and authorized.")

    if is_connected:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # -------------------------------------------------------------
                # 1. Stable Diffusion 1.5 RePaint Path
                # -------------------------------------------------------------
                if selected_model == "sd":
                    img_arr = (img_512.astype(np.float32) / 255.0)  # (512, 512, 3)
                    # SD expects 1.0 at hole, 0.0 at keep:
                    standard_mask_tensor = (standard_mask.astype(np.float32) / 255.0)[..., np.newaxis]  # (512, 512, 1)

                    local_img = os.path.join(tmpdir, "image.raw")
                    local_mask = os.path.join(tmpdir, "mask.raw")
                    img_arr.tofile(local_img)
                    standard_mask_tensor.tofile(local_mask)

                    # Push to device sd_runtime
                    subprocess.run(["adb", "push", local_img, f"{sd_base}/image.raw"], check=True, timeout=10)
                    subprocess.run(["adb", "push", local_mask, f"{sd_base}/mask.raw"], check=True, timeout=10)

                    # Execute SD runner
                    sd_cmd = (
                        f"cd {sd_base} && "
                        f"export LD_LIBRARY_PATH={sd_base}:$LD_LIBRARY_PATH && "
                        f"export ADSP_LIBRARY_PATH='{sd_base};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && "
                        f"./sd_qidk_runner_encoder \"{prompt}\""
                    )
                    t0 = time.perf_counter()
                    subprocess.run(["adb", "shell", sd_cmd], check=True, timeout=180)
                    sd_ms = (time.perf_counter() - t0) * 1000.0

                    local_out = os.path.join(tmpdir, "sd_output.png")
                    subprocess.run(["adb", "pull", f"{sd_base}/sd_output.png", local_out], check=True, timeout=15)

                    if os.path.isfile(local_out):
                        sd_img = np.array(Image.open(local_out).convert("RGB"))
                        sd_final = composite_inpaint_result(img_512, sd_img, standard_mask, feather=True)
                        return sd_final, {
                            "execution_mode": "QUALCOMM_HEXAGON_NPU_LIVE (SD 1.5 RePaint)",
                            "device": dev_name,
                            "model_executed": profile["name"],
                            "snpe_latency_ms": round(sd_ms, 2),
                            "total_pipeline_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
                            "energy_j": profile["energy_j"],
                            "power_w": profile["power_w"],
                            "thermal_rise_c": profile["thermal_rise_c"],
                            "route_info": route_info,
                        }

                # -------------------------------------------------------------
                # 2. Feed-Forward GANs (MIGAN, LaMa Dilated, AOT-GAN)
                # -------------------------------------------------------------
                else:
                    img_raw, mask_raw = mask_to_raw_tensors(img_512, standard_mask, model=selected_model)
                    local_img_raw = os.path.join(tmpdir, "live_img.raw")
                    local_mask_raw = os.path.join(tmpdir, "live_mask.raw")
                    img_raw.tofile(local_img_raw)
                    mask_raw.tofile(local_mask_raw)

                    dev_input = f"{device_base}/input"
                    subprocess.run(["adb", "shell", f"mkdir -p {dev_input}"], check=True, timeout=5)
                    subprocess.run(["adb", "push", local_img_raw, f"{dev_input}/live_img.raw"], check=True, timeout=10)
                    subprocess.run(["adb", "push", local_mask_raw, f"{dev_input}/live_mask.raw"], check=True, timeout=10)

                    input_line = f"image:={dev_input}/live_img.raw mask:={dev_input}/live_mask.raw"
                    create_list_cmd = f"echo '{input_line}' > {device_base}/live_input.txt"
                    subprocess.run(["adb", "shell", create_list_cmd], check=True, timeout=5)

                    out_dir_dev = f"{device_base}/host_live_output"
                    clean_cmd = f"rm -rf {out_dir_dev} && mkdir -p {out_dir_dev}"
                    subprocess.run(["adb", "shell", clean_cmd], check=True, timeout=5)

                    env_setup = (
                        f"export LD_LIBRARY_PATH={device_base}/lib:{device_base}; "
                        f"export ADSP_LIBRARY_PATH='{device_base}/dsp/lib;{device_base}/dsp;/dsp'; "
                        f"export PATH=$PATH:{device_base}/bin:{device_base}; "
                        f"cd {device_base}"
                    )
                    snpe_run = f"{device_base}/snpe-net-run --container {profile['dlc']} --input_list live_input.txt --output_dir host_live_output {profile['runtime_flag']}"
                    full_cmd = f"{env_setup} && {snpe_run}"

                    t_snpe_0 = time.perf_counter()
                    subprocess.run(["adb", "shell", full_cmd], capture_output=True, text=True, timeout=25)
                    snpe_ms = (time.perf_counter() - t_snpe_0) * 1000.0

                    local_out_dir = os.path.join(tmpdir, "host_live_output")
                    os.makedirs(local_out_dir, exist_ok=True)
                    subprocess.run(["adb", "pull", f"{out_dir_dev}/.", local_out_dir], check=True, timeout=10)

                    candidates = ["output_0.raw", "painted_image.raw"]
                    raw_found = None
                    for root_dir, _, files in os.walk(local_out_dir):
                        for cand in candidates:
                            if cand in files:
                                raw_found = os.path.join(root_dir, cand)
                                break
                        if raw_found:
                            break

                    if raw_found and os.path.isfile(raw_found):
                        raw_arr = np.fromfile(raw_found, dtype=np.float32)
                        if len(raw_arr) == 512 * 512 * 3:
                            raw_max = float(raw_arr.max())
                            raw_min = float(raw_arr.min())
                            if raw_max <= 1.05:
                                if raw_min < -0.1:
                                    raw_arr = (raw_arr + 1.0) * 0.5 * 255.0
                                else:
                                    raw_arr = raw_arr * 255.0
                            raw_u8 = np.clip(raw_arr, 0, 255).astype(np.uint8).reshape((512, 512, 3))
                            final_u8 = composite_inpaint_result(img_512, raw_u8, standard_mask, feather=True)
                            total_ms = (time.perf_counter() - t_start) * 1000.0
                            return final_u8, {
                                "execution_mode": "QUALCOMM_HEXAGON_NPU_LIVE",
                                "device": dev_name,
                                "model_executed": profile["name"],
                                "snpe_latency_ms": round(snpe_ms, 2),
                                "total_pipeline_ms": round(total_ms, 2),
                                "energy_j": profile["energy_j"],
                                "power_w": profile["power_w"],
                                "thermal_rise_c": profile["thermal_rise_c"],
                                "route_info": route_info,
                            }
        except Exception as e:
            if not allow_cpu_fallback:
                raise RuntimeError(f"Qualcomm NPU Hardware Execution Failed: {e}")
            print(f"⚠️ Live inference warning: {e}. Falling back to simulation.")

    if not allow_cpu_fallback:
        raise RuntimeError(f"Qualcomm NPU Failed: Inference completed without returning a valid output tensor for model {profile['name']}.")

    # High-fidelity CPU Fallback
    inpaint_mode = cv2.INPAINT_TELEA if selected_model == "migan" else cv2.INPAINT_NS
    inpainted = cv2.inpaint(img_512, standard_mask, inpaintRadius=5, flags=inpaint_mode)
    final_u8 = composite_inpaint_result(img_512, inpainted, standard_mask, feather=False)
    total_ms = (time.perf_counter() - t_start) * 1000.0

    return final_u8, {
        "execution_mode": "CPU_SIMULATED_TELEMETRY",
        "device": f"{dev_name} (Simulation Fallback)",
        "model_executed": profile["name"],
        "snpe_latency_ms": profile["latency_ms"],
        "total_pipeline_ms": round(total_ms, 2),
        "energy_j": profile["energy_j"],
        "power_w": profile["power_w"],
        "thermal_rise_c": profile["thermal_rise_c"],
        "route_info": route_info,
    }
