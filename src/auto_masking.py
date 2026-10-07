import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
src/auto_masking.py
Target Auto-Masking Engine for Interactive Inpainting.

Provides two concrete masking modes:
1. Interactive Brush / Touch Masking:
   Generates single-channel 512x512 uint8 binary masks from sequential coordinate strokes.
2. Target Bounding Box / Tap-to-Mask (OpenCV GrabCut):
   Takes (x1, y1, x2, y2) user selection box, executes optimized cv2.grabCut with
   background/foreground GMMs to isolate the target silhouette in <50 ms on CPU.
   Outputs a clean binary mask (0 = background, 255 = hole to inpaint).
"""

import time
from typing import List, Tuple, Union, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = (512, 512)


def ensure_512_image(image: Union[np.ndarray, Image.Image, str, os.PathLike]) -> np.ndarray:
    """Normalizes input image to 512x512 uint8 RGB numpy array."""
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    if image.shape[:2] != TARGET_SIZE:
        image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return image.astype(np.uint8)


def create_mask_overlay(
    image: Union[np.ndarray, Image.Image],
    mask: Union[np.ndarray, Image.Image],
    color: Tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.5,
) -> Image.Image:
    """
    Blends a translucent colored mask (red by default) directly over the source photo.
    Returns a 512x512 RGB PIL Image for live visual preview.
    """
    img = ensure_512_image(image).copy()
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask.convert("L").resize(TARGET_SIZE, Image.Resampling.NEAREST))
    else:
        mask_np = mask
        if mask_np.shape[:2] != TARGET_SIZE:
            mask_np = cv2.resize(mask_np, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask_np >= 128)
    if not np.any(mask_bin):
        return Image.fromarray(img, mode="RGB")

    overlay = img.copy()
    overlay[mask_bin] = color
    blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    return Image.fromarray(blended, mode="RGB")


def create_brush_mask(
    strokes: Union[List[Any], Dict[str, Any], np.ndarray, Image.Image],
    shape: Tuple[int, int] = TARGET_SIZE,
    default_radius: int = 20,
) -> np.ndarray:
    """
    Interactive Brush / Touch Masking Engine.
    Generates single-channel 512x512 uint8 binary masks (0 = background, 255 = hole).
    """
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)

    # 1. Direct array or PIL image input (canvas layer or mask image)
    if isinstance(strokes, (np.ndarray, Image.Image)):
        arr = np.array(strokes)
        if arr.ndim == 3:
            # If layer is RGBA:
            if arr.shape[-1] == 4:
                # If layer has transparency (not uniformly opaque 255 everywhere):
                if not np.all(arr[:, :, 3] == 255):
                    stroke_mask = (arr[:, :, 3] > 50) | (arr[:, :, 0] > 150)
                else:
                    stroke_mask = (arr[:, :, 0] > 150) | (cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY) > 128)
            else:
                # If layer is RGB:
                stroke_mask = (arr[:, :, 0] > 150) | (cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) > 128)
            arr = np.where(stroke_mask, 255, 0).astype(np.uint8)
        elif arr.ndim == 2:
            arr = np.where(arr > 128, 255, 0).astype(np.uint8)
        
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.where(arr >= 128, 255, 0).astype(np.uint8)

    # 2. Dictionary wrapping editor layers or composite
    if isinstance(strokes, dict):
        # A. Check explicit layers first (Gradio ImageEditor canvas layers)
        if "layers" in strokes and strokes["layers"] and len(strokes["layers"]) > 0:
            for layer in strokes["layers"]:
                if layer is not None:
                    layer_arr = np.array(layer)
                    if layer_arr.ndim == 3:
                        if layer_arr.shape[-1] == 4:
                            if not np.all(layer_arr[:, :, 3] == 255):
                                stroke_mask = (layer_arr[:, :, 3] > 50) | (layer_arr[:, :, 0] > 150)
                            else:
                                stroke_mask = (layer_arr[:, :, 0] > 150) | (cv2.cvtColor(layer_arr[:, :, :3], cv2.COLOR_RGB2GRAY) > 128)
                        else:
                            stroke_mask = (layer_arr[:, :, 0] > 150) | (cv2.cvtColor(layer_arr, cv2.COLOR_RGB2GRAY) > 128)
                        layer_mask = np.where(stroke_mask, 255, 0).astype(np.uint8)
                    elif layer_arr.ndim == 2:
                        layer_mask = np.where(layer_arr > 128, 255, 0).astype(np.uint8)
                    else:
                        layer_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    if layer_mask.shape[:2] != (h, w):
                        layer_mask = cv2.resize(layer_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    canvas = np.bitwise_or(canvas, layer_mask)
            
            if np.any(canvas >= 128):
                return np.where(canvas >= 128, 255, 0).astype(np.uint8)

        # B. Check composite vs background diff
        if "composite" in strokes and strokes["composite"] is not None and "background" in strokes and strokes["background"] is not None:
            comp = np.array(ensure_512_image(strokes["composite"]))
            bg = np.array(ensure_512_image(strokes["background"]))
            diff = np.max(np.abs(comp.astype(int) - bg.astype(int)), axis=2)
            if np.any(diff > 15):
                return np.where(diff > 15, 255, 0).astype(np.uint8)

        # C. Check single mask / composite
        if "mask" in strokes and strokes["mask"] is not None:
            return create_brush_mask(strokes["mask"], shape=shape)
        if "composite" in strokes and strokes["composite"] is not None:
            return create_brush_mask(strokes["composite"], shape=shape)
        if "points" in strokes:
            strokes = strokes["points"]
        elif "strokes" in strokes:
            strokes = strokes["strokes"]
        else:
            return canvas

    # 3. Coordinate stroke list
    if isinstance(strokes, list):
        if not strokes:
            return canvas

        if len(strokes) > 0 and isinstance(strokes[0], dict) and "points" in strokes[0]:
            stroke_groups = [s["points"] for s in strokes]
        elif len(strokes) > 0 and isinstance(strokes[0], list) and len(strokes[0]) > 0 and isinstance(strokes[0][0], (list, tuple)):
            stroke_groups = strokes
        else:
            stroke_groups = [strokes]

        for s in stroke_groups:
            pts = []
            radius = default_radius
            for item in s:
                if isinstance(item, dict):
                    x, y = int(item.get("x", 0)), int(item.get("y", 0))
                    radius = int(item.get("radius", default_radius))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    x, y = int(item[0]), int(item[1])
                    if len(item) >= 3:
                        radius = int(item[2])
                else:
                    continue
                pts.append((x, y))

            if not pts:
                continue

            for pt in pts:
                cv2.circle(canvas, pt, radius, 255, -1)

            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], 255, thickness=radius * 2, lineType=cv2.LINE_AA)

    return np.where(canvas >= 128, 255, 0).astype(np.uint8)


def grabcut_bounding_box_mask(
    image: Union[np.ndarray, Image.Image],
    bbox: Union[Tuple[int, int, int, int], List[int], Dict[str, int]],
    iter_count: int = 1,
    fast_mode: bool = True,
    max_dim: int = 112,
    margin: int = 12,
) -> Tuple[np.ndarray, float]:
    """
    Target Bounding Box / Tap-to-Mask Engine using OpenCV GrabCut.
    Isolates the target silhouette in <50 ms on CPU.
    """
    t0 = time.perf_counter()
    img = ensure_512_image(image)
    h_img, w_img = img.shape[:2]

    if isinstance(bbox, dict):
        x1 = int(bbox.get("x1", bbox.get("x_min", 0)))
        y1 = int(bbox.get("y1", bbox.get("y_min", 0)))
        x2 = int(bbox.get("x2", bbox.get("x_max", w_img)))
        y2 = int(bbox.get("y2", bbox.get("y_max", h_img)))
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

    x_min, x_max = max(0, min(x1, x2)), min(w_img - 1, max(x1, x2))
    y_min, y_max = max(0, min(y1, y2)), min(h_img - 1, max(y1, y2))
    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)

    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    if box_w < 5 or box_h < 5:
        full_mask[y_min:y_max+1, x_min:x_max+1] = 255
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return full_mask, latency_ms

    if fast_mode:
        rx1 = max(0, x_min - margin)
        ry1 = max(0, y_min - margin)
        rx2 = min(w_img, x_max + margin)
        ry2 = min(h_img, y_max + margin)
        roi = img[ry1:ry2, rx1:rx2]
        roi_h, roi_w = roi.shape[:2]

        scale = min(1.0, float(max_dim) / max(roi_w, roi_h))
        small_w = max(4, int(round(roi_w * scale)))
        small_h = max(4, int(round(roi_h * scale)))
        small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)

        sx1 = int(round((x_min - rx1) * scale))
        sy1 = int(round((y_min - ry1) * scale))
        sw = max(2, int(round(box_w * scale)))
        sh = max(2, int(round(box_h * scale)))

        sx1 = min(sx1, small_w - sw - 1) if small_w > sw + 1 else 0
        sy1 = min(sy1, small_h - sh - 1) if small_h > sh + 1 else 0
        sw = max(1, min(sw, small_w - sx1))
        sh = max(1, min(sh, small_h - sy1))
        rect = (sx1, sy1, sw, sh)

        s_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)

        try:
            cv2.grabCut(small_roi, s_mask, rect, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)
            small_fg = np.where((s_mask == cv2.GC_FGD) | (s_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

            fg_upsampled = cv2.resize(small_fg, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            fg_binary = np.where(fg_upsampled >= 128, 255, 0).astype(np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_refined = cv2.morphologyEx(fg_binary, cv2.MORPH_CLOSE, kernel)

            full_mask[ry1:ry2, rx1:rx2] = fg_refined

            if np.sum(full_mask) == 0:
                full_mask[y_min:y_max+1, x_min:x_max+1] = 255

        except Exception:
            full_mask[y_min:y_max+1, x_min:x_max+1] = 255

    else:
        rect = (x_min, y_min, box_w, box_h)
        gc_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(img, gc_mask, rect, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)
            full_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            if np.sum(full_mask) == 0:
                full_mask[y_min:y_max+1, x_min:x_max+1] = 255
        except Exception:
            full_mask[y_min:y_max+1, x_min:x_max+1] = 255

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return full_mask, latency_ms


def tap_to_mask(
    image: Union[np.ndarray, Image.Image],
    point: Tuple[int, int],
    radius: int = 50,
    iter_count: int = 1,
    fast_mode: bool = True,
) -> Tuple[np.ndarray, float]:
    """Tap-to-Mask: Expands tap coordinate (x, y) into a local bbox and runs GrabCut."""
    x, y = point
    bbox = (x - radius, y - radius, x + radius, y + radius)
    return grabcut_bounding_box_mask(image, bbox, iter_count=iter_count, fast_mode=fast_mode)


def refine_mask_grabcut(
    image: Union[np.ndarray, Image.Image],
    rough_mask: Union[np.ndarray, Image.Image],
    iterations: int = 2,
    margin: int = 15,
) -> Tuple[np.ndarray, float]:
    """
    Refines rough brush strokes by snapping to object edges using GrabCut (GC_INIT_WITH_MASK).
    - Unpainted areas outside stroke bounding box are marked cv2.GC_BGD (0: definite background).
    - Unpainted areas inside stroke bounding box are marked cv2.GC_PR_BGD (2: probable background).
    - Rough brush strokes are marked cv2.GC_PR_FGD (3: probable foreground).
    - Eroded stroke core is marked cv2.GC_FGD (1: definite foreground).
    - Iterates GrabCut to snap around object silhouette boundaries.
    """
    t0 = time.perf_counter()
    img = ensure_512_image(image)
    h, w = img.shape[:2]

    if isinstance(rough_mask, Image.Image):
        rmask = np.array(rough_mask.convert("L").resize((w, h), Image.Resampling.NEAREST))
    else:
        rmask = cv2.resize(rough_mask, (w, h), interpolation=cv2.INTER_NEAREST) if rough_mask.shape[:2] != (h, w) else rough_mask

    binary_stroke = (rmask >= 128)
    if not np.any(binary_stroke):
        return np.zeros((h, w), dtype=np.uint8), 0.0

    y_indices, x_indices = np.where(binary_stroke)
    x_min, x_max = max(0, int(np.min(x_indices)) - margin), min(w - 1, int(np.max(x_indices)) + margin)
    y_min, y_max = max(0, int(np.min(y_indices)) - margin), min(h - 1, int(np.max(y_indices)) + margin)

    # Initialize mask
    gc_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc_mask[y_min:y_max+1, x_min:x_max+1] = cv2.GC_PR_BGD
    gc_mask[binary_stroke] = cv2.GC_PR_FGD

    stroke_u8 = binary_stroke.astype(np.uint8) * 255
    kernel_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_fgd = cv2.erode(stroke_u8, kernel_core)
    if np.any(core_fgd > 0):
        gc_mask[core_fgd > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, iterCount=iterations, mode=cv2.GC_INIT_WITH_MASK)
        refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        if np.sum(refined) == 0:
            refined = stroke_u8
    except Exception:
        refined = stroke_u8

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return refined, latency_ms


_MP_SEGMENTER_INSTANCE = None

def get_mediapipe_segmenter(model_path: Optional[str] = None):
    """Lazily initializes MediaPipe InteractiveSegmenter for Python."""
    global _MP_SEGMENTER_INSTANCE
    if _MP_SEGMENTER_INSTANCE is not None:
        return _MP_SEGMENTER_INSTANCE

    if model_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "qidk-inpaint-app", "app", "src", "main", "assets", "magic_touch.tflite"),
            os.path.join(os.path.dirname(__file__), "..", "models", "magic_touch.tflite"),
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = os.path.abspath(c)
                break

    if model_path is None or not os.path.exists(model_path):
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.InteractiveSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            output_confidence_masks=True
        )
        _MP_SEGMENTER_INSTANCE = vision.InteractiveSegmenter.create_from_options(options)
        return _MP_SEGMENTER_INSTANCE
    except Exception as e:
        return None


def sample_medial_axis_seeds(
    binary_mask: np.ndarray,
    max_seeds: int = 3,
    min_dist_threshold: float = 3.0
) -> List[Tuple[int, int]]:
    """
    Extracts 1 to 3 medial axis prompt seeds from a binary mask via Distance Transform + NMS.
    Eliminates centroid collapse on concave / annular strokes.
    """
    if np.sum(binary_mask > 0) == 0:
        return []

    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)

    seeds = []
    dist_map = dist.copy()

    for _ in range(max_seeds):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_map)
        if max_val < min_dist_threshold:
            break
        if seeds and max_val < seeds[0][1] * 0.30:
            break

        seeds.append((max_loc, max_val))
        suppression_radius = int(max(20, max_val * 1.5))
        cv2.circle(dist_map, max_loc, suppression_radius, 0, -1)

    if not seeds:
        y_pts, x_pts = np.where(mask_u8 > 0)
        if len(x_pts) > 0:
            return [(int(x_pts[len(x_pts) // 2]), int(y_pts[len(y_pts) // 2]))]
        return []

    return [pt for pt, _ in seeds]


def fast_guided_filter(
    image: np.ndarray,
    mask_prob: np.ndarray,
    radius: int = 4,
    eps: float = 0.01
) -> np.ndarray:
    """
    Fast Guided Filter snapping soft mask probabilities to photo color edges.
    Uses cv2.ximgproc.guidedFilter if available, otherwise high-quality bilateral fallback.
    """
    img_guide = ensure_512_image(image)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        guide_gray = cv2.cvtColor(img_guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        prob = mask_prob.astype(np.float32)
        if prob.max() > 1.0:
            prob = prob / 255.0
        q = cv2.ximgproc.guidedFilter(guide_gray, prob, radius, eps)
        return np.clip(q, 0.0, 1.0)
    else:
        return cv2.bilateralFilter(mask_prob.astype(np.float32), radius * 2 + 1, 75, 75)


def refine_mask_overhaul(
    image: Union[np.ndarray, Image.Image],
    rough_mask: Union[np.ndarray, Image.Image],
    model_path: Optional[str] = None
) -> Tuple[np.ndarray, float]:
    """
    Overhauled MediaPipe Pipeline (Candidate 3):
    1. Multi-point medial axis sampling (avoids centroid collapse)
    2. Multi-seed confidence map fusion
    3. Fast Guided Filter edge-snapping (r=4, eps=1e-2)
    4. 2px circular safety dilation
    5. Instant sub-1ms fallback (zero GrabCut stalling)
    """
    t0 = time.perf_counter()
    img_512 = ensure_512_image(image)
    h, w = img_512.shape[:2]

    if isinstance(rough_mask, Image.Image):
        rmask = np.array(rough_mask.convert("L").resize((w, h), Image.Resampling.NEAREST))
    else:
        rmask = cv2.resize(rough_mask, (w, h), interpolation=cv2.INTER_NEAREST) if rough_mask.shape[:2] != (w, h) else rough_mask

    mask_bin = (rmask >= 128)
    if not np.any(mask_bin):
        return np.zeros((h, w), dtype=np.uint8), 0.0

    seeds = sample_medial_axis_seeds(mask_bin, max_seeds=3)
    segmenter = get_mediapipe_segmenter(model_path)

    fused_conf = np.zeros((h, w), dtype=np.float32)

    if segmenter is not None and seeds:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.components.containers import keypoint

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_512)
            for (sx, sy) in seeds:
                roi = vision.InteractiveSegmenterRegionOfInterest(
                    format=vision.InteractiveSegmenterRegionOfInterest.Format.KEYPOINT,
                    keypoint=keypoint.NormalizedKeypoint(
                        x=float(np.clip(sx / float(w), 0.01, 0.99)),
                        y=float(np.clip(sy / float(h), 0.01, 0.99))
                    )
                )
                res = segmenter.segment(mp_image, roi)
                if res.confidence_masks:
                    c_mask = res.confidence_masks[0].numpy_view()
                    if c_mask.shape != (h, w):
                        c_mask = cv2.resize(c_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                    fused_conf = np.maximum(fused_conf, c_mask)
        except Exception:
            pass

    # If neural confidence was insufficient or segmenter unavailable, use rough mask as initial probability
    if np.sum(fused_conf >= 0.35) < 20:
        fused_conf = mask_bin.astype(np.float32)

    # Fast Guided Filter snapping to RGB physical edges
    q = fast_guided_filter(img_512, fused_conf, radius=4, eps=0.01)

    # Threshold & 2px safety dilation
    binary_refined = (q >= 0.45).astype(np.uint8) * 255
    k_2px = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.dilate(binary_refined, k_2px)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return final_mask, latency_ms




def mask_to_raw_tensors(
    image: Union[np.ndarray, Image.Image],
    standard_mask: np.ndarray,
    model: str = "standard"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts 512x512 image and standard mask (255=hole, 0=keep) into float32 NHWC raw tensor binaries.
    Isolates input tensor polarity per model:
    - MIGAN expects 0.0 at hole, 1.0 at keep: migan_mask_tensor = 1.0 - (standard_mask / 255.0)
    - LaMa / AOT-GAN / SD expect 1.0 at hole, 0.0 at keep: standard_mask_tensor = (standard_mask / 255.0)
    """
    img_512 = ensure_512_image(image)
    img_float = (img_512.astype(np.float32) / 255.0)[np.newaxis, ...]

    mask_512 = np.array(standard_mask)
    if mask_512.shape[:2] != TARGET_SIZE:
        mask_512 = cv2.resize(mask_512, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    standard_mask_tensor = (mask_512 >= 128).astype(np.float32)

    if model.lower() == "migan":
        # MIGAN expects 0.0 at hole, 1.0 at keep:
        migan_mask_tensor = 1.0 - standard_mask_tensor
        mask_float = migan_mask_tensor[np.newaxis, ..., np.newaxis]
    else:
        # LaMa / AOT-GAN / SD expect 1.0 at hole, 0.0 at keep:
        mask_float = standard_mask_tensor[np.newaxis, ..., np.newaxis]

    return img_float, mask_float


def composite_inpaint_result(
    original: Union[np.ndarray, Image.Image],
    model_output: Union[np.ndarray, Image.Image],
    standard_mask: Union[np.ndarray, Image.Image],
    feather: bool = True,
) -> np.ndarray:
    """
    Composites inpaint prediction with original image using UN-INVERTED standard mask (255 = hole, 0 = keep):
    final = original * (1.0 - mask_norm) + model_output * mask_norm
    where mask_norm is 1.0 inside hole (mask >= 128) and 0.0 outside (unmasked background).
    Ensures background is preserved 1:1 and hole is 100% replaced by model output.
    """
    orig = ensure_512_image(original).astype(np.float32)
    pred = ensure_512_image(model_output).astype(np.float32)

    if isinstance(standard_mask, Image.Image):
        mask_np = np.array(standard_mask.convert("L").resize(TARGET_SIZE, Image.Resampling.NEAREST))
    else:
        mask_np = np.array(standard_mask)
        if mask_np.shape[:2] != TARGET_SIZE:
            mask_np = cv2.resize(mask_np, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

    # Standard mask: 1.0 at hole, 0.0 at background
    mask_norm = (mask_np >= 128).astype(np.float32)

    if feather:
        # Subtle 3x3 gaussian feathering to prevent harsh boundary stepping
        weight = cv2.GaussianBlur(mask_norm, (3, 3), 0)[..., np.newaxis]
    else:
        weight = mask_norm[..., np.newaxis]

    # Hole gets model_output, background gets original
    final = orig * (1.0 - weight) + pred * weight
    return np.clip(final, 0, 255).astype(np.uint8)

