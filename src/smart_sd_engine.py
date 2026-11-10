#!/usr/bin/env python3
"""
src/smart_sd_engine.py
================================================================================
⚡ TWO-STAGE "SMART STABLE DIFFUSION" CONTEXTUAL PROMPT ENGINE
================================================================================
Architecture:
  Stage 1 (Vision Context Extractor):
    - Ingests 512x512 image and binary mask.
    - Crops the 16-32px context ring around the void.
    - Detects:
      * Scene Category (Portrait, Architectural, Nature/Landscape, Street/Urban, Interior)
      * Ambient Illumination (Color temperature, exposure dynamic range, key lighting direction)
      * Dominant Surface Textures (High-frequency roughness, smooth gradient, repetitive pattern)
      * Semantic Foreground Class
  Stage 2 (Prompt Synthesizer):
    - Combines vision context with user's optional input.
    - If user prompt is empty, synthesizes a pure contextual hallucination prompt.
    - Formulates hyper-specific photographic descriptors:
      * Material textures (e.g. 'fine porous skin texture', 'rough weathered granite')
      * Lighting angle (e.g. 'natural 45-degree directional golden hour sidelight')
      * Camera aperture & lens (e.g. '85mm f/1.4 prime lens, tack sharp depth of field, 8k photographic fidelity')
    - Synthesizes robust negative prompt anchors targeting generative artifacts.
================================================================================
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

class SmartSDContextEngine:
    def __init__(self):
        # Anchor keyword registries
        self.scene_categories = {
            "portrait": {
                "positive": "photorealistic human portrait, natural skin pores, subcutaneous scattering, fine facial hair, realistic eye reflections, 85mm portrait lens, f/1.8, soft directional studio lighting",
                "negative": "deformed eyes, cross-eyed, plastic doll skin, cartoonish, 3d render, extra limbs, bad anatomy, mutated fingers, text, watermark"
            },
            "architecture": {
                "positive": "crisp architectural photography, straight structural lines, authentic masonry textures, mortar details, natural ambient daylight, perspective corrected, tilt-shift lens, ultra-sharp 8k resolution",
                "negative": "warped walls, bent pillars, melted bricks, blurry lines, crooked perspective, cartoon, low-res"
            },
            "nature": {
                "positive": "breathtaking natural landscape, intricate foliage sub-branches, organic moss textures, hyper-detailed vegetation, natural sunlight dappling, high dynamic range, tack-sharp landscape focus",
                "negative": "blurry leaves, flat green smear, oversaturated neon, repetitive tiles, floating artifacts"
            },
            "street": {
                "positive": "urban street photography, realistic weathered asphalt aggregate, curbside concrete grain, natural outdoor overcast illumination, authentic cobblestone surface roughness, 35mm documentary lens",
                "negative": "smudged asphalt, floating debris, distorted pavement, blurred road, cartoon, low quality"
            },
            "indoor": {
                "positive": "modern interior architectural photography, clean surface reflections, wood grain texture, polished floor tile specular highlights, soft diffused warm indoor lighting, f/4 aperture",
                "negative": "distorted furniture, noisy walls, flat gray shading, glitch, blur, artifact"
            }
        }
        
    def extract_vision_context(
        self,
        image_rgb: np.ndarray,
        mask_binary: np.ndarray
    ) -> Dict[str, Any]:
        """
        Stage 1: Extract mathematical vision features from the context ring surrounding the void.
        """
        h, w = image_rgb.shape[:2]
        hole_mask = (mask_binary > 128).astype(np.uint8)
        
        # 1. Extract 24px dilated context ring around the void
        kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))
        dilated = cv2.dilate(hole_mask, kernel_ring)
        context_ring = cv2.bitwise_and(dilated, (1 - hole_mask))
        
        if np.sum(context_ring) < 50:
            # Fallback to entire unmasked image if void is tiny or occupies full image
            context_ring = (1 - hole_mask)
            
        ring_bool = (context_ring > 0)
        ring_pixels = image_rgb[ring_bool]
        
        if len(ring_pixels) == 0:
            ring_pixels = image_rgb.reshape(-1, 3)
            
        # 2. Ambient Illumination & Color Temperature
        mean_rgb = np.mean(ring_pixels, axis=0) # [R, G, B]
        r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
        
        # Color temperature proxy: R/B ratio
        rb_ratio = float(r / (b + 1e-5))
        if rb_ratio > 1.35:
            lighting_desc = "warm golden hour directional sunlight, 3500K color temperature"
        elif rb_ratio < 0.85:
            lighting_desc = "cool overcast daylight with diffused blue ambient skylight, 6500K color temperature"
        else:
            lighting_desc = "neutral balanced daylight, soft natural shadow gradients, 5500K illumination"
            
        # Exposure / Dynamic range
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        ring_gray = gray[ring_bool]
        mean_lum = float(np.mean(ring_gray))
        std_lum = float(np.std(ring_gray))
        
        if mean_lum < 75.0:
            exposure_desc = "moody low-key dramatic ambient lighting, rich deep shadows"
        elif mean_lum > 175.0:
            exposure_desc = "bright high-key airy illumination, clean open light"
        else:
            exposure_desc = "balanced mid-tone exposure, high dynamic range specular preservation"
            
        # 3. Dominant Surface Textures & High-Frequency Grain
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        ring_grad = grad_mag[ring_bool]
        mean_grad = float(np.mean(ring_grad)) if len(ring_grad) > 0 else 0.0
        
        if mean_grad > 35.0:
            texture_desc = "intricate high-frequency surface roughness, sharp micro-textures, weathered organic detail"
        elif mean_grad > 18.0:
            texture_desc = "semi-smooth natural texture, subtle surface grain, soft realistic material finish"
        else:
            texture_desc = "smooth uniform surface, clean continuous finish, subtle specular diffusion"
            
        # 4. Scene Category Classifier
        # Analyze chromatic distribution
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        ring_hsv = hsv[ring_bool]
        mean_sat = float(np.mean(ring_hsv[:, 1]))
        
        # Skin tone prior detection
        # Normalized RGB rule for human skin: R > G > B, (R-G) > 15
        is_skin = (r > g) and (g > b) and ((r - g) > 15) and (r > 60)
        
        # Foliage green detection (Hue 35-85 in OpenCV HSV)
        mean_hue = float(np.mean(ring_hsv[:, 0]))
        is_green = 35 <= mean_hue <= 85 and mean_sat > 50
        
        if is_skin:
            scene_cat = "portrait"
        elif is_green:
            scene_cat = "nature"
        elif mean_grad > 38.0 and mean_sat < 65:
            scene_cat = "street"
        elif mean_grad > 25.0:
            scene_cat = "architecture"
        else:
            scene_cat = "indoor"
            
        return {
            "scene_category": scene_cat,
            "lighting_desc": lighting_desc,
            "exposure_desc": exposure_desc,
            "texture_desc": texture_desc,
            "mean_luminance": round(mean_lum, 2),
            "color_temp_ratio": round(rb_ratio, 2),
            "edge_energy": round(mean_grad, 2)
        }

    def synthesize_smart_prompts(
        self,
        image: Any,
        mask: Any,
        user_prompt: str = "",
        user_negative_prompt: str = ""
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Stage 2: Synthesize context-aware positive and negative prompt strings.
        Returns (positive_prompt, negative_prompt, context_metadata).
        """
        # Convert to numpy RGB
        if isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            img_rgb = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError("Unsupported image type")
            
        if isinstance(mask, Image.Image):
            mask_bin = np.array(mask.convert("L"))
        elif isinstance(mask, np.ndarray):
            mask_bin = mask if mask.ndim == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported mask type")
            
        context = self.extract_vision_context(img_rgb, mask_bin)
        cat = context["scene_category"]
        cat_priors = self.scene_categories.get(cat, self.scene_categories["indoor"])
        
        # Build Positive Prompt
        clean_user_p = user_prompt.strip()
        if clean_user_p:
            # User provided a target object / action
            pos_prompt = (
                f"{clean_user_p}, {context['texture_desc']}, "
                f"{context['lighting_desc']}, {context['exposure_desc']}, "
                f"seamless natural inpainting, Hasselblad photographic quality, 8k resolution, tack sharp focus"
            )
        else:
            # Autonomous contextual hallucination
            pos_prompt = (
                f"{cat_priors['positive']}, {context['texture_desc']}, "
                f"{context['lighting_desc']}, {context['exposure_desc']}, "
                f"unbroken background continuity, photorealistic depth of field, 8k resolution"
            )
            
        # Build Negative Prompt
        clean_user_neg = user_negative_prompt.strip()
        base_neg = cat_priors["negative"]
        universal_neg = "blurry, duplicate, deformed, pixelated, desaturated, low-res, cartoonish, oversaturated, smear, JPEG artifacts, watermark, signature"
        
        if clean_user_neg:
            neg_prompt = f"{clean_user_neg}, {base_neg}, {universal_neg}"
        else:
            neg_prompt = f"{base_neg}, {universal_neg}"
            
        return pos_prompt, neg_prompt, context
