#!/usr/bin/env python3
"""
scripts/benchmarks/extract_50_features.py
================================================================================
⚡ 50-DIMENSIONAL VECTORIZED TOPOLOGICAL & CONTEXTUAL FEATURE ENGINE
================================================================================
High-throughput, vectorized extraction on 128x128 proxy grid across all 6,000 images:
  1. Geometry & Distance (1-12)
  2. Spatial & Placement (13-16)
  3. Gradients & Boundary Differentials (17-24)
  4. Texture & Multi-Scale GLCM/LBP (25-32)
  5. Frequency & Spectral Power (33-38)
  6. Color & Chromatic Information (39-44)
  7. Structural Vertices & Curvature (45-50)

Multiprocessing: 8 parallel host workers.
Output: artifacts/dataset_50_features_6000.csv
================================================================================
"""

import os
import sys
import time
import math
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import euler_number as calc_euler_number

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILED_DIR = REPO_ROOT / "Benchmark" / "6000_images" / "compiled"
IMAGES_DIR = COMPILED_DIR / "images"
MASKS_DIR = COMPILED_DIR / "masks"
IMAGE_MANIFEST = COMPILED_DIR / "dataset_manifest.csv"
MASK_MANIFEST = COMPILED_DIR / "masks_manifest.csv"

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ARTIFACTS_DIR / "dataset_50_features_6000.csv"

PROXY_SIZE = (128, 128)

def compute_50_features(sample_tuple) -> Dict[str, Any]:
    sample_id, img_path_str, mask_path_str, category = sample_tuple
    
    img_bgr = cv2.imread(img_path_str)
    mask_gray = cv2.imread(mask_path_str, cv2.IMREAD_GRAYSCALE)
    if img_bgr is None or mask_gray is None:
        return {}

    img_128 = cv2.resize(img_bgr, PROXY_SIZE, interpolation=cv2.INTER_AREA)
    mask_128 = cv2.resize(mask_gray, PROXY_SIZE, interpolation=cv2.INTER_NEAREST)
    bin_mask = (mask_128 >= 128).astype(np.uint8)
    
    h, w = PROXY_SIZE
    total_px = h * w
    hole_px = int(np.sum(bin_mask))
    if hole_px == 0:
        return {}

    gray_128 = cv2.cvtColor(img_128, cv2.COLOR_BGR2GRAY)
    
    # -------------------------------------------------------------
    # Group 1: Geometry & Distance (1-12)
    # -------------------------------------------------------------
    rho = float(hole_px) / float(total_px)
    
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    if perimeter == 0:
        perimeter = 1.0
        
    compactness = float((4.0 * math.pi * hole_px) / (perimeter ** 2))
    compactness = min(1.0, compactness)
    
    dist_map = distance_transform_edt(bin_mask)
    hole_depths = dist_map[bin_mask == 1]
    r_max = float(np.max(dist_map))
    mean_depth = float(np.mean(hole_depths)) if len(hole_depths) > 0 else 0.0
    std_depth = float(np.std(hole_depths)) if len(hole_depths) > 0 else 0.0
    max_to_mean_ratio = float(r_max / (mean_depth + 1e-5))
    
    hull_area = 0.0
    for c in contours:
        if len(c) >= 3:
            hull = cv2.convexHull(c)
            hull_area += cv2.contourArea(hull)
    solidity = float(hole_px / hull_area) if hull_area > 0 else 1.0
    solidity = min(1.0, max(0.01, solidity))
    
    x, y, bw, bh = cv2.boundingRect(bin_mask)
    aspect_ratio = float(bw) / float(bh + 1e-5)
    extent = float(hole_px) / float((bw * bh) + 1e-5)
    
    n_labels, _, _, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    n_cc = max(1, n_labels - 1)
    
    try:
        e_num = int(calc_euler_number(bin_mask, connectivity=2))
    except:
        e_num = 1

    # -------------------------------------------------------------
    # Group 2: Spatial & Placement (13-16)
    # -------------------------------------------------------------
    M = cv2.moments(bin_mask)
    if M["m00"] != 0:
        cx_norm = (M["m10"] / M["m00"]) / float(w)
        cy_norm = (M["m01"] / M["m00"]) / float(h)
    else:
        cx_norm, cy_norm = 0.5, 0.5
        
    centroid_offset_x = abs(cx_norm - 0.5)
    centroid_offset_y = abs(cy_norm - 0.5)
    radial_eccentricity = math.sqrt((cx_norm - 0.5)**2 + (cy_norm - 0.5)**2) / 0.7071
    
    rot_x = min(abs(cx_norm - 1.0/3.0), abs(cx_norm - 2.0/3.0))
    rot_y = min(abs(cy_norm - 1.0/3.0), abs(cy_norm - 2.0/3.0))
    rot_proximity = 1.0 - min(1.0, (rot_x + rot_y))

    # -------------------------------------------------------------
    # Context Boundary Ring (Dilate 4px on 128 = 16px on 512)
    # -------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(bin_mask, kernel, iterations=1)
    boundary_ring = (dilated > 0) & (bin_mask == 0)
    if np.sum(boundary_ring) == 0:
        boundary_ring = (bin_mask == 0)

    # -------------------------------------------------------------
    # Group 3: Gradients & Boundary Differentials (17-24)
    # -------------------------------------------------------------
    sobelx = cv2.Sobel(gray_128, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_128, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    
    b_sobel = sobel_mag[boundary_ring]
    edge_density_sobel = float(np.mean(b_sobel)) if len(b_sobel) > 0 else 0.0
    edge_std_sobel = float(np.std(b_sobel)) if len(b_sobel) > 0 else 0.0
    edge_max_sobel = float(np.max(b_sobel)) if len(b_sobel) > 0 else 0.0
    
    scharrx = cv2.Scharr(gray_128, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray_128, cv2.CV_64F, 0, 1)
    scharr_energy = float(np.mean(np.sqrt(scharrx**2 + scharry**2)[boundary_ring]))
    
    canny_edges = cv2.Canny(gray_128, 100, 200)
    canny_density = float(np.mean(canny_edges[boundary_ring] > 0))
    
    grad_angles = np.arctan2(sobely[boundary_ring], sobelx[boundary_ring] + 1e-7)
    hist_angle, _ = np.histogram(grad_angles, bins=36, range=(-math.pi, math.pi), density=True)
    hist_angle = hist_angle[hist_angle > 0]
    angle_entropy = float(-np.sum(hist_angle * np.log2(hist_angle))) if len(hist_angle) > 0 else 0.0
    
    lap = cv2.Laplacian(gray_128, cv2.CV_64F)
    b_lap = lap[boundary_ring]
    lap_var = float(np.var(b_lap)) if len(b_lap) > 0 else 0.0
    lap_mean = float(np.mean(np.abs(b_lap))) if len(b_lap) > 0 else 0.0

    # -------------------------------------------------------------
    # Group 4: Texture & Multi-Scale GLCM/LBP (25-32)
    # -------------------------------------------------------------
    # Quantize gray to 16 levels for fast GLCM
    gray_16 = (gray_128 // 16).astype(np.uint8)
    glcm = graycomatrix(gray_16, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
    
    glcm_contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    glcm_dissimilarity = float(graycoprops(glcm, 'dissimilarity')[0, 0])
    glcm_homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    glcm_energy = float(graycoprops(glcm, 'energy')[0, 0])
    glcm_correlation = float(graycoprops(glcm, 'correlation')[0, 0])
    
    # Fast LBP (P=8, R=1)
    lbp = local_binary_pattern(gray_128, P=8, R=1, method='uniform')
    b_lbp = lbp[boundary_ring]
    hist_lbp, _ = np.histogram(b_lbp, bins=10, range=(0, 10), density=True)
    hist_lbp = hist_lbp[hist_lbp > 0]
    lbp_entropy = float(-np.sum(hist_lbp * np.log2(hist_lbp))) if len(hist_lbp) > 0 else 0.0
    lbp_uniformity = float(np.sum(hist_lbp ** 2)) if len(hist_lbp) > 0 else 0.0
    
    blur_s1 = cv2.GaussianBlur(gray_128, (3, 3), 1.0)
    blur_s2 = cv2.GaussianBlur(gray_128, (5, 5), 2.0)
    var_s1 = float(cv2.Laplacian(blur_s1, cv2.CV_64F).var())
    var_s2 = float(cv2.Laplacian(blur_s2, cv2.CV_64F).var())
    ms_lap_ratio = float(var_s1 / (var_s2 + 1e-5))

    # -------------------------------------------------------------
    # Group 5: Frequency & Spectral Power (33-38)
    # -------------------------------------------------------------
    f_shift = np.fft.fftshift(np.fft.fft2(gray_128))
    mag_spec = np.abs(f_shift) ** 2
    total_energy = float(np.sum(mag_spec)) + 1e-7
    
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r_grid = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_max_f = math.sqrt(cx**2 + cy**2)
    
    low_mask = r_grid < (0.1 * r_max_f)
    mid_mask = (r_grid >= 0.1 * r_max_f) & (r_grid < 0.3 * r_max_f)
    high_mask = r_grid >= (0.3 * r_max_f)
    
    low_energy = float(np.sum(mag_spec[low_mask])) / total_energy
    mid_energy = float(np.sum(mag_spec[mid_mask])) / total_energy
    high_energy = float(np.sum(mag_spec[high_mask])) / total_energy
    high_to_low_ratio = float(high_energy / (low_energy + 1e-7))
    
    # Spectral decay slope
    r_int = np.clip(r_grid.astype(int), 1, 63)
    rad_prof = np.bincount(r_int.ravel(), weights=mag_spec.ravel())[:64]
    rad_counts = np.bincount(r_int.ravel())[:64] + 1e-7
    rad_mean = rad_prof / rad_counts
    freqs = np.arange(1, 64)
    valid = rad_mean[1:64] > 0
    if np.sum(valid) > 5:
        log_f = np.log(freqs[valid])
        log_p = np.log(rad_mean[1:64][valid])
        slope, _ = np.polyfit(log_f, log_p, 1)
        decay_slope = float(slope)
    else:
        decay_slope = -1.5
        
    theta = np.arctan2(yy - cy, xx - cx)
    ang_bins = 12
    ang_int = np.clip(((theta + math.pi) / (2 * math.pi) * ang_bins).astype(int), 0, ang_bins - 1)
    ang_prof = np.bincount(ang_int.ravel(), weights=mag_spec.ravel(), minlength=ang_bins)
    anisotropy = float(np.std(ang_prof) / (np.mean(ang_prof) + 1e-7))

    # -------------------------------------------------------------
    # Group 6: Color & Chromatic Information (39-44)
    # -------------------------------------------------------------
    ent_channels = []
    for c in range(3):
        ch_vals = img_128[:, :, c][boundary_ring]
        h_c, _ = np.histogram(ch_vals, bins=32, range=(0, 256), density=True)
        h_c = h_c[h_c > 0]
        ent_channels.append(-np.sum(h_c * np.log2(h_c)) if len(h_c) > 0 else 0.0)
    rgb_entropy = float(np.mean(ent_channels))
    
    img_lab = cv2.cvtColor(img_128, cv2.COLOR_BGR2LAB)
    l_vals = img_lab[:, :, 0][boundary_ring].astype(float)
    a_vals = img_lab[:, :, 1][boundary_ring].astype(float)
    b_vals = img_lab[:, :, 2][boundary_ring].astype(float)
    delta_e = np.sqrt(np.var(l_vals) + np.var(a_vals) + np.var(b_vals)) if len(l_vals) > 0 else 0.0
    
    img_hsv = cv2.cvtColor(img_128, cv2.COLOR_BGR2HSV)
    sat_vals = img_hsv[:, :, 1][boundary_ring].astype(float)
    hsv_sat_mean = float(np.mean(sat_vals)) if len(sat_vals) > 0 else 0.0
    hsv_sat_std = float(np.std(sat_vals)) if len(sat_vals) > 0 else 0.0
    
    r_mean = float(np.mean(img_128[:, :, 2][boundary_ring]))
    b_mean = float(np.mean(img_128[:, :, 0][boundary_ring]))
    color_temp_proxy = float(r_mean / (b_mean + 1e-5))
    
    # Border vs Global divergence
    hist_b, _ = np.histogram(gray_128[boundary_ring], bins=16, range=(0, 256), density=True)
    hist_g, _ = np.histogram(gray_128, bins=16, range=(0, 256), density=True)
    color_div = float(np.sum(np.abs(hist_b - hist_g)))

    # -------------------------------------------------------------
    # Group 7: Structural Vertices & Curvature (45-50)
    # -------------------------------------------------------------
    corners = cv2.goodFeaturesToTrack(gray_128, maxCorners=80, qualityLevel=0.01, minDistance=5)
    corner_count = 0
    if corners is not None:
        for pt in corners:
            px, py = int(pt[0][0]), int(pt[0][1])
            if 0 <= py < h and 0 <= px < w and boundary_ring[py, px]:
                corner_count += 1
                
    fast = cv2.FastFeatureDetector_create(threshold=25)
    kp = fast.detect(gray_128, None)
    fast_density = float(len(kp) / (perimeter + 1e-5)) if kp else 0.0
    
    curvatures = []
    for c in contours:
        if len(c) >= 5:
            pts = c.squeeze()
            if pts.ndim == 2 and len(pts) >= 5:
                dx = np.gradient(pts[:, 0])
                dy = np.gradient(pts[:, 1])
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                k_val = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-5)
                curvatures.extend(k_val)
    mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0
    extrema_count = int(np.sum(np.array(curvatures) > 0.5)) if curvatures else 0
    
    convex_p = sum(cv2.arcLength(cv2.convexHull(c), True) for c in contours if len(c) >= 3)
    tortuosity = float(perimeter / (convex_p + 1e-5))
    
    # Skeleton branch points proxy
    skel = dist_map > (r_max * 0.5)
    branch_points = int(cv2.connectedComponents(skel.astype(np.uint8))[0] - 1)

    return {
        "sample_id": sample_id,
        "category": category,
        # 1-12
        "mask_area_ratio_rho": round(rho, 4),
        "mask_perimeter": round(perimeter, 2),
        "compactness_isoperimetric": round(compactness, 4),
        "max_inscribed_radius_rmax": round(r_max, 2),
        "mean_hole_depth": round(mean_depth, 2),
        "std_hole_depth": round(std_depth, 2),
        "max_to_mean_depth_ratio": round(max_to_mean_ratio, 3),
        "convex_hull_solidity": round(solidity, 4),
        "mask_aspect_ratio": round(aspect_ratio, 3),
        "mask_extent": round(extent, 4),
        "connected_components_count": int(n_cc),
        "euler_number": int(e_num),
        # 13-16
        "centroid_offset_x": round(centroid_offset_x, 4),
        "centroid_offset_y": round(centroid_offset_y, 4),
        "radial_eccentricity": round(radial_eccentricity, 4),
        "rule_of_thirds_proximity": round(rot_proximity, 4),
        # 17-24
        "boundary_edge_density_sobel": round(edge_density_sobel, 2),
        "boundary_edge_std_sobel": round(edge_std_sobel, 2),
        "boundary_edge_max_sobel": round(edge_max_sobel, 2),
        "boundary_scharr_energy": round(scharr_energy, 2),
        "boundary_canny_density": round(canny_density, 4),
        "gradient_direction_entropy": round(angle_entropy, 3),
        "boundary_laplacian_var": round(lap_var, 2),
        "boundary_laplacian_mean": round(lap_mean, 2),
        # 25-32
        "glcm_contrast": round(glcm_contrast, 3),
        "glcm_dissimilarity": round(glcm_dissimilarity, 3),
        "glcm_homogeneity": round(glcm_homogeneity, 4),
        "glcm_energy": round(glcm_energy, 4),
        "glcm_correlation": round(glcm_correlation, 4),
        "lbp_entropy": round(lbp_entropy, 3),
        "lbp_uniformity": round(lbp_uniformity, 4),
        "multi_scale_laplacian_ratio": round(ms_lap_ratio, 3),
        # 33-38
        "fft_low_freq_energy": round(low_energy, 4),
        "fft_mid_freq_energy": round(mid_energy, 4),
        "fft_high_freq_energy": round(high_energy, 4),
        "fft_high_to_low_ratio": round(high_to_low_ratio, 4),
        "spectral_decay_slope": round(decay_slope, 3),
        "spectral_azimuthal_anisotropy": round(anisotropy, 3),
        # 39-44
        "context_rgb_shannon_entropy": round(rgb_entropy, 3),
        "context_lab_delta_e_mean": round(delta_e, 2),
        "context_hsv_saturation_mean": round(hsv_sat_mean, 2),
        "context_hsv_saturation_std": round(hsv_sat_std, 2),
        "context_color_temperature_proxy": round(color_temp_proxy, 3),
        "border_to_global_color_divergence": round(color_div, 4),
        # 45-50
        "shi_tomasi_corner_count": int(corner_count),
        "fast_keypoint_density": round(fast_density, 3),
        "contour_curvature_mean": round(mean_curvature, 4),
        "contour_curvature_extrema": int(extrema_count),
        "boundary_tortuosity": round(tortuosity, 3),
        "skeleton_branch_points": int(branch_points)
    }

def main():
    print("=" * 80)
    print(" ⚡ VECTORIZED 50-FEATURE EXTRACTION ENGINE (6,000 SAMPLES)")
    print("=" * 80)

    img_df = pd.read_csv(IMAGE_MANIFEST)
    print(f"[INPUT] Loaded dataset manifest: {len(img_df)} images.")

    tasks = []
    for _, row in img_df.iterrows():
        s_id = int(row["sample_id"])
        fname = row["filename"]
        cat = row["category"]
        img_p = str(IMAGES_DIR / fname)
        mask_p = str(MASKS_DIR / fname)
        tasks.append((s_id, img_p, mask_p, cat))

    print(f"[WORKERS] Launching {mp.cpu_count()} parallel worker processes...")
    t0 = time.time()
    
    n_workers = min(8, mp.cpu_count())
    with mp.Pool(processes=n_workers) as pool:
        # Use imap_unordered for fast streaming
        results = []
        for i, res in enumerate(pool.imap_unordered(compute_50_features, tasks, chunksize=50)):
            if res:
                results.append(res)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  Processed {i+1}/6000 ({rate:.1f} samples/s, ~{(6000-i-1)/rate:.0f}s remaining)...")

    total_time = time.time() - t0
    print(f"\n[BENCHMARK] Extracted 50 features across {len(results)} samples in {total_time:.1f}s ({total_time/len(results)*1000:.2f} ms/sample)!")
    
    df_out = pd.DataFrame(results)
    df_out.sort_values(by="sample_id", inplace=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[EXPORT] Saved 50-dimensional feature matrix to: {OUT_CSV}")
    print("=" * 80)

if __name__ == "__main__":
    main()
