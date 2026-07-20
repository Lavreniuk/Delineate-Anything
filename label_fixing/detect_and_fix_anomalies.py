# Author: Yevhenii Salii (https://github.com/15Litrov)

#from osgeo import gdal
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy import ndimage
from scipy.ndimage import minimum_filter, maximum_filter

#gdal.UseExceptions()

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_label(path, img):
    lines = [np.array([int(img.shape[0] * float(v)) for v in line.split(' ')[1:]], dtype="int32").reshape(-1, 2) for line in open(path).readlines()]

    instances = np.zeros((img.shape[0], img.shape[1]), dtype="int32")
    for i in range(len(lines)):
        instances = cv2.fillPoly(instances, [lines[i]], color=int(i+1))

    return instances

def get_cv(image, label):
    image = cv2.medianBlur(image, 5)

    unique_ids = np.unique(label)

    # 3. Vectorized calculation for each RGB channel
    means_per_channel = []
    stds_per_channel = []

    for c in range(3):  # Loop over 3 channels (constant/negligible overhead)
        channel_data = image[..., c]
        
        # Fully vectorized calculation across all IDs at once
        mean_c = ndimage.mean(channel_data, labels=label, index=unique_ids)
        std_c = ndimage.standard_deviation(channel_data, labels=label, index=unique_ids)
        
        means_per_channel.append(mean_c)
        stds_per_channel.append(std_c)

    # 4. Stack results into shape (num_labels, 3)
    means = np.stack(means_per_channel, axis=-1)
    stds = np.stack(stds_per_channel, axis=-1)

    return np.max(stds / means, axis=1)

def find_deviating_labels_dynamic(
    rgb_image, 
    label_mask, 
    thresholds=[0.125, 0.25, 0.4], 
    error_rate_cutoffs=[0.50, 0.25, 0.10], 
    pixel_count_cutoffs=[256, 128, 64],
    brightness_scaling=(32.0, 2.0, 128.0, 1.0), # (low_brightness_th, low_scale, high_brightness_th, high_scale)
    blur_kernel=(7, 7), 
    erode_kernel_size=7,
    noise_erode_size=3
):
    """
    Finds label IDs where the filtered error footprint at ANY threshold level 
    exceeds its dynamic or static constraints, adjusting thresholds dynamically 
    based on the overall brightness amplitude of the object instance.
    """
    assert len(thresholds) == len(error_rate_cutoffs) == len(pixel_count_cutoffs), \
        "Thresholds, rate cutoffs, and pixel cutoffs must have identical lengths."

    # 1. Preprocessing and loopless mask erosion
    blurred_img = cv2.GaussianBlur(rgb_image, blur_kernel, 2)
    
    footprint = np.ones((erode_kernel_size, erode_kernel_size), dtype=bool)
    min_f = minimum_filter(label_mask, footprint=footprint)
    max_f = maximum_filter(label_mask, footprint=footprint)
    eroded_mask = np.where(min_f == max_f, min_f, 0)
    
    # Track dimensions
    orig_shape = label_mask.shape
    num_thresholds = len(thresholds)
    
    # 2. Extract total clean pixels per polygon label
    mask_flat = eroded_mask.ravel()
    valid_idx = mask_flat > 0
    mask_valid = mask_flat[valid_idx]
    
    if len(mask_valid) == 0:
        empty_img = np.zeros(orig_shape, dtype=np.uint8)
        return np.array([], dtype=np.int32), empty_img, empty_img

    max_label = np.max(mask_flat)
    total_counts = np.bincount(mask_valid, minlength=max_label + 1)
    total_counts_safe = np.where(total_counts == 0, 1, total_counts)
    
    # 3. Vectorial Mean and Amplitude Calculations
    img_flat = blurred_img.reshape(-1, 3).astype(np.float64)
    img_valid = img_flat[valid_idx]
    
    sums_R = np.bincount(mask_valid, weights=img_valid[:, 0], minlength=max_label + 1)
    sums_G = np.bincount(mask_valid, weights=img_valid[:, 1], minlength=max_label + 1)
    sums_B = np.bincount(mask_valid, weights=img_valid[:, 2], minlength=max_label + 1)
    
    means = np.column_stack((sums_R, sums_G, sums_B)) / total_counts_safe[:, np.newaxis]
    mean_amplitudes = np.linalg.norm(means, axis=1)
    mean_amplitudes[mean_amplitudes == 0] = 1.0
    
    # --- Brightness Adaptive Scaling (New Step) ---
    # Unpack the scaling configuration parameters
    low_val, low_scale, high_val, high_scale = brightness_scaling
    
    # Linearly interpolate a multiplier for each unique label based on its mean amplitude.
    # For values below low_val, it caps at low_scale. Above high_val, it caps at high_scale.
    label_scale_multipliers = np.interp(mean_amplitudes, [low_val, high_val], [low_scale, high_scale])
    
    # 4. Global Pixel Distance Deviations
    pixel_means = means[eroded_mask]
    pixel_amps = mean_amplitudes[eroded_mask]
    # Map multipliers array to match full 2D space layout
    pixel_multipliers = label_scale_multipliers[eroded_mask] 
    
    abs_deviation = np.linalg.norm(blurred_img - pixel_means, axis=2)
    thresh_arr = np.array(thresholds, dtype=np.float64)
    
    # 5. Multichannel Broadcaster (H, W, Num_Thresholds)
    # The thresholds are multiplied by both baseline amplitude AND our dynamic brightness scalar
    adaptive_thresholds = (thresh_arr[np.newaxis, np.newaxis, :] * pixel_amps[:, :, np.newaxis] * pixel_multipliers[:, :, np.newaxis])
    
    all_severity_masks_2d = abs_deviation[:, :, np.newaxis] > adaptive_thresholds
    all_severity_masks_2d &= (eroded_mask > 0)[:, :, np.newaxis]
    
    # 6. Noise Reduction via Spatial Multi-layer Erosion
    noise_footprint = np.ones((noise_erode_size, noise_erode_size, 1), dtype=bool) 
    filtered_severity_masks_2d = minimum_filter(all_severity_masks_2d, footprint=noise_footprint)
    
    # 7. Generate Visual Deviation Map
    deviation_map = np.sum(filtered_severity_masks_2d, axis=2).astype(np.uint8)
    filtered_severity_flat = filtered_severity_masks_2d.reshape(-1, num_thresholds)[valid_idx]
    
    # 8. Loopless Tally Matrix Construction
    error_counts_matrix = np.zeros((max_label + 1, num_thresholds))
    for t_idx in range(num_thresholds):
        error_counts_matrix[:, t_idx] = np.bincount(
            mask_valid, 
            weights=filtered_severity_flat[:, t_idx], 
            minlength=max_label + 1
        )
        
    # 9. Dynamic Boundary Evaluations
    cutoff_arr = np.array(error_rate_cutoffs, dtype=np.float64)
    pixel_cutoff_arr = np.array(pixel_count_cutoffs, dtype=np.float64)
    
    ratio_limits_pixels = total_counts_safe[:, np.newaxis] * cutoff_arr[np.newaxis, :]
    static_limits_pixels = pixel_cutoff_arr[np.newaxis, :]
    
    effective_pixel_thresholds = np.minimum(ratio_limits_pixels, static_limits_pixels)
    
    violates_limits = error_counts_matrix > effective_pixel_thresholds
    has_any_violation = np.any(violates_limits, axis=1)
    
    # 10. Filter IDs and Build Target Mask Output
    matching_ids = np.where(has_any_violation[1:])[0] + 1
    matching_labels_mask = np.isin(label_mask, matching_ids).astype(np.uint8)
    
    return matching_ids, deviation_map, matching_labels_mask

def get_robust_mode_color_image(img, label_mask, blur_ksize=5, quantize_step=16):
    """
    Generates an image where each polygon is filled with its robust mode RGB color.
    Uses blurring and color quantization to group similar colors before counting.
    """
    # 1. Blur to remove high-frequency single-pixel noise
    # medianBlur is excellent here because it preserves sharp edges between regions
    blurred = cv2.medianBlur(img, blur_ksize)
    
    # 2. Quantize colors to bin similar shades together
    # E.g., if step=16, we reduce 256 values down to 16 distinct bins per channel.
    # We use integer division to find the bin, then multiply back.
    quantized = (blurred.astype(np.int32) // quantize_step) * quantize_step
    
    # Add half the step back to center the color visually (prevents darkening)
    quantized = np.clip(quantized + (quantize_step // 2), 0, 255)
    
    mask_flat = label_mask.ravel()
    img_flat = quantized.reshape(-1, 3)
    
    valid_idx = mask_flat > 0
    img_valid = img_flat[valid_idx]
    mask_valid = mask_flat[valid_idx]
    
    if len(mask_valid) == 0:
        return np.zeros_like(img)
        
    max_label = np.max(mask_flat)
    
    # 3. Pack the 3 quantized color channels into a single 32-bit integer
    colors_packed = (img_valid[:, 0] << 16) | (img_valid[:, 1] << 8) | img_valid[:, 2]
    
    # 4. Combine label (shifted 32 bits) with color into 64-bit int
    combined = (mask_valid.astype(np.int64) << 32) | colors_packed.astype(np.int64)
    
    # 5. Count occurrences of these exact label + color bins
    uniques, counts = np.unique(combined, return_counts=True)
    
    labels_unq = uniques >> 32
    colors_unq = uniques & 0xFFFFFFFF
    
    # 6. Sort by label (ascending), then by counts (descending)
    order = np.lexsort((-counts, labels_unq))
    
    sorted_labels = labels_unq[order]
    sorted_colors = colors_unq[order]
    
    # 7. Extract the robust mode (first occurrence after sorting)
    _, first_idx = np.unique(sorted_labels, return_index=True)
    
    mode_labels = sorted_labels[first_idx]
    mode_colors = sorted_colors[first_idx]
    
    # 8. Unpack the winning bin colors back into RGB channels
    modes = np.zeros((max_label + 1, 3), dtype=np.uint8)
    modes[mode_labels, 0] = (mode_colors >> 16) & 0xFF
    modes[mode_labels, 1] = (mode_colors >> 8) & 0xFF
    modes[mode_labels, 2] = mode_colors & 0xFF
    
    # 9. Broadcast back into the 2D image shape
    mode_color_img = modes[label_mask]
    
    return mode_color_img

def get_mean_color_image(img, label_mask):
    """Generates an image where each polygon is filled with its mean RGB color."""
    mask_flat = label_mask.ravel()
    img_flat = img.reshape(-1, 3).astype(np.float64)
    
    valid_idx = mask_flat > 0
    img_valid = img_flat[valid_idx]
    mask_valid = mask_flat[valid_idx]
    
    max_label = np.max(mask_flat)
    counts = np.bincount(mask_valid, minlength=max_label + 1)
    counts_safe = np.where(counts == 0, 1, counts)[:, np.newaxis]
    
    sums_R = np.bincount(mask_valid, weights=img_valid[:, 0], minlength=max_label + 1)
    sums_G = np.bincount(mask_valid, weights=img_valid[:, 1], minlength=max_label + 1)
    sums_B = np.bincount(mask_valid, weights=img_valid[:, 2], minlength=max_label + 1)
    
    means = np.column_stack((sums_R, sums_G, sums_B)) / counts_safe
    
    # Broadcast the mean colors back into the 2D image shape
    mean_color_img = means[label_mask].astype(np.uint8)
    return mean_color_img

def synthesize_low_quality_regions(
    img, 
    label, 
    dev_labels_map, 
    blur_ksize=11, 
    bounds=[32, 32, 32]
):
    """
    Replaces targeted image fragments with semi-synthetic data.
    Uses bounded modulus arithmetic to inject local texture back into homogeneous mean colors.
    """
    # img_mean = get_mean_color_image(img, label)
    img_mean = get_robust_mode_color_image(img, label, 7, 8)

    # Convert to HSV to separate color (H) from intensity/brightness (S/V)
    hsv_orig = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mean = cv2.cvtColor(img_mean, cv2.COLOR_BGR2HSV).astype("float32")
    mean_h, mean_s, mean_v = cv2.split(hsv_mean)

    # 1. Extract High-Frequency Texture (H, S, and V channels)
    h_blurred = cv2.medianBlur(hsv_orig[:, :, 0], blur_ksize).astype("float32")
    s_blurred = cv2.medianBlur(hsv_orig[:, :, 1], blur_ksize).astype("float32")
    v_blurred = cv2.medianBlur(hsv_orig[:, :, 2], blur_ksize).astype("float32")
    
    h_diff = hsv_orig[:, :, 0].astype("float32") - h_blurred
    s_diff = hsv_orig[:, :, 1].astype("float32") - s_blurred
    v_diff = hsv_orig[:, :, 2].astype("float32") - v_blurred

    def bound_texture(diff_array, base):
        return base / 4 - np.abs(np.mod(diff_array + base / 4, base) - base / 2)

    h_texture = bound_texture(h_diff, bounds[0])
    s_texture = bound_texture(s_diff, bounds[1])
    v_texture = bound_texture(v_diff, bounds[2])

    # 3. Construct the Semi-Synthetic Image
    synth_h = mean_h + h_texture
    synth_s = mean_s + s_texture
    synth_v = mean_v + v_texture

    # 4. Enforce OpenCV HSV Boundaries
    # Hue in OpenCV is 0-179. We use modulo to wrap around seamlessly (e.g., red going past 179 back to 0).
    synth_h = np.mod(synth_h, 180) 
    # S and V are 0-255. We use clip because they hit hard limits (black/white).
    synth_s = np.clip(synth_s, 0, 255)
    synth_v = np.clip(synth_v, 0, 255)

    # Merge and convert back to BGR
    synth_hsv = cv2.merge([synth_h, synth_s, synth_v]).astype("uint8")
    synth_bgr = cv2.cvtColor(synth_hsv, cv2.COLOR_HSV2BGR)

    # 5. Apply the mask
    mask = dev_labels_map[:, :, None] == 1
    result = np.where(mask, synth_bgr, img)

    return result

def get_khalimsky_edges_permissive(image_band, kernel, stat='min'):
    """
    Computes edge values (1-cells) for a (2N+1) x (2M+1) Khalimsky grid.
    If the kernel is asymmetric or odd in one dimension, it skips evaluating 
    the unaligned edge direction and leaves those grid cells as 0.
    
    Parameters:
      image_band (2D array): Input image channel (N x M).
      kernel (2D array): Custom numpy array representing the kernel.
      stat (str): 'min', 'max', or 'sum'.
    """
    kh_h, kh_w = kernel.shape
    N, M = image_band.shape
    kh_N, kh_M = 2 * N + 1, 2 * M + 1
    
    # Choose base morphology functions
    if stat == 'min':
        filter_func = ndimage.minimum_filter
    elif stat == 'max':
        filter_func = ndimage.maximum_filter
    elif stat == 'sum':
        filter_func = lambda img, fp, mode: ndimage.correlate(img, fp, mode=mode)
    else:
        raise ValueError("stat must be 'min', 'max', or 'sum'")

    # Allocate sparse Khalimsky canvas
    dtype = np.float32 if stat == 'sum' else image_band.dtype
    edge_grid = np.zeros((kh_N, kh_M), dtype=dtype)
    
    # ----------------------------------------------------
    # 1. Horizontal Edges (Odd, Even) -> Requires EVEN Width
    # ----------------------------------------------------
    if kh_w % 2 == 0:
        kernel_left = kernel[:, :kh_w // 2]
        kernel_right = kernel[:, kh_w // 2:]
        
        if stat == 'sum':
            left_half = filter_func(image_band, kernel_left, mode='reflect')
            right_half = filter_func(image_band, kernel_right, mode='reflect')
            right_half = np.roll(right_half, -1, axis=1)
            h_edges = left_half + right_half
        else:
            footprint_l = (kernel_left != 0)
            footprint_r = (kernel_right != 0)
            left_half = filter_func(image_band, footprint=footprint_l, mode='reflect')
            right_half = filter_func(image_band, footprint=footprint_r, mode='reflect')
            right_half = np.roll(right_half, -1, axis=1)
            h_edges = np.minimum(left_half, right_half) if stat == 'min' else np.maximum(left_half, right_half)

        # Map to grid
        edge_grid[1::2, 2:-1:2] = h_edges[:, :M-1]
        edge_grid[1::2, 0]  = h_edges[:, 0]
        edge_grid[1::2, -1] = h_edges[:, -1]
    else:
        # Width is odd; skip horizontal edges
        pass

    # ----------------------------------------------------
    # 2. Vertical Edges (Even, Odd) -> Requires EVEN Height
    # ----------------------------------------------------
    if kh_h % 2 == 0:
        kernel_top = kernel[:kh_h // 2, :]
        kernel_bottom = kernel[kh_h // 2:, :]
        
        if stat == 'sum':
            top_half = filter_func(image_band, kernel_top, mode='reflect')
            bottom_half = filter_func(image_band, kernel_bottom, mode='reflect')
            bottom_half = np.roll(bottom_half, -1, axis=0)
            v_edges = top_half + bottom_half
        else:
            footprint_t = (kernel_top != 0)
            footprint_b = (kernel_bottom != 0)
            top_half = filter_func(image_band, footprint=footprint_t, mode='reflect')
            bottom_half = filter_func(image_band, footprint=footprint_b, mode='reflect')
            bottom_half = np.roll(bottom_half, -1, axis=0)
            v_edges = np.minimum(top_half, bottom_half) if stat == 'min' else np.maximum(top_half, bottom_half)

        # Map to grid
        edge_grid[2:-1:2, 1::2] = v_edges[:N-1, :]
        edge_grid[0, 1::2]  = v_edges[0, :]
        edge_grid[-1, 1::2] = v_edges[-1, :]
    else:
        # Height is odd; skip vertical edges
        pass

    return edge_grid

def create_khalimsky_grid(instance_map):
    edge_instances_mask = np.zeros((2 * instance_map.shape[0] + 1, 2 * instance_map.shape[1] + 1), dtype="uint8")
    edge_instances_mask[2:-1:2, 1::2] |= instance_map[:-1, :] != instance_map[1:, :]
    edge_instances_mask[1::2, 2:-1:2] |= instance_map[:, :-1] != instance_map[:, 1:]

    # 0. Calculating anchorage - how many edges goes through pixel junction
    edge_instances_mask[2:-1:2, 2:-1:2] = edge_instances_mask[1:-2:2, 2:-1:2] + edge_instances_mask[3::2, 2:-1:2] + edge_instances_mask[2:-1:2, 1:-2:2] + edge_instances_mask[2:-1:2, 3::2]

    # 1. Top Row of Junctions (y=0, excluding corners)
    edge_instances_mask[0, 2:-1:2] = edge_instances_mask[0, 1:-2:2] + edge_instances_mask[0, 3::2] + edge_instances_mask[1, 2:-1:2]

    # 2. Bottom Row of Junctions (y=max, excluding corners)
    edge_instances_mask[-1, 2:-1:2] = edge_instances_mask[-1, 1:-2:2] + edge_instances_mask[-1, 3::2] + edge_instances_mask[-2, 2:-1:2]

    # 3. Left Column of Junctions (x=0, excluding corners)
    edge_instances_mask[2:-1:2, 0] = edge_instances_mask[1:-2:2, 0] + edge_instances_mask[3::2, 0] + edge_instances_mask[2:-1:2, 1]

    # 4. Right Column of Junctions (x=max, excluding corners)
    edge_instances_mask[2:-1:2, -1] = edge_instances_mask[1:-2:2, -1] + edge_instances_mask[3::2, -1] + edge_instances_mask[2:-1:2, -2]

    # Top-Left [0,0]
    edge_instances_mask[0, 0] = edge_instances_mask[0, 1] + edge_instances_mask[1, 0]
    # Top-Right [0,-1]
    edge_instances_mask[0, -1] = edge_instances_mask[0, -2] + edge_instances_mask[1, -1]
    # Bottom-Left [-1,0]
    edge_instances_mask[-1, 0] = edge_instances_mask[-1, 1] + edge_instances_mask[-2, 0]
    # Bottom-Right [-1,-1]
    edge_instances_mask[-1, -1] = edge_instances_mask[-1, -2] + edge_instances_mask[-2, -1]

    return edge_instances_mask

def get_edge_strength(img, label):
    kernel_hor = np.array([[1, 1, 1, 1]])
    kernel_ver = np.array([[1], [1], [1], [1]])
    horiz = np.stack([get_khalimsky_edges_permissive(img[:, :, i], kernel_hor, 'max') - get_khalimsky_edges_permissive(img[:, :, i], kernel_hor, 'min') for i in range(3)], axis=2)
    verti = np.stack([get_khalimsky_edges_permissive(img[:, :, i], kernel_ver, 'max') - get_khalimsky_edges_permissive(img[:, :, i], kernel_ver, 'min') for i in range(3)], axis=2)
    return np.max(np.maximum(horiz, verti), axis=2)

def detect_problematic_edges(edge_strength, edge_instances, 
                             mean_thresh=0.3, min_strength_thresh=0.2, ratio_thresh=0.4):
    """
    Detects weak edge instances based on mean strength and proportion of weak pixels.
    
    Parameters:
    -----------
    edge_strength : np.ndarray (2D float)
        Array containing edge strengths (typically normalized between 0 and 1).
    edge_instances : np.ndarray (2D int)
        Array containing edge IDs. 0 represents background/no edge.
    mean_thresh : float
        Edges with a mean strength below this are flagged.
    min_strength_thresh : float
        The threshold below which an individual pixel is considered "weak".
    ratio_thresh : float
        If more than this percentage of an edge's pixels are weak, it's flagged.
        
    Returns:
    --------
    problematic_mask : np.ndarray (2D bool)
        A mask where True indicates a problematic edge instance.
    problematic_ids : list
        List of the flagged edge IDs.
    """
    # 1. Find all unique edge IDs (excluding background ID 0)
    unique_ids = np.unique(edge_instances)
    unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) == 0:
        return np.zeros_like(edge_instances, dtype=bool), []

    # 2. Vectorized computation of pixel counts and sums per ID
    # np.bincount efficiently aggregates data based on integer IDs
    pixel_counts = np.bincount(edge_instances.ravel())
    sum_strengths = np.bincount(edge_instances.ravel(), weights=edge_strength.ravel())
    
    # 3. Condition 1: Identify IDs with low mean strength
    # Prevent division by zero for safety
    safe_counts = np.where(pixel_counts == 0, 1, pixel_counts)
    mean_strengths = sum_strengths / safe_counts
    
    # Filter for our active unique IDs
    low_mean_mask = mean_strengths[unique_ids] < mean_thresh
    bad_by_mean = unique_ids[low_mean_mask]
    
    # 4. Condition 2: Identify IDs with "too many" weak pixels
    # Create a binary mask of where pixels are critically weak
    weak_pixels_mask = (edge_strength < min_strength_thresh) & (edge_instances != 0)
    # Count how many weak pixels belong to each instance ID
    weak_counts_per_id = np.bincount(edge_instances.ravel(), weights=weak_pixels_mask.ravel())
    
    # Calculate the ratio of weak pixels for each unique ID
    weak_ratios = weak_counts_per_id[unique_ids] / pixel_counts[unique_ids]
    too_many_weak_mask = weak_ratios > ratio_thresh
    bad_by_ratio = unique_ids[too_many_weak_mask]
    
    # 5. Combine results
    problematic_ids = np.union1d(bad_by_mean, bad_by_ratio).tolist()
    
    # 6. Generate the final 2D boolean mask
    # np.isin checks every pixel to see if its ID is in our problematic list
    problematic_mask = np.isin(edge_instances, problematic_ids)
    
    return problematic_mask, problematic_ids

def create_image_res_via_khalimsky_downsample(img, edge_strength):
    """
    1. Creates a full Khalimsky grid (2H+1, 2W+1).
    2. Overlays contrast colors where edge strength > 0.
    3. Downsamples back to (H, W) by distributing half of edge values 
       and a quarter of vertex values to their constituent pixels.
    """
    H, W, C = img.shape
    kh_H, kh_W = 2 * H + 1, 2 * W + 1
    
    if edge_strength.shape != (kh_H, kh_W):
        raise ValueError(f"edge_strength must have shape (2H+1, 2W+1), expected ({kh_H}, {kh_W})")
    
    # ==========================================
    # 1. BUILD FULL KHALIMSKY GRID (Your 1st Code)
    # ==========================================
    kh_img = np.zeros((kh_H, kh_W, C), dtype=np.uint8)
    img_f = img.astype(np.float32)
    
    kh_img[1::2, 1::2] = img

    # Internal elements
    h_mean_color = ((img_f[:, :-1] + img_f[:, 1:]) / 2.0).astype(np.uint8)
    kh_img[1::2, 2:-1:2] = h_mean_color

    v_mean_color = ((img_f[:-1, :] + img_f[1:, :]) / 2.0).astype(np.uint8)
    kh_img[2:-1:2, 1::2] = v_mean_color

    vertex_mean_color = ((img_f[:-1, :-1] + img_f[:-1, 1:] + 
                           img_f[1:, :-1] + img_f[1:, 1:]) / 4.0).astype(np.uint8)
    kh_img[2:-1:2, 2:-1:2] = vertex_mean_color

    # External borders
    kh_img[0, 1::2] = img[0, :]; kh_img[-1, 1::2] = img[-1, :]
    kh_img[0, 2:-1:2] = h_mean_color[0, :]; kh_img[-1, 2:-1:2] = h_mean_color[-1, :]
    kh_img[1::2, 0] = img[:, 0]; kh_img[1::2, -1] = img[:, -1]
    kh_img[2:-1:2, 0] = v_mean_color[:, 0]; kh_img[2:-1:2, -1] = v_mean_color[:, -1]
    kh_img[0, 0] = img[0, 0]; kh_img[0, -1] = img[0, -1]
    kh_img[-1, 0] = img[-1, 0]; kh_img[-1, -1] = img[-1, -1]

    # ==========================================
    # 2. MODIFY EDGES BY STRENGTH (Your 1st Code)
    edge_mask = edge_strength > 0
    if np.any(edge_mask):
        if C == 3:
            brightness = (0.299 * kh_img[..., 0] + 0.587 * kh_img[..., 1] + 0.114 * kh_img[..., 2])
        else:
            brightness = np.mean(kh_img, axis=-1)
            
        # Target color: 0.0 (pure shadow multiplier) or 1.0 (pure highlight multiplier)
        contrast_target = np.where(brightness[..., None] > 64, 0.0, 1.0)
        
        # Normalize edge_strength to [0.0, 1.0]
        alpha = edge_strength[..., None]
        
        # Blend the target multiplier into a neutral 0.5 (no-change) baseline
        blend_source = (1.0 - alpha) * 0.5 + alpha * contrast_target
        
        # Standard mathematical Soft-Light formula (multiplicative shading)
        kh_img_f = kh_img.astype(np.float32) / 255.0
        
        # Formula: (2 * blend * image) + (image^2 * (1 - 2 * blend))
        soft_light = (2.0 * blend_source * kh_img_f) + (kh_img_f**2 * (1.0 - 2.0 * blend_source))
        
        # Scale back up to uint8 bounds
        blended_kh = np.clip(soft_light * 255.0, 0, 255)
        
        kh_img = np.where(edge_mask[..., None], blended_kh, kh_img_f * 255.0).astype(np.uint8)

    # ==========================================
    # 3. LINEAR DOWNSAMPLE TO IMAGE RESOLUTION
    # ==========================================
    # Convert to float for clean mathematical distribution / averaging
    kh_f = kh_img.astype(np.float32)
    
    # Extract the components relative to each center pixel (1::2, 1::2)
    centers = kh_f[1::2, 1::2]
    
    # Orthogonal neighbors (Edges get distributed 50% to each side)
    top_edges    = kh_f[0:-1:2, 1::2]
    bottom_edges = kh_f[2::2,   1::2]
    left_edges   = kh_f[1::2,   0:-1:2]
    right_edges  = kh_f[1::2,   2::2]
    
    # Diagonal neighbors (Vertices get distributed 25% to all 4 surrounding pixels)
    top_left_vert     = kh_f[0:-1:2, 0:-1:2]
    top_right_vert    = kh_f[0:-1:2, 2::2]
    bottom_left_vert  = kh_f[2::2,   0:-1:2]
    bottom_right_vert = kh_f[2::2,   2::2]
    
    # Sum up the contributions based on kernel footprint area
    downsampled = (
        1.00 * centers +
        0.50 * (top_edges + bottom_edges + left_edges + right_edges) +
        0.25 * (top_left_vert + top_right_vert + bottom_left_vert + bottom_right_vert)
    ) / 4.0 # Normalized by total kernel weight area (1 + 4*0.5 + 4*0.25 = 4.0)

    return np.clip(downsampled, 0, 255).astype(np.uint8)