# filename: depth_utils.py

import cupy as cp
import numpy as np

def create_valid_mask_from_depth_concentration(depth_map, min_depth=0.7, max_depth=50.0, step_size=0.1, min_points_threshold=20):
    """
    Creates a CuPy boolean mask by quantizing the depth map and filtering out
    depth regions that contain a low concentration of points.

    This function works by:
    1. Quantizing the depth map into discrete bins of size `step_size` within the
       `min_depth` to `max_depth` range.
    2. Counting the number of points that fall into each bin.
    3. Creating a mask where a bin is marked as valid only if it contains a number
       of points greater than or equal to `min_points_threshold`.
    4. Mapping this bin-validity mask back to the original depth map to create
       a final pixel-wise boolean mask.

    Args:
        depth_map (cp.ndarray): The input 2D depth map (H, W) as a CuPy array.
        min_depth (float): The minimum valid depth in meters.
        max_depth (float): The maximum valid depth in meters.
        step_size (float): The size of each depth quantization bin in meters.
        min_points_threshold (int): The minimum number of points required for a
                                    depth bin to be considered valid.

    Returns:
        cp.ndarray: A CuPy boolean mask (cp.bool_) of the same shape, where
                    True indicates a pixel is valid and False indicates a pixel
                    should be filtered out.
    """
    if depth_map.ndim != 2 or depth_map.shape != (480, 640):
        raise ValueError("Input depth map must be a 2D CuPy array of shape (480, 640).")
    
    # 1. Create a mask for depths within the valid range
    valid_range_mask = (depth_map >= min_depth) & (depth_map <= max_depth)
    
    # 2. Quantize the depth map for valid points
    # This maps each depth value to an integer bin index
    quantized_depth = cp.floor((depth_map - min_depth) / step_size).astype(cp.int32)
    
    # Set invalid points to a negative number so they are not counted by bincount
    quantized_depth[~valid_range_mask] = -1

    # Get a flattened array of only the valid, quantized depths
    valid_quantized_depths = quantized_depth[valid_range_mask]

    # 3. Count points per bin
    # bincount is a very efficient GPU histogram-like function
    bin_counts = cp.bincount(valid_quantized_depths)

    # 4. Create a mask for the bins themselves, based on the point threshold
    # The size of this mask will be equal to the number of depth bins
    valid_bin_mask = bin_counts >= min_points_threshold

    # Handle the case where bin 0 is not present, bincount will return an array without it
    # We create a placeholder mask and fill it correctly
    num_bins = int(cp.ceil((max_depth - min_depth) / step_size))
    bin_validity_lookup = cp.zeros(num_bins, dtype=cp.bool_)
    
    if valid_bin_mask.shape[0] > 0:
        bin_indices = cp.arange(valid_bin_mask.shape[0])
        bin_validity_lookup[bin_indices] = valid_bin_mask
    
    # 5. Map the bin-validity mask back to the original depth map
    # Create a mask of the same size as the original depth map, initialized to False
    final_mask = cp.zeros_like(depth_map, dtype=cp.bool_)
    
    # Use the quantized depths as indices into the bin validity lookup table
    # Only apply this lookup for the points within the original valid range
    final_mask[valid_range_mask] = bin_validity_lookup[quantized_depth[valid_range_mask]]

    return final_mask