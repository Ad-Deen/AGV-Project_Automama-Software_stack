# filename: image_utils.py

import numpy as np
import cv2

def mask_red_pixels(frame, threshold=150):
    """
    Masks pixels in an RGB image frame where the red channel value is above a given threshold.

    Args:
        frame (np.ndarray): The input image frame in RGB format (H, W, 3).
        threshold (int): The red channel intensity value above which pixels will be masked.
                         (Default: 150, value range is 0-255).

    Returns:
        np.ndarray: The masked image frame where red pixels are set to black.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Input frame must be a 3-channel RGB image.")
    
    # Create a copy of the frame to avoid modifying the original
    masked_frame = frame.copy()
    
    # Extract the red channel (index 0)
    red_channel = masked_frame[:, :, 0]
    
    # Create a boolean mask where the red channel is above the threshold
    red_pixel_mask = red_channel > threshold
    
    # Stack the 2D mask to a 3D mask to apply it to all three channels
    three_channel_mask = np.stack([red_pixel_mask, red_pixel_mask, red_pixel_mask], axis=-1)
    
    # Set the pixels that meet the condition to black
    masked_frame[three_channel_mask] = 0
    
    return masked_frame