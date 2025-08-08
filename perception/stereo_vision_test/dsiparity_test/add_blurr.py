import os
import cv2
import numpy as np

def apply_gaussian_blur(image, ksize=(9, 9), sigma=5):
    """
    Apply Gaussian blur to the input image.
    Args:
        image: Input BGR image.
        ksize: Kernel size (must be odd integers).
        sigma: Standard deviation for Gaussian kernel.
    Returns:
        Blurred image.
    """
    return cv2.GaussianBlur(image, ksize, sigma)

# === CONFIGURATION ===
image_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"

# Image filenames
left_name = "left_stereo/left_000.png"
right_name = "right_stereo/right_000.png"

# Construct full paths
left_img_path = os.path.join(image_dir, left_name)
right_img_path = os.path.join(image_dir, right_name)

# Load images
left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)

if left_img is None or right_img is None:
    raise FileNotFoundError("One or both input images not found!")

# === APPLY BLUR ===
blurred_left = apply_gaussian_blur(left_img, ksize=(13, 13), sigma=10)  #0.85 Sigma and 7,7 ksize equalizes blur calc
blurred_right = apply_gaussian_blur(right_img, ksize=(0, 0), sigma=2)

# === SAVE IMAGES ===
left_blurred_path = os.path.join(image_dir, "left_stereo/blur_left.png")
right_blurred_path = os.path.join(image_dir, "right_stereo/blur_right.png")
# left_blurred_path = os.path.join(image_dir, "left_stereo/left_000.png")
# right_blurred_path = os.path.join(image_dir, "right_stereo/right_000.png")

cv2.imwrite(left_blurred_path, blurred_left)
cv2.imwrite(right_blurred_path, blurred_right)

print(f"Saved blurred left image to: {left_blurred_path}")
print(f"Saved blurred right image to: {right_blurred_path}")
