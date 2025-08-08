import os
import cv2
import numpy as np

def compute_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def compute_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return np.mean(gray)

def compute_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return np.std(gray)

def compute_texture_strength(image):
    """
    Computes texture richness by summing gradient magnitudes.
    High texture = better stereo matching.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return np.mean(magnitude)

# === CONFIGURATION ===
image_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"
left_img_path = os.path.join(image_dir, "left_stereo/left_000.png")
right_img_path = os.path.join(image_dir, "right_stereo/right_000.png")

# === LOAD IMAGES ===
left_img = cv2.imread(left_img_path)
right_img = cv2.imread(right_img_path)

if left_img is None or right_img is None:
    raise FileNotFoundError("Could not load one or both images. Check your file paths.")

# === ANALYSIS METRICS ===
def analyze_image(image, label):
    blur = compute_blurriness(image)
    bright = compute_brightness(image)
    contrast = compute_contrast(image)
    texture = compute_texture_strength(image)
    print(f"--- {label} Image ---")
    print(f"Blurriness       : {blur:.2f}")
    print(f"Mean Brightness  : {bright:.2f}")
    print(f"Contrast (StdDev): {contrast:.2f}")
    print(f"Texture Strength : {texture:.2f}")
    print()

analyze_image(left_img, "Left")
analyze_image(right_img, "Right")
