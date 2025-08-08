import os
import cv2
import numpy as np

def enhance_features_contrast(image):
    # Convert to LAB color space to better manipulate contrast/brightness
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge channels and convert back to BGR
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Sharpen image using unsharp masking
    blurred = cv2.GaussianBlur(enhanced, (0,0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

    return sharpened

# === CONFIGURATION ===
image_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"
right_name = "right_stereo/right_000.png"
right_img_path = os.path.join(image_dir, right_name)

right_img = cv2.imread(right_img_path)
if right_img is None:
    raise FileNotFoundError("Right image not found!")

enhanced_right = enhance_features_contrast(right_img)

output_path = os.path.join(image_dir, "right_stereo/enhanced_right.png")
cv2.imwrite(output_path, enhanced_right)
print(f"Saved enhanced right image to: {output_path}")
