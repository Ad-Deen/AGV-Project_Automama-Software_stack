import cv2
import numpy as np
import os
import glob

def analyze_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    overexposed = np.sum(gray > 240) / gray.size * 100
    underexposed = np.sum(gray < 15) / gray.size * 100

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / gray.size * 100

    return {
        'brightness': mean_brightness,
        'contrast': contrast,
        'overexposed_pct': overexposed,
        'underexposed_pct': underexposed,
        'lap_var': laplacian_var,
        'edge_density_pct': edge_density
    }

def evaluate_stereo_folder(left_path, right_path, pattern="*.png"):
    left_images = sorted(glob.glob(os.path.join(left_path, pattern)))
    right_images = sorted(glob.glob(os.path.join(right_path, pattern)))

    assert len(left_images) == len(right_images), "Mismatched stereo pair count!"

    print(f"Found {len(left_images)} stereo pairs.\n")

    for i, (l_path, r_path) in enumerate(zip(left_images, right_images)):
        img_left = cv2.imread(l_path)
        img_right = cv2.imread(r_path)

        stats_left = analyze_image(img_left)
        stats_right = analyze_image(img_right)

        print(f"=== Stereo Pair {i + 1} ===")
        print(f"Left Image:  {os.path.basename(l_path)}")
        for k, v in stats_left.items():
            print(f"  {k:<18}: {v:.2f}")
        print(f"Right Image: {os.path.basename(r_path)}")
        for k, v in stats_right.items():
            print(f"  {k:<18}: {v:.2f}")
        print("-" * 40)

# ---- USAGE ----
# Update these paths to your stereo image directories
left_folder = "automama/perception/stereo_vision_test/single_cam_KD_callib/right_stereo/"
right_folder = "automama/perception/stereo_vision_test/single_cam_KD_callib/left_stereo/"

evaluate_stereo_folder(left_folder, right_folder)
