import numpy as np
import cv2
import os
import yaml

def save_calibration_to_yaml(filename, M1, D1, M2, D2, R, T, E, F, R1, R2, P1, P2, Q):
    """
    Saves the stereo calibration matrices to a YAML file.
    """
    data = {
        'M1': M1.tolist(),
        'D1': D1.tolist(),
        'M2': M2.tolist(),
        'D2': D2.tolist(),
        'R': R.tolist(),
        'T': T.tolist(),
        'E': E.tolist(),
        'F': F.tolist(),
        'R1': R1.tolist(),
        'R2': R2.tolist(),
        'P1': P1.tolist(),
        'P2': P2.tolist(),
        'Q': Q.tolist()
    }
    try:
        with open(filename, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)
        print(f"\nCalibration data saved to {filename}")
    except Exception as e:
        print(f"Error saving calibration to YAML: {e}")

# --- INPUT & OUTPUT FILE PATHS ---
input_data_file = 'calibration_data/stereo_calibration_points.npz'
output_yaml_file = 'stereo_calibration_results.yaml' # The final calibration matrices

# --- LOAD COLLECTED DATA ---
print(f"Loading calibration data from: {input_data_file}")
try:
    with np.load(input_data_file) as data:
        objpoints = data['objpoints']
        imgpoints_left = data['imgpoints_left']
        imgpoints_right = data['imgpoints_right']
        img_shape = tuple(data['img_shape']) # Convert back to tuple
        # checkerboard_size = tuple(data['checkerboard_size']) # If you need to verify
        # square_size = data['square_size'] # If you need to verify
    
    # Ensure points are in the correct format (list of numpy arrays)
    objpoints = [p for p in objpoints]
    imgpoints_left = [p for p in imgpoints_left]
    imgpoints_right = [p for p in imgpoints_right]

except FileNotFoundError:
    print(f"Error: Calibration data file '{input_data_file}' not found.")
    print("Please run 'capture_and_detect.py' first to generate the data.")
    exit()
except Exception as e:
    print(f"Error loading data from '{input_data_file}': {e}")
    exit()

print(f"Loaded {len(objpoints)} image pairs for calibration.")
print(f"Image resolution: {img_shape[1]}x{img_shape[0]} pixels.")

# --- MONOCULAR CALIBRATION (for initial K, D guesses for stereo) ---
print("\n--- Performing Monocular Calibration for each camera ---")

# Termination criteria for monocular calibration
criteria_mono = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Calibrate Right Camera (M1, D1)
print("\nCalibrating Right Camera (M1, D1)...")
# Note: img_shape[::-1] is (width, height) as required by OpenCV functions
ret_mono_right, M1, D1, rvecs_right_mono, tvecs_right_mono = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_shape[::-1], None, None, criteria=criteria_mono
)
print("Right Camera Monocular RMS error:", ret_mono_right)
print("Right Camera Matrix (M1):\n", M1)
print("Right Distortion Coefficients (D1):\n", D1)

# Calibrate Left Camera (M2, D2)
print("\nCalibrating Left Camera (M2, D2)...")
ret_mono_left, M2, D2, rvecs_left_mono, tvecs_left_mono = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_shape[::-1], None, None, criteria=criteria_mono
)
print("Left Camera Monocular RMS error:", ret_mono_left)
print("Left Camera Matrix (M2):\n", M2)
print("Left Distortion Coefficients (D2):\n", D2)

# --- STEREO CALIBRATION ---
print("\n--- Performing Stereo Calibration ---")
print("This step refines intrinsics (M, D) and computes extrinsics (R, T, E, F).")

# Flags for stereoCalibrate:
# cv2.CALIB_USE_INTRINSIC_GUESS: Use the M1, D1, M2, D2 from monocular as initial guess.
# cv2.CALIB_ZERO_TANGENT_DIST: Assumes tangential distortion coefficients are zero and fixes them.
# cv2.CALIB_FIX_PRINCIPAL_POINT: Fixes the principal point at the center.
# For a robust calibration, `CALIB_USE_INTRINSIC_GUESS` with `CALIB_ZERO_TANGENT_DIST` is often a good start.
# If your cameras have significant tangential distortion, remove `CALIB_ZERO_TANGENT_DIST`.
flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST

# Termination criteria for stereo calibration optimization
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

# Perform stereo calibration
# Note: The order of imgpoints and camera matrices here (imgpoints_right, imgpoints_left and M1, D1, M2, D2)
# means that R and T describe the transformation from the 'right' camera's frame to the 'left' camera's frame.
ret_stereo, M1, D1, M2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_right, imgpoints_left, # image points for camera1 (right), then camera2 (left)
    M1, D1, M2, D2, # Initial intrinsic guesses (from monocular)
    img_shape[::-1], # Image size (width, height)
    criteria_stereo,
    flags
)
print("Stereo Calibration RMS error:", ret_stereo)

print("\n--- Final Calibration Results ---")
print("Right Camera Matrix (M1):\n", M1)
print("Right Distortion Coefficients (D1):\n", D1)
print("\nLeft Camera Matrix (M2):\n", M2)
print("Left Distortion Coefficients (D2):\n", D2)
print("\nRotation Matrix (R) from Right Camera to Left Camera:\n", R)
print("Translation Vector (T) from Right Camera to Left Camera:\n", T)
print("Essential Matrix (E):\n", E)
print("Fundamental Matrix (F):\n", F)

# Calculate baseline (magnitude of translation vector)
baseline_meters = np.linalg.norm(T)
print(f"\nCalculated Baseline: {baseline_meters * 100:.2f} cm")

# --- STEREO RECTIFICATION ---
print("\n--- Performing Stereo Rectification ---")
# This calculates R1, R2, P1, P2, Q for rectified images.
# alpha=0: All black pixels if some pixels from the original image cannot be mapped (maximum zoom, minimum valid area).
# alpha=1: All original pixels are preserved, but a larger image might be returned (minimum zoom, maximum valid area).
# newImageSize should typically be the same as img_shape[::-1] for consistency.
R1, R2, P1, P2, Q, roi_right, roi_left = cv2.stereoRectify(
    M1, D1, M2, D2,
    img_shape[::-1], # Image size (width, height)
    R, T,
    alpha=0,
    newImageSize=(img_shape[1], img_shape[0])
)

print("\nRectification Transform (R1) for Right Camera:\n", R1)
print("New Projection Matrix (P1) for Right Camera:\n", P1)
print("\nRectification Transform (R2) for Left Camera:\n", R2)
print("New Projection Matrix (P2) for Left Camera:\n", P2)
print("\nDisparity-to-Depth Mapping Matrix (Q):\n", Q)

# --- SAVE ALL MATRICES TO YAML ---
save_calibration_to_yaml(output_yaml_file, M1, D1, M2, D2, R, T, E, F, R1, R2, P1, P2, Q)

print("\nFull stereo calibration process completed successfully!")
print(f"Calibration data saved to '{output_yaml_file}'")