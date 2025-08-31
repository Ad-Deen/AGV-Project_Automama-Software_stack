import cv2
import numpy as np
import glob
import os
import yaml

# -------------------------
# Calibration pattern config
# -------------------------
chessboard_size = (8, 6)
square_size = 0.025  # in meters

# -------------------------
# Load camera intrinsics from YAML files
# -------------------------
calib_dir = "automama/perception/stereo_vision_test/single_cam_KD_callib"
with open(os.path.join(calib_dir, "camera_calibration1.yaml")) as f:
    left_data = yaml.safe_load(f)
with open(os.path.join(calib_dir, "camera_calibration0.yaml")) as f:
    right_data = yaml.safe_load(f)

K1 = np.array(left_data["camera_matrix"])
D1 = np.array(left_data["distortion_coefficients"])
K2 = np.array(right_data["camera_matrix"])
D2 = np.array(right_data["distortion_coefficients"])
print("Camera Intrinsics loaded")
# -------------------------
# Load stereo image pairs
# -------------------------
left_images = sorted(glob.glob(os.path.join(calib_dir, "left_undistorted/*.png")))
right_images = sorted(glob.glob(os.path.join(calib_dir, "right_undistorted/*.png")))
assert len(left_images) == len(right_images), "Mismatch in number of stereo images"
print("stereo iamge pairs loaded")
# -------------------------
# Detect chessboard corners
# -------------------------
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_left = []
imgpoints_right = []

img_shape = None
print("detecting chess board corners")
for left_path, right_path in zip(left_images, right_images):
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)

    if imgL is None or imgR is None:
        print(f"Failed to read: {left_path} or {right_path}")
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = grayL.shape[::-1]

    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size)

    if retL and retR:
        objpoints.append(objp)

        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

        imgpoints_left.append(cornersL)
        imgpoints_right.append(cornersR)
        print(f"Chessboard found in {os.path.basename(left_path)} / {os.path.basename(right_path)}")
    else:
        print(f"Chessboard not found in {os.path.basename(left_path)} / {os.path.basename(right_path)}")

# -------------------------
# Stereo calibration
# -------------------------
print("Stereo callibration started")
if len(objpoints) < 5:
    raise RuntimeError("Not enough valid stereo detections for calibration")
print(f"object points gathered {len(objpoints)}")
print(f"imagepoints left points gathered {len(imgpoints_left)}")
print(f"imagepoints right gathered {len(imgpoints_right)}")
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
print("stereo callibrate begins .. . .")
ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    K1, D1, K2, D2,
    img_shape, flags=flags, criteria=criteria
)
print("stereo callibrate done .. . .")
# -------------------------
# Stereo rectification
# -------------------------
print("stereo rectification started .. . .")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1, K2, D2, img_shape, R, T, flags=cv2.CALIB_ZERO_DISPARITY
)
print("stereo rectification done .. . .")
baseline = np.linalg.norm(T)
print(f"Calibration successful. Baseline: {baseline:.4f} meters")

# -------------------------
# Save calibration results
# -------------------------
stereo_calib_path = os.path.join(calib_dir, "stereo_calibration.yaml")
with open(stereo_calib_path, "w") as f:
    yaml.dump({
        "R": R.tolist(),
        "T": T.tolist(),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "image_width": img_shape[0],
        "image_height": img_shape[1],
        "baseline": baseline
    }, f)

print(f"Stereo calibration data saved to: {stereo_calib_path}")
