import cv2
import numpy as np
import glob
import os
import yaml

# ==== CONFIG ====
IMAGE_DIR = "automama/perception/stereo_vision_test/single_cam_KD_callib/captured_images"
CHESSBOARD_SIZE = (8, 6)
SQUARE_SIZE = 0.025
OUTPUT_DIR = "automama/perception/stereo_vision_test/single_cam_KD_callib"
CALIBRATION_YAML = os.path.join(OUTPUT_DIR, "camera_calibration.yaml")

def save_calibration_yaml(filename, camera_matrix, dist_coeffs):
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist()
    }
    with open(filename, 'w') as f:
        yaml.dump(data, f)
    print(f"Calibration saved to YAML: {filename}")

def main():
    # Prepare object points
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    images = glob.glob(os.path.join(IMAGE_DIR, '*.png')) + glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    if len(images) == 0:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Found {len(images)} images for calibration.")

    img_shape = None
    valid_images = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to read {fname}, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            valid_images += 1
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            objpoints.append(objp)
            imgpoints.append(corners2)

            img_draw = cv2.drawChessboardCorners(img.copy(), CHESSBOARD_SIZE, corners2, ret)
            cv2.imshow('Corners', img_draw)
            cv2.waitKey(100)
        else:
            print(f"Chessboard not found in {fname}, skipping.")

    cv2.destroyAllWindows()

    if valid_images < 5:
        print("Not enough valid images for calibration. Need at least 5.")
        return

    print("Calibrating camera...")
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("\nCalibration results:")
    print(f"RMS re-projection error: {ret}")
    print("Camera matrix (K):")
    print(K)
    print("Distortion coefficients (D):")
    print(D.ravel())

    # Save to YAML
    save_calibration_yaml(CALIBRATION_YAML, K, D)

if __name__ == "__main__":
    main()
