import cv2
import numpy as np
import os
import yaml

# Paths to calibration
calib_dir = "automama/perception/stereo_vision_test/single_cam_KD_callib"
with open(os.path.join(calib_dir, "camera_calibration1.yaml")) as f:
    left_data = yaml.safe_load(f)
with open(os.path.join(calib_dir, "camera_calibration0.yaml")) as f:
    right_data = yaml.safe_load(f)
with open(os.path.join(calib_dir, "stereo_calibration.yaml")) as f:
    stereo_data = yaml.safe_load(f)

# Intrinsics
K1 = np.array(left_data["camera_matrix"], dtype=np.float64)
D1 = np.array(left_data["distortion_coefficients"], dtype=np.float64)
K2 = np.array(right_data["camera_matrix"], dtype=np.float64)
D2 = np.array(right_data["distortion_coefficients"], dtype=np.float64)

# Extrinsics
R_base = np.array(stereo_data["R"], dtype=np.float64)
T_base = np.array(stereo_data["T"], dtype=np.float64).reshape(3, 1)

# Helper: convert angles (in degrees) to rotation matrix
def euler_to_rotmat(rx, ry, rz):
    r = np.radians([rx, ry, rz])
    Rx = cv2.Rodrigues(np.array([r[0], 0, 0]))[0]
    Ry = cv2.Rodrigues(np.array([0, r[1], 0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, r[2]]))[0]
    return Rz @ Ry @ Rx

def save_stereo_calibration(R_adj, T_adj, path,K1,K2,D1,D2):

    
    image_size = (1280, 720)

    # Compute stereo rectification and projection matrices
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        cameraMatrix1=K1, distCoeffs1=D1,
        cameraMatrix2=K2, distCoeffs2=D2,
        imageSize=image_size,
        R=R_adj, T=T_adj,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # Assemble final dictionary in correct format
    stereo_tuned = {
        "P1": P1.tolist(),
        "P2": P2.tolist(),
        "Q": Q.tolist(),
        "R": R_adj.tolist(),
        "R1": R1.tolist(),
        "R2": R2.tolist(),
        "T": T_adj.tolist(),
        "image_width": image_size[0],
        "image_height": image_size[1]
    }

    # Save to YAML
    with open(path, 'w') as f:
        yaml.dump(stereo_tuned, f, sort_keys=False)
    print(f"Saved tuned stereo calibration to {path}")


# GStreamer video source
def gstreamer_pipeline(sensor_id, flip_method=2, width=1280, height=720):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate=30/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )

# Open camera
cap_left = cv2.VideoCapture(gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
cap_right = cv2.VideoCapture(gstreamer_pipeline(0), cv2.CAP_GSTREAMER)

# Image size
image_size = (1280, 720)

# Offset parameters
rx, ry, rz = 0.0, 0.0, 0.0  # rotation offsets in degrees
tx, ty, tz = 0.0, 0.0, 0.0  # translation offsets in meters

print("Use keys to adjust stereo offset:")
print("r/f = roll, t/g = pitch, y/h = yaw")
print("i/k = X, j/l = Y, u/o = Z")
print("Press SPACE to reset, q to save & quit")

# Main loop
while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()
    if not ret1 or not ret2:
        break

    # Build R, T with offsets
    R_delta = euler_to_rotmat(rx, ry, rz)
    R_adj = R_delta @ R_base
    T_adj = T_base + np.array([[tx], [ty], [tz]])

    # Stereo rectify
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R_adj, T_adj,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )

    # Remap
    map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    rect_l = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

    blend = cv2.addWeighted(rect_l, 0.7, rect_r, 0.7, 0)
    blend = cv2.resize(blend, (int(image_size[0]*0.6), int(image_size[1]*0.6)))
    cv2.imshow("Rectified Blend left", blend)
    # cv2.imshow("Rectified Blend left", rect_l)
    # cv2.imshow("Rectified Blend right", rect_r)

    key = cv2.waitKey(1) & 0xFF
    step_rot = 0.1  # degrees
    step_trans = 0.01  # meters

    if key == ord('r'): rx += step_rot
    elif key == ord('f'): rx -= step_rot
    elif key == ord('t'): ry += step_rot
    elif key == ord('g'): ry -= step_rot
    elif key == ord('y'): rz += step_rot
    elif key == ord('h'): rz -= step_rot
    elif key == ord('i'): tx += step_trans
    elif key == ord('k'): tx -= step_trans
    elif key == ord('j'): ty += step_trans
    elif key == ord('l'): ty -= step_trans
    elif key == ord('u'): tz += step_trans
    elif key == ord('o'): tz -= step_trans
    elif key == 32:  # spacebar
        rx = ry = rz = tx = ty = tz = 0.0
    elif key == ord('q'):
        save_path = os.path.join(calib_dir, "stereo_calibration_tuned.yaml")
        save_stereo_calibration(R_adj, T_adj, save_path,K1,K2,D1,D2)
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
