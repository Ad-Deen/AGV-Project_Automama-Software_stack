import cv2
import numpy as np
import os
import yaml

# ---------- Paths ----------
calib_dir = "automama/perception/stereo_vision_test/single_cam_KD_callib"
yaml_left = os.path.join(calib_dir, "camera_calibration1.yaml")
yaml_right = os.path.join(calib_dir, "camera_calibration0.yaml")
left_save_dir = os.path.join(calib_dir, "left_undistorted")
right_save_dir = os.path.join(calib_dir, "right_undistorted")
os.makedirs(left_save_dir, exist_ok=True)
os.makedirs(right_save_dir, exist_ok=True)

# ---------- Load Intrinsics from YAML ----------
def load_yaml_intrinsics(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    K = np.array(data['camera_matrix'])
    D = np.array(data['distortion_coefficients'][0])  # Assuming coefficients inside a nested list
    return K, D

K1, D1 = load_yaml_intrinsics(yaml_left)  
K2, D2 = load_yaml_intrinsics(yaml_right) 
print(K1)
print(K2)
print(D1)
print(D2)

# ---------- GStreamer Pipeline ----------
def gstreamer_pipeline(sensor_id, flip_method=2,width=640,height=480):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate=30/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )

flip_method = 2
cap_right = cv2.VideoCapture(gstreamer_pipeline(0, flip_method), cv2.CAP_GSTREAMER)
cap_left = cv2.VideoCapture(gstreamer_pipeline(1, flip_method), cv2.CAP_GSTREAMER)

# ---------- Grab Frame for Dimensions ----------
ret1, frame_left = cap_left.read()
ret2, frame_right = cap_right.read()
h, w = frame_left.shape[:2]

# ---------- Undistort Map ----------
map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, None, K1, (w, h), cv2.CV_32FC1)
map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, None, K2, (w, h), cv2.CV_32FC1)

# ---------- Chessboard Config ----------
pattern_size = (8, 6)
square_size = 0.025  # 25 mm
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size

# ---------- Storage Arrays ----------
objpoints = []
imgpoints_left = []
imgpoints_right = []

frame_count = 0
while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()
    if not ret1 or not ret2:
        print("Failed to grab frames.")
        break

    undist_left = cv2.remap(frame_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
    undist_right = cv2.remap(frame_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)
    disp_left = cv2.resize(undist_left,(int(1280*0.5),int(720*0.9)))
    disp_right = cv2.resize(undist_right,(int(1280*0.45),int(720*0.9)))
    display = np.hstack((disp_left, disp_right))
    cv2.imshow("Undistorted Stereo", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        grayL = cv2.cvtColor(undist_left, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(undist_right, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = cv2.findChessboardCorners(grayL, pattern_size)
        foundR, cornersR = cv2.findChessboardCorners(grayR, pattern_size)

        print(f"[{frame_count:03d}] Chessboard - Left: {foundL}, Right: {foundR}")
        if foundL and foundR:
            cv2.imwrite(f"{left_save_dir}/left_{frame_count:03d}.png", undist_left)
            cv2.imwrite(f"{right_save_dir}/right_{frame_count:03d}.png", undist_right)

            objpoints.append(objp)
            imgpoints_left.append(cornersL)
            imgpoints_right.append(cornersR)

            print(f"Saved stereo pair #{frame_count}")
            frame_count += 1
        else:
            print("Chessboard not detected in both frames.")

    elif key == ord('q'):
        break

# ---------- Save Collected Data ----------
np.savez(os.path.join(calib_dir, "stereo_calib_data.npz"),
         objpoints=objpoints,
         imgpoints_left=imgpoints_left,
         imgpoints_right=imgpoints_right)

print(f"\nSaved {frame_count} stereo pairs and chessboard points.")
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
