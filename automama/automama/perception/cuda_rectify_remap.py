import cv2
import numpy as np
import vpi
import yaml
import os

# ---------- GStreamer Pipeline ----------
def gstreamer_pipeline(sensor_id, flip_method=2, width=1280, height=736):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate=30/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink sync=true"
    )


# ---------- Load Calibration Parameters ----------
def load_yaml_params(calib_dir):
    with open(os.path.join(calib_dir, "camera_calibration0.yaml")) as f:
        left = yaml.safe_load(f)
    with open(os.path.join(calib_dir, "camera_calibration1.yaml")) as f:
        right = yaml.safe_load(f)
    with open(os.path.join(calib_dir, "stereo_calibration.yaml")) as f:
        stereo = yaml.safe_load(f)
    return left, right, stereo
def cv2_maps_to_vpi_dynamic_map(map1, map2):
    """
    Convert OpenCV map1 and map2 to VPI dynamic remap format.

    Args:
        map1: X map (float32)
        map2: Y map (float32)
    
    Returns:
        Dense flow map of shape (H, W, 2) and dtype float32.
    """
    if map1.dtype != np.float32 or map2.dtype != np.float32:
        raise ValueError("Expected CV_32FC1 maps")

    return np.stack((map1, map2), axis=2).astype(np.float32)

# /home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib/camera_calibration0.yaml
calib_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"
left_data, right_data, stereo_data = load_yaml_params(calib_dir)

K1 = np.array(left_data["camera_matrix"], dtype=np.float64)
D1 = np.array(left_data["distortion_coefficients"], dtype=np.float64)
K2 = np.array(right_data["camera_matrix"], dtype=np.float64)
D2 = np.array(right_data["distortion_coefficients"], dtype=np.float64)
R = np.array(stereo_data["R"], dtype=np.float64)
T = np.array(stereo_data["T"], dtype=np.float64).reshape(3, 1)

# Image size
image_size = (1280, 736)

# ---------- Compute Rectification Parameters ----------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

# ---------- Compute OpenCV Rectification Maps ----------
map1_l, map2_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map1_r, map2_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

warpl = vpi.WarpMap(vpi.WarpGrid(image_size))
wxl,wyl = np.asarray(warpl).transpose(2,1,0)
print((map1_l.transpose(1,0)).shape)    #(1280, 736)
print(wxl.shape)                        #(1280, 736)
print(np.asarray(warpl).shape)          #(736, 1280, 2)
wxl[:] = map1_l.transpose(1,0)
wyl[:] = map2_l.transpose(1,0)

warpr = vpi.WarpMap(vpi.WarpGrid(image_size))
wxr,wyr = np.asarray(warpr).transpose(2,1,0)
print((map1_r.transpose(1,0)).shape)    #(1280, 736)
print(wxr.shape)                        #(1280, 736)
print(np.asarray(warpr).shape)          #(736, 1280, 2)
wxr[:] = map1_r.transpose(1,0)
wyr[:] = map2_r.transpose(1,0)


# while True:
#     img = cv2.imread(imgName)
#     with vpi.Backend.CUDA:
#         output = input.remap(warp)
#         imgCorrected = vpi.asimage(img).convert(vpi.Format.NV12_ER).remap(warpl, interp=vpi.Interp.LINEAR).convert(vpi.Format.RGB8)
#     cv2.imshow("undistort_python{}{:03d}", imgCorrected.cpu())
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
