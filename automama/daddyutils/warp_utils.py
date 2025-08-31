import os
import yaml
import numpy as np
import cv2
import vpi


def load_and_generate_warp_maps(calib_dir):
    with open(os.path.join(calib_dir, "camera_calibration1.yaml")) as f:
        left_data = yaml.safe_load(f)
    with open(os.path.join(calib_dir, "camera_calibration0.yaml")) as f:
        right_data = yaml.safe_load(f)

    K1 = np.array(left_data["camera_matrix"], dtype=np.float64)
    D1 = np.array(left_data["distortion_coefficients"], dtype=np.float64)
    K2 = np.array(right_data["camera_matrix"], dtype=np.float64)
    D2 = np.array(right_data["distortion_coefficients"], dtype=np.float64)

    D1 = np.zeros(5) if D1.size != 5 else D1
    D2 = np.zeros(5) if D2.size != 5 else D2

    with open(os.path.join(calib_dir, "stereo_calibration.yaml")) as f:
        stereo_data = yaml.safe_load(f)

    R = np.array(stereo_data["R"], dtype=np.float64).reshape((3, 3))
    T = np.array(stereo_data["T"], dtype=np.float64).reshape((3, 1))

    image_size = (640, 480)
    R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0)
    print(P2)
    print(P1)
    print(Q)
    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    warpl = vpi.WarpMap(vpi.WarpGrid(image_size))
    warpr = vpi.WarpMap(vpi.WarpGrid(image_size))

    wxl, wyl = np.asarray(warpl).transpose(2, 1, 0)
    wxr, wyr = np.asarray(warpr).transpose(2, 1, 0)

    wxl[:] = map1_left.transpose(1, 0)
    wyl[:] = map2_left.transpose(1, 0)
    wxr[:] = map1_right.transpose(1, 0)
    wyr[:] = map2_right.transpose(1, 0)

    return warpl, warpr , Q
