import cv2
import numpy as np
import os
import yaml

# ---------- Paths ----------
calib_dir = "automama/perception/stereo_vision_test/single_cam_KD_callib"
left_save_dir = os.path.join(calib_dir, "left_stereo")
right_save_dir = os.path.join(calib_dir, "right_stereo")
os.makedirs(left_save_dir, exist_ok=True)
os.makedirs(right_save_dir, exist_ok=True)
# Load camera calibration data
with open(os.path.join(calib_dir, "camera_calibration1.yaml")) as f:
    left_data = yaml.safe_load(f)

with open(os.path.join(calib_dir, "camera_calibration0.yaml")) as f:
    right_data = yaml.safe_load(f)

# Extract intrinsics and distortion coefficients for both cameras
K1 = np.array(left_data["camera_matrix"], dtype=np.float64)
D1 = np.array(left_data["distortion_coefficients"], dtype=np.float64)
K2 = np.array(right_data["camera_matrix"], dtype=np.float64)
D2 = np.array(right_data["distortion_coefficients"], dtype=np.float64)

# Ensure distortion coefficients are 1D arrays of size 5 (k1, k2, p1, p2, k3)
if D1.size != 5:
    D1 = np.zeros(5)  # Replace with zeros or your known distortion parameters
if D2.size != 5:
    D2 = np.zeros(5)  # Replace with zeros or your known distortion parameters

print("Camera Intrinsics loaded")

# Load stereo calibration data (extrinsics)
# with open(os.path.join(calib_dir, "stereo_calibration_tuned.yaml")) as f:   #fined tuned matrix
with open(os.path.join(calib_dir, "stereo_calibration.yaml")) as f:   #Raw matrix
    data = yaml.safe_load(f)

# Extract the rotation matrix R and translation vector T between cameras
R = np.array(data["R"], dtype=np.float64)  # Rotation matrix between cameras
T = np.array(data["T"], dtype=np.float64)  # Translation vector between cameras

# Ensure R is 3x3 and T is 3x1
if R.shape != (3, 3):
    R = np.reshape(R, (3, 3))
if T.shape != (3, 1):
    T = np.reshape(T, (3, 1))

print("Camera Extrinsics loaded")

# Define image size (width, height)
image_size = (640, 480)  # Replace with your actual image size if different

# Stereo rectification - Note the correct parameter order
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1,      # Left camera matrix and distortion
    K2, D2,      # Right camera matrix and distortion
    image_size,  # Image size
    R, T,        # Rotation and translation between cameras
    flags=cv2.CALIB_ZERO_DISPARITY,  # Optional: flags for rectification
    alpha=0      # Optional: alpha parameter (0-1, controls cropping)
)

# Compute rectification maps for both cameras 2 -> left , 1-> right
map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2 , R2, P2, image_size, cv2.CV_16SC2)

# ---------- GStreamer Pipeline ----------
def gstreamer_pipeline(sensor_id, flip_method=2, width=640, height=480):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate=30/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    )
# Create the capture objects for both cameras
flip_method = 2
cap_left = cv2.VideoCapture(gstreamer_pipeline(1, flip_method), cv2.CAP_GSTREAMER)
cap_right = cv2.VideoCapture(gstreamer_pipeline(0, flip_method), cv2.CAP_GSTREAMER)

frame_count = 0
# ---------- Stream and Show Rectified Frames in Real-Time ----------
while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()
    
    # if not ret1 or not ret2:
    #     print("Failed to grab frames.")
    #     break

    # Apply the rectification maps to undistort and rectify frames
    undist_left = cv2.remap(frame_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
    undist_right = cv2.remap(frame_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

    # Resize both images to match (if necessary)
    # undist_left = cv2.resize(undist_left, (image_size[0], image_size[1]))
    # undist_right = cv2.resize(undist_right, (image_size[0], image_size[1]))

    # Alpha blending the rectified images (50% opacity)
    blended = cv2.addWeighted(undist_left, 0.5, undist_right, 0.5, 0)
    blended = cv2.resize(blended, (int(image_size[0]*0.9), int(image_size[1]*0.9)))

    # Show the blended result
    cv2.imshow("Blended Rectified Stereo Pair", blended)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"{left_save_dir}/left_{frame_count:03d}.png", undist_left)
        cv2.imwrite(f"{right_save_dir}/right_{frame_count:03d}.png", undist_right)

        print(f"Saved stereo pair #{frame_count}")
        frame_count += 1
    # Exit if 'q' is pressed
    # key = cv2.waitKey(1) & 0xFF
    elif key == ord('q'):
        break

# Release the video captures and close the windows
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
