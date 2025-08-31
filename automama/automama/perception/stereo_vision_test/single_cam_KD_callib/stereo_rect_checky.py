import cv2
import numpy as np
import os
import yaml
from glob import glob

# Load camera calibration data
calib_dir = "automama/perception/stereo_vision_test/single_cam_KD_callib"

# Load left camera data (intrinsics and distortion coefficients)
with open(os.path.join(calib_dir, "camera_calibration1.yaml")) as f:
    left_data = yaml.safe_load(f)

# Load right camera data (intrinsics and distortion coefficients)
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
with open(os.path.join(calib_dir, "stereo_calibration.yaml")) as f:
    data = yaml.safe_load(f)

# Extract the rotation matrix R and translation vector T between cameras
# These should be the extrinsic parameters from stereo calibration
R = np.array(data["R"], dtype=np.float64)  # Rotation matrix between cameras
T = np.array(data["T"], dtype=np.float64)  # Translation vector between cameras

# Ensure R is 3x3 and T is 3x1
if R.shape != (3, 3):
    R = np.reshape(R, (3, 3))
if T.shape != (3, 1):
    T = np.reshape(T, (3, 1))

print("Camera Extrinsics loaded")

# Define image size (width, height)
image_size = (1280, 736)  # Replace with your actual image size if different

# Stereo rectification - Note the correct parameter order
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1,      # Left camera matrix and distortion
    K2, D2,      # Right camera matrix and distortion
    image_size,  # Image size
    R, T,        # Rotation and translation between cameras
    flags=cv2.CALIB_ZERO_DISPARITY,  # Optional: flags for rectification
    alpha=0      # Optional: alpha parameter (0-1, controls cropping)
)

# Compute rectification maps for both cameras
map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2) #(Source already undistorted)
map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2 , R2, P2, image_size, cv2.CV_16SC2)

# Save rectification maps as YAML files
rectification_file = os.path.join(calib_dir, "rectification_maps.yaml")
fs = cv2.FileStorage(rectification_file, cv2.FILE_STORAGE_WRITE)
fs.write("map1_left", map1_left)
fs.write("map2_left", map2_left)
fs.write("map1_right", map1_right)
fs.write("map2_right", map2_right)
fs.release()  # Close the file storage after use


print("Rectification maps saved to:", rectification_file)

# Directories with the input images
left_images_dir = "automama/perception/stereo_vision_test/single_cam_KD_callib/left_undistorted"
right_images_dir = "automama/perception/stereo_vision_test/single_cam_KD_callib/right_undistorted"

# List all PNG images in the directories
left_images = sorted(glob(os.path.join(left_images_dir, "*.png")))
right_images = sorted(glob(os.path.join(right_images_dir, "*.png")))

# Rectify and show images
for left_img_path, right_img_path in zip(left_images, right_images):
    # Read the images
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        print(f"Could not load images: {left_img_path}, {right_img_path}")
        continue
    
    # Undistort and rectify the images using remap
    rectified_left = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
    
    # Resize both images to match (if necessary)
    rectified_left = cv2.resize(rectified_left, image_size)
    rectified_right = cv2.resize(rectified_right, image_size)
    
    # Alpha blending the rectified images (50% opacity)
    blended = cv2.addWeighted(rectified_left, 0.5, rectified_right, 0.5, 0)
    
    # Show the blended result
    cv2.imshow(f"Blended Rectified Stereo Pair ({os.path.basename(left_img_path)})", blended)
    
    # Wait for a key press before moving to the next image
    cv2.waitKey(0)

# cv2.destroyAllWindows()
print("Rectification and Display Complete!")