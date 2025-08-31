import cv2
import numpy as np
import os
import yaml
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ---------- Paths ----------
calib_dir = "automama/perception/stereo_vision_test/single_cam_KD_callib"

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

# Load stereo calibration data (extrinsics)
with open(os.path.join(calib_dir, "stereo_calibration.yaml")) as f:
    data = yaml.safe_load(f)

R = np.array(data["R"], dtype=np.float64)  # Rotation matrix between cameras
T = np.array(data["T"], dtype=np.float64)  # Translation vector between cameras

# Define image size (width, height)
image_size = (1280, 736)

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1,      # Right camera matrix and distortion
    K2, D2,      # Left camera matrix and distortion
    image_size,  # Image size
    R, T,        # Rotation and translation between cameras
    flags=cv2.CALIB_ZERO_DISPARITY,  # Optional: flags for rectification
    alpha=0      # Optional: alpha parameter (0-1, controls cropping)
)

# Compute rectification maps for both cameras
map1_left, map2_left = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)
map1_right, map2_right = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)

# ---------- GStreamer Pipeline ----------
def gstreamer_pipeline(sensor_id, flip_method=2, width=1280, height=736):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate=1/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink sync=true"
    )

# Create the capture objects for both cameras
flip_method = 2
cap_left = cv2.VideoCapture(gstreamer_pipeline(0, flip_method), cv2.CAP_GSTREAMER)
cap_right = cv2.VideoCapture(gstreamer_pipeline(1, flip_method), cv2.CAP_GSTREAMER)

# ---------- Load TensorRT Engine for Depth Estimation ----------
# --- Constants (use real calibration values) ---
focal_length_px = 1000.0  # from camera intrinsic matrix (K[0, 0])
baseline_meters = np.linalg.norm(T)
# engine_path = "automama/perception/stereo_vision_test/single_cam_KD_callib/models/fast_acvnet_736_1280.engine"
engine_path = "automama/perception/stereo_vision_test/single_cam_KD_callib/models/cgi_stereo_kitti_480x640.engine"
# --- Load TensorRT engine ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# --- Allocate memory for inputs and outputs ---
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

# Get tensor names
input_names = []
output_names = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_names.append(name)
    else:
        output_names.append(name)

# Allocate memory for inputs
for name in input_names:
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    shape = context.get_tensor_shape(name)
    size = int(np.prod(shape))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    inputs.append((host_mem, device_mem))
    context.set_tensor_address(name, device_mem)

# Allocate memory for outputs
for name in output_names:
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    shape = context.get_tensor_shape(name)
    size = int(np.prod(shape))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    outputs.append((host_mem, device_mem))
    context.set_tensor_address(name, device_mem)

# --- Preprocess function ---
def preprocess_from_array(img):
    img = cv2.resize(img, (640, 480))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return np.expand_dims(img, axis=0).astype(np.float32)  # [1,3,736,1280]

# ---------- Stream and Show Rectified Frames in Real-Time ----------
while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()
    
    if not ret1 or not ret2:
        print("Failed to grab frames.")
        break

    # Apply the rectification maps to undistort and rectify frames
    undist_left = cv2.remap(frame_left, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
    undist_right = cv2.remap(frame_right, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)

    # Preprocess the stereo images for depth estimation
    left_img = preprocess_from_array(undist_left)
    right_img = preprocess_from_array(undist_right)

    np.copyto(inputs[0][0], left_img.ravel())
    np.copyto(inputs[1][0], right_img.ravel())

    # --- Run inference ---
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    cuda.memcpy_htod_async(inputs[1][1], inputs[1][0], stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

    # --- Extract and normalize disparity map ---
    disparity_map = outputs[0][0].reshape((1, 1, 480, 640))[0, 0, :, :]
    disp_vis = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min() + 1e-6)
    disp_vis = (disp_vis * 255).astype(np.uint8)

    # --- Compute depth map ---
    # depth_map = (focal_length_px * baseline_meters) / (disparity_map + 1e-6)
    # depth_vis = (np.clip(depth_map / 10.0, 0, 1) * 255).astype(np.uint8)  # Normalize for visualization

    # --- Visualize disparity and depth maps ---
    colored_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    # colored_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("Disparity Map", colored_disp)
    # cv2.imshow("Colored Depth Map", colored_depth)

    # Exit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video captures and close the windows
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
