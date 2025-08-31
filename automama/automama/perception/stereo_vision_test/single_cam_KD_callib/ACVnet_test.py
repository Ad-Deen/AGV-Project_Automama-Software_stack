import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# --- Constants (use real calibration values) ---
focal_length_px = 1270.0 # from camera intrinsic matrix (K[0, 0])
baseline_meters = 0.095 # e.g., 9.5 cm
# engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib/ONNX-ACVNet-Stereo-Depth-Estimation/models/fast_acvnet.engine"
engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib/models/fast_acvnet_736_1280.engine"
left_img_path = "automama/perception/stereo_vision_test/single_cam_KD_callib/left_stereo/left_002.png"
right_img_path = "automama/perception/stereo_vision_test/single_cam_KD_callib/right_stereo/right_002.png"
# left_img_path = "automama/perception/im2.png"
# right_img_path = "automama/perception/im6.png"
# --- Preprocess function ---
def enhance_image(img, gamma=1.2, saturation_scale=1.3, sharpness_strength=1.5):
    # ----- Step 1: Gamma Correction (Brighten image) -----
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                      for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)

    # ----- Step 2: CLAHE for Contrast -----
    lab = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_contrast = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # ----- Step 3: Increase Saturation (Color Bloom) -----
    hsv = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation_scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    img_bloom = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ----- Step 4: Sharpening -----
    blurred = cv2.GaussianBlur(img_bloom, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(img_bloom, sharpness_strength, blurred, -0.5, 0)

    return sharpened

def leftpreprocess(image_path):
    img = cv2.imread(image_path)
    # img = enhance_image(img)
    print(f"image shape {img.shape}")
    cv2.imshow("img", img)
    cv2.waitKey(0)
    img = cv2.resize(img, (1280, 736))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) # CHW
    return np.expand_dims(img, axis=0).astype(np.float32) # [1,3,736,1280]

def rightpreprocess(image_path):
    img = cv2.imread(image_path)
    # img = enhance_image(img)
    print(f"image shape {img.shape}")
    cv2.imshow("img", img)
    cv2.waitKey(0)
    img = cv2.resize(img, (1280, 736))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) # CHW
    return np.expand_dims(img, axis=0).astype(np.float32) # [1,3,736,1280]
# --- Load TensorRT engine ---
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# --- Allocate memory for inputs and outputs using new tensor-based API ---
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

# --- Load and preprocess stereo images ---
left_img = leftpreprocess(left_img_path)
right_img = rightpreprocess(right_img_path)

# right_img_array = cv2.imread(left_img_path)
# right_img_shifted = shift_image_left(right_img_array)
# right_img = preprocess_from_array(right_img_shifted)
# shift_image_left
np.copyto(inputs[0][0], left_img.ravel())
np.copyto(inputs[1][0], right_img.ravel())

# --- Run inference ---
cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
cuda.memcpy_htod_async(inputs[1][1], inputs[1][0], stream)
context.execute_async_v3(stream_handle=stream.handle)
cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
stream.synchronize()

# --- Extract and normalize disparity map ---
disparity_map = outputs[0][0].reshape((1, 1, 736, 1280))[0, 0, :, :]
disp_vis = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min() + 1e-6)
disp_vis = (disp_vis * 255).astype(np.uint8)
disp_vis = cv2.resize(disp_vis, (960, 620))
# left_vis = cv2.resize(left_img, (960, 620))
# right_vis = cv2.resize(right_img, (960, 620))
# --- Visualize disparity ---
colored_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
cv2.imshow("Disparity Map", colored_disp)
# cv2.imshow("Left", left_vis)
# cv2.imshow("Right", right_vis)

# --- Compute depth map ---
# depth_map = (focal_length_px * baseline_meters) / (disparity_map + 1e-6)
# depth_vis = (np.clip(depth_map / 10.0, 0, 1) * 255).astype(np.uint8) # Normalize for visualization

# --- Create colored depth map ---
# colored_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# --- Show both disparity and depth maps ---
# cv2.imshow("Disparity Map", colored_disp)
# cv2.imshow("Colored Depth Map", colored_depth)
cv2.waitKey(0)
cv2.destroyAllWindows()