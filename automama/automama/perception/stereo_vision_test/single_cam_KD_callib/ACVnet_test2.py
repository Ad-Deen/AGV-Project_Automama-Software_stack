import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import glob
import os
import time

# Constants
focal_length_px = 1270.0  # from camera intrinsic matrix (K[0, 0])
baseline_meters = 0.095   # e.g., 9.5 cm
# engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib/models/fast_acvnet_736_1280.engine"
engine_path = "automama/perception/stereo_vision_test/single_cam_KD_callib/models/cgi_stereo_kitti_480x640.engine"

left_folder = "automama/perception/stereo_vision_test/single_cam_KD_callib/left_stereo/"
right_folder = "automama/perception/stereo_vision_test/single_cam_KD_callib/right_stereo/"

# Your existing preprocessing functions here (leftpreprocess, rightpreprocess)...

def leftpreprocess(image_path):
    img = cv2.imread(image_path)
    print(f"Left image shape: {img.shape}")
    img = cv2.resize(img, (640, 480))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return np.expand_dims(img, axis=0).astype(np.float32)

def rightpreprocess(image_path):
    img = cv2.imread(image_path)
    print(f"Right image shape: {img.shape}")
    img = cv2.resize(img, (640, 480))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    return np.expand_dims(img, axis=0).astype(np.float32)

# Load TRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

input_names = []
output_names = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_names.append(name)
    else:
        output_names.append(name)

for name in input_names:
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    shape = context.get_tensor_shape(name)
    size = int(np.prod(shape))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    inputs.append((host_mem, device_mem))
    context.set_tensor_address(name, device_mem)

for name in output_names:
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    shape = context.get_tensor_shape(name)
    size = int(np.prod(shape))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    outputs.append((host_mem, device_mem))
    context.set_tensor_address(name, device_mem)

# Get sorted lists of images
left_images = sorted(glob.glob(os.path.join(left_folder, "*.png")))
right_images = sorted(glob.glob(os.path.join(right_folder, "*.png")))

# Sanity check for pairs
assert len(left_images) == len(right_images), "Left and right folder image counts differ!"

# Loop over all stereo pairs
for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    print(f"\nProcessing pair {i+1}:")
    print(f" Left: {left_path}")
    print(f" Right: {right_path}")

    left_img = leftpreprocess(left_path)
    right_img = rightpreprocess(right_path)

    # Copy data to host memory
    np.copyto(inputs[0][0], left_img.ravel())
    np.copyto(inputs[1][0], right_img.ravel())
    # Start timing before inference
    start_time = time.perf_counter()  # <-- added
    # Copy to device and execute inference
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    cuda.memcpy_htod_async(inputs[1][1], inputs[1][0], stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()

    # End timing after inference
    end_time = time.perf_counter()  # <-- added

    # Print inference time
    inference_time_ms = (end_time - start_time) * 1000  # <-- added
    print(f"Inference Time: {inference_time_ms:.2f} ms")  # <-- added
    # Extract and normalize disparity map
    disparity_map = outputs[0][0].reshape((1, 1, 480, 640))[0, 0, :, :]
    disp_vis = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min() + 1e-6)
    disp_vis = (disp_vis * 255).astype(np.uint8)
    disp_vis = cv2.resize(disp_vis, (640, 480))

    # Visualize disparity map
    colored_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    cv2.imshow("Disparity Map Pair", colored_disp)

    # Wait for key, press 'q' to quit early
    key = cv2.waitKey(1000) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
