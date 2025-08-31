import numpy as np
import jetson_utils  # recommended over deprecated jetson.utils
import vpi
import cupy as cp
import cv2
import yaml
import os
import open3d as o3d
import time
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
with open(os.path.join(calib_dir, "stereo_calibration.yaml")) as f:
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
image_size = (1280, 720)  # Replace with your actual image size if different

# Stereo rectification - Note the correct parameter order
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K1, D1,      # Left camera matrix and distortion
    K2, D2,      # Right camera matrix and distortion
    image_size,  # Image size
    R, T,        # Rotation and translation between cameras
    flags=cv2.CALIB_ZERO_DISPARITY,  # Optional: flags for rectification
    alpha=0      # Optional: alpha parameter (0-1, controls cropping)
)
Q_cpmat = cp.asarray(Q)
# Compute rectification maps for both cameras 2 -> left , 1-> right
map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2 , R2, P2, image_size, cv2.CV_32FC1)
# --------------remap params setup ------------------------------------
warpl = vpi.WarpMap(vpi.WarpGrid(image_size))
warpr = vpi.WarpMap(vpi.WarpGrid(image_size))
wxl,wyl = np.asarray(warpl).transpose(2,1,0)
wxr,wyr = np.asarray(warpr).transpose(2,1,0)
# print((map1_left.transpose(1,0)).shape)    #(1280, 736)
# print(wxl.shape)                        #(1280, 736)
# print(np.asarray(warpl).shape)   
# print(image_size)       #(736, 1280, 2)
wxl[:] = map1_left.transpose(1,0)
wyl[:] = map2_left.transpose(1,0)
# print((map1_left.transpose(1,0)).shape)

wxr[:] = map1_right.transpose(1,0)
wyr[:] = map2_right.transpose(1,0)

#                             # disparity as reference
# # params
# confidenceU16 = None
# calcConf = False    #Turning off confidence 
#                     # calculations for now
# downscale = 1
# backend = 'ofa-pva-vic' #for orin ano architecture
# verbose = False     #for added printing

windowSize = 5
conf_threshold = 0.4
p1= 0
p2 = 32
p2_alpha = 2
uniq = 0
numpass = 0
diagonals = 1
numPasses = 1
minDisparity = 0
maxdisp = 256

# -------------------VPI Cuda Buffer to CuPy Interface--------------
class CudaArrayWrapper:
    def __init__(self, cuda_iface: dict):
        # Ensure required fields are tuples
        self.__cuda_array_interface__ = {
            'shape': tuple(cuda_iface['shape']),
            'strides': tuple(cuda_iface['strides']) if 'strides' in cuda_iface else None,
            'typestr': cuda_iface['typestr'],
            'data': cuda_iface['data'],
            'version': cuda_iface['version']
        }

def vpi_cuda_to_cupy_wrapper(vpi_cuda_buffer):
    """
    Converts a VPI CudaBuffer object into a CuPy-compatible wrapper.
    
    Parameters:
        vpi_cuda_buffer: VPI CudaBuffer or Image.cuda() object
    
    Returns:
        CudaArrayWrapper instance compatible with cupy.asarray()
    """
    iface = vpi_cuda_buffer.__cuda_array_interface__

    return CudaArrayWrapper(iface)


# -------------------------Cupy to VPI interface -------------------------------------
class VPIImageWrapper:
    def __init__(self, cuda_iface: dict):
        # Ensure required fields are tuples
        self.__cuda_array_interface__ = {
            'shape': list(cuda_iface['shape']),
            'strides': list(cuda_iface['strides']) if 'strides' in cuda_iface else None,
            'typestr': cuda_iface['typestr'],
            'data': cuda_iface['data'],
            'version': cuda_iface['version']
        }
# -----------------------------  Temporal Dispaity Filter---------------------------

class TemporalDisparityFilter:
    def __init__(self, threshold=1.0):
        self.prev_frame = None
        self.threshold = threshold

    def filter(self, current_frame: cp.ndarray) -> cp.ndarray:
        # First frame, nothing to compare with
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return current_frame.copy()

        # Compute absolute difference between frames
        diff = cp.abs(current_frame - self.prev_frame)

        # Mask out pixels with large temporal fluctuations
        stable_mask = diff < self.threshold

        # Keep only stable pixels, zero out noisy ones
        filtered_frame = cp.where(stable_mask, current_frame, 0)

        # Update previous frame
        self.prev_frame = current_frame.copy()

        return filtered_frame
    
temporal_filter = TemporalDisparityFilter(threshold=5.0)

# -------------------------Point Cloud Extractor-----------------------------
def reproject_disparity_to_pointcloud(disparity_cp, Q_cp=Q_cpmat):
    H, W = disparity_cp.shape

    # Generate homogeneous grid [u, v, d, 1]
    u, v = cp.meshgrid(cp.arange(W), cp.arange(H))  # Shape (H, W)
    d = disparity_cp

    ones = cp.ones_like(d)

    # Stack into shape (H, W, 4)
    points_hom = cp.stack([u, v, d, ones], axis=-1).reshape(-1, 4)  # (N, 4)

    # Reproject using Q: (N, 4) × (4, 4)^T = (N, 4)
    Q_T = Q_cp.T  # Transpose since we're doing row-vector multiplication
    points_3d_hom = points_hom @ Q_T  # (N, 4)

    # Normalize by W
    points_3d = points_3d_hom[:, :3] / points_3d_hom[:, 3:4]

    # Reshape back to image shape (H, W, 3)
    return points_3d.reshape(H, W, 3)
# -------------------------------------------------------------
# ------------- PCD Visualizer ----------------------------------------
class RealTimePointCloudRenderer:
    def __init__(self, window_name="PointCloud Viewer"):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name, width=960, height=720)
        self.pcd = o3d.geometry.PointCloud()
        self.added = False

    def update(self, point_cloud_np):
        # Reshape to (N, 3)
        points_flat = point_cloud_np.reshape(-1, 3)

        # Filter out invalid points
        valid_mask = np.isfinite(points_flat).all(axis=1)
        cleaned_points = points_flat[valid_mask]

        self.pcd.points = o3d.utility.Vector3dVector(cleaned_points)

        if not self.added:
            self.vis.add_geometry(self.pcd)
            self.added = True
        else:
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()
# ----------------------------------------------------------------
# Median filter kernel
# kernel = [[1, 1, 1],
#           [1, 1, 1],
#           [1, 1, 1]]



# Streams for left and right independent pre-processing
streamLeft = vpi.Stream()
streamRight = vpi.Stream()
# ------------ Sensor 0 --------------------
display1 = jetson_utils.glDisplay()

camera1 = jetson_utils.gstCamera(1280, 720, '1')
camera1.Open()
# # ----------------- Sensor 1 ----------------
display2 = jetson_utils.glDisplay()

camera2 = jetson_utils.gstCamera(1280, 720, '0')
camera2.Open()
# # --------------------- Streams ---------------
# Streams for left and right independent pre-processing
streamLeft = vpi.Stream()
streamRight = vpi.Stream()
streamStereo = streamLeft   #Using stream left to calculate
renderer = RealTimePointCloudRenderer()
# # -------------main loop ----------------
while display1.IsOpen() and display2.IsOpen():
    frame1, width, height = camera1.CaptureRGBA(zeroCopy=1)
    frame2, width, height = camera2.CaptureRGBA(zeroCopy=1)
    # print(f"frame1 NV12 info{frame1}")
    # print(frame1.__cuda_array_interface__)
    # {'shape': (720, 1280, 4), 'typestr': '<f4', 'data': (8666144768, False), 'version': 3}
    array1 = cp.asarray(frame1, cp.uint8)
    array_cp1 = array1[:, :, :3]
    # print(f"cp info{array_cp1}")
    # print(array_cp1.__cuda_array_interface__)
    # {'shape': (720, 1280, 3), 'typestr': '|u1', 'descr': [('', '|u1')], 
    # 'stream': 1, 'version': 3, 'strides': (5120, 4, 1), 'data': (8849956864, False)}
    # print("cupy buffer shappe")
    # print(array_cp1.__cuda_array_interface__)
    # left = vpi.asimage(array_cp1)
    array2 = cp.asarray(frame2, cp.uint8)
    array_cp2 = array2[:, :, :3]
    # print(f"custom array ={array}")

    # streamLeft.sync()
    # streamRight.sync()

    with vpi.Backend.CUDA:
        with streamLeft:
            # left = vpi.asimage(cp.asnumpy(array_cp1)).remap(warpl).convert(vpi.Format.U16).gaussian_filter(11, 1.9,border=vpi.Border.ZERO).convert(vpi.Format.Y16_ER)
            left = vpi.asimage(cp.asnumpy(array_cp1)).remap(warpl).convert(vpi.Format.Y16_ER)
            # print(left_img.format)
            # left_img = left.convert(vpi.Format.RGB8)
            # cv2.imshow("grey",left_img)
            # left = left.rescale((image_size[0]//2,image_size[1]//2), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            # left = left_img.convert(vpi.Format.Y16_ER)
        with streamRight:
            right = vpi.asimage(cp.asnumpy(array_cp2)).remap(warpr).convert(vpi.Format.Y16_ER)
            # right_img = right.convert(vpi.Format.RGB8)
            # right = right_img.convert(vpi.Format.Y16_ER)
    streamLeft.sync()
    streamRight.sync()
    # print(left.size)
    with vpi.Backend.VIC:
        with streamLeft:
            left = left.convert(vpi.Format.Y16_ER_BL)\
                .rescale((256*3, 256*2), interp=vpi.Interp.LINEAR)
        with streamRight:
            right = right.convert(vpi.Format.Y16_ER_BL)\
                        .rescale((256*3, 256*2), interp=vpi.Interp.LINEAR)
    streamLeft.sync()
    streamRight.sync()
    
    # Estimate stereo disparity.
    with streamStereo, vpi.Backend.OFA :
        output = vpi.stereodisp(left, right, window=5, maxdisp=maxdisp,
                                p1=10, p2=190,     # p1 5, p2 180
                                p2alpha=0, 
                                # mindisp=minDisparity,
                                # uniqueness=uniq,
                                includediagonals=3, 
                                numpasses=3,
                                # confthreshold=55535     #max limit 55535
                                )
    # # print('done!\nI Post-processing ... ', end='', flush=True)
    # # # Postprocess results and save them to disk
    with streamStereo, vpi.Backend.CUDA:
        output= output.convert(vpi.Format.S16, backend=vpi.Backend.VIC)\
                    .convert(vpi.Format.RGB8, scale=1.0/(32*maxdisp)*255)\
                    # .rescale((1280*3//4, 720*3//4), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
        with output.rlock_cuda() as cuda_buffer:
            # print("vpi cuda buffer shape")
            # print(cuda_buffer.__cuda_array_interface__)
            cp_array = cp.asarray(vpi_cuda_to_cupy_wrapper(cuda_buffer))
            # print(cp_array)
            filtered = temporal_filter.filter(cp_array)
            # print(filtered.shape)
            # pcd_cp = reproject_disparity_to_pointcloud(filtered)
            # print(pcd_cp)
            # renderer.update(cp.asnumpy(pcd_cp))

            # time.sleep(0.05)  # 20 FPS


    # # Show it
    # display.RenderOnce(frame, width, height)
    display1.RenderOnce(jetson_utils.cudaFromNumpy(cp.asnumpy(filtered)), width, height)
    # display1.RenderOnce(jetson_utils.cudaFromNumpy(left_img.cpu()), width, height)
    display1.SetTitle(f"Left | {width}x{height} | {display1.GetFPS():.1f} FPS")
    # display2.RenderOnce(jetson_utils.cudaFromNumpy(right_img.cpu()), width, height)
    # display2.SetTitle(f"Right| {width}x{height} | {display2.GetFPS():.1f} FPS")
    print("no issue !")
camera1.Close()
camera2.Close()
# renderer.close()