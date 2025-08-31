import numpy as np
import jetson_utils  # recommended over deprecated jetson.utils
import vpi
import cupy as cp
import cv2
import yaml
import os
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
# print(np.asarray(warpl).shape)          #(736, 1280, 2)
wxl[:] = map1_left.transpose(1,0)
wyl[:] = map2_left.transpose(1,0)
# print((map1_left.transpose(1,0)).shape)

wxr[:] = map1_right.transpose(1,0)
wyr[:] = map2_right.transpose(1,0)


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
maxdisp = 128
# # -------------main loop ----------------
while display1.IsOpen() and display2.IsOpen():
    frame1, width, height = camera1.CaptureRGBA(zeroCopy=1)
    frame2, width, height = camera2.CaptureRGBA(zeroCopy=1)

    array1 = cp.asarray(frame1, cp.uint8)
    array_cp1 = array1[:, :, :3]
    array2 = cp.asarray(frame2, cp.uint8)
    array_cp2 = array2[:, :, :3]
    # print(f"custom array ={array}")



    with vpi.Backend.CUDA:
        with streamLeft:
            left = vpi.asimage(cp.asnumpy(array_cp1)).remap(warpl)
            # left = left.rescale((image_size[0]//2,image_size[1]//2), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            # left = left.convert(vpi.Format.Y16_ER)
        with streamRight:
            right = vpi.asimage(cp.asnumpy(array_cp2)).remap(warpr)
            # right = right.rescale((image_size[0]//2, image_size[1]//2), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            # right = right.convert(vpi.Format.Y16_ER)
    # print(left.size)
    # with vpi.Backend.VIC:
    #     with streamLeft:
    #         left = left.convert(vpi.Format.Y16_ER_BL)
    #     with streamRight:
    #         right = right.convert(vpi.Format.Y16_ER_BL)
    
    # # print(f'I Input left image: {left.size} {left.format}\n'
    # #     f'I Input right image: {right.size} {right.format}', flush=True)
    # # outWidth = (left.size[0] + downscale - 1) // downscale
    # # outHeight = (left.size[1] + downscale - 1) // downscale
    
    # # if calcConf:
    # #      confidenceU16 = vpi.Image((outWidth, outHeight), vpi.Format.U16)

    # # if backend == 'ofa-pva-vic' and maxDisparity not in {128, 256}:
    # #      maxDisparity = 128 if (maxDisparity // 128) < 1 else 256
    # #      if verbose:
    # #          print(f'W {backend} only supports 128 or 256 maxDisparity. Overriding to {maxDisparity}', flush=True)
    
    # # if verbose:
    # #      if 'ofa' not in backend:
    # #          print('W Ignoring P2 alpha and number of passes since not an OFA backend', flush=True)
    # #      if backend != 'cuda':
    # #          print('W Ignoring uniqueness since not a CUDA backend', flush=True)
    # #      print('I Estimating stereo disparity ... ', end='', flush=True)

    # # Estimate stereo disparity.
    # with streamStereo, vpi.Backend.OFA :
    #         output = vpi.stereodisp(left, right, window=5, maxdisp=maxdisp,
    #                                 # p1=p1, p2=p2, p2alpha=p2_alpha, mindisp=minDisparity,
    #                                 # uniqueness=uniq,includediagonals=diagonals, 
    #                                 # numpasses=numPasses,confthreshold=55535
    #                                 )
    # print('done!\nI Post-processing ... ', end='', flush=True)
    # # # Postprocess results and save them to disk
    # with streamStereo, vpi.Backend.CUDA:
    #     output= output.convert(vpi.Format.S16, backend=vpi.Backend.VIC).convert(vpi.Format.RGB8, scale=1.0/(32*maxdisp)*255)
    # #     # Some backends outputs disparities in block-linear format, we must convert them to
    #     # pitch-linear for consistency with other backends.
    #     if disparityS16.format == vpi.Format.S16_BL:
    #         disparityS16 = disparityS16.convert(vpi.Format.S16, backend=vpi.Backend.VIC)
    # disparityU8 = disparityS16.convert(vpi.Format.U8, scale=255.0/(32*128)).cpu()

    # disparityColor = cv2.applyColorMap(disparityU8, cv2.COLORMAP_JET)

    # # Converts to RGB for output with PIL.
    # disparityColor = cv2.cvtColor(disparityColor, cv2.COLOR_BGR2RGB)
    # left = vpi.asimage(cp.asnumpy(array_cp1))
    # right = vpi.asimage(cp.asnumpy(array_cp2))
    # ------------------------------------------------------
    # with vpi.Backend.CUDA:
    #     with streamLeft:
    #         gray1 = left.convert(vpi.Format.U8)
    #         blurred1 = gray1.box_filter(11, border=vpi.Border.ZERO)
    #         output1 = blurred1.convert(vpi.Format.RGB8)
    # with vpi.Backend.CUDA:
    #     with streamRight:
    #         gray2 = right.convert(vpi.Format.U8)
    #         blurred2 = gray2.box_filter(11, border=vpi.Border.ZERO)
    #         output2 = blurred2.convert(vpi.Format.RGB8)

    # # Show it
    # display.RenderOnce(frame, width, height)
    display1.RenderOnce(jetson_utils.cudaFromNumpy(left.cpu()), width, height)
    display1.SetTitle(f"Sensor 0 | {width}x{height} | {display1.GetFPS():.1f} FPS")
    display2.RenderOnce(jetson_utils.cudaFromNumpy(right.cpu()), width, height)
    display2.SetTitle(f"Sensor 1 | {width}x{height} | {display2.GetFPS():.1f} FPS")
    # print("no issue !")
camera1.Close()
camera2.Close()
