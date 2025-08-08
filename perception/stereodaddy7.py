import rclpy
from rclpy.node import Node
import cv2
from daddyutils.camera_handler3 import JetsonCameraHandler
from daddyutils.warp_utils import load_and_generate_warp_maps
from daddyutils.tensorrt_infer3 import TensorRTInference
# import pycuda.autoinit
import cupy as cp
from cupy import RawKernel
import pycuda.driver as cuda
import numpy as np
from daddyutils.o3D_vis import Open3DVisualizer
from daddyutils.redmask import mask_red_pixels
from daddyutils.consistancy import create_valid_mask_from_depth_concentration
import vpi
#------------------Point cloud generator class ---------------------
class PointCloudGenerator:
    def __init__(self, Q_cp: cp.ndarray, voxel_size: float = 0.1):
        self.Q_cp = Q_cp
        self.voxel_size = voxel_size
        self.temporal_filter = TemporalDisparityFilter(threshold=15.0)

        self.height = 480
        self.width = 640
        self.x_coords = cp.arange(self.width, dtype=cp.float32)
        self.y_coords = cp.arange(self.height, dtype=cp.float32)
        self.x_grid, self.y_grid = cp.meshgrid(self.x_coords, self.y_coords)

    def voxel_downsample(self, points: cp.ndarray) -> cp.ndarray:
        voxel_indices = cp.floor(points / self.voxel_size).astype(cp.int32)
        keys = voxel_indices[:, 0] * 73856093 ^ voxel_indices[:, 1] * 19349663 ^ voxel_indices[:, 2] * 83492791
        unique_keys, unique_indices = cp.unique(keys, return_index=True)
        return points[unique_indices]

    def generate_point_cloud(self, disparity_gpu: cp.ndarray) -> cp.ndarray:
        filtered = self.temporal_filter.filter(disparity_gpu)
        disp_min = filtered.min()
        disp_max = filtered.max()

        valid_mask = (filtered > 0.1) & (filtered < disp_max * 0.9)
        valid_disp = filtered[valid_mask]
        x_coords = self.x_grid[valid_mask]
        y_coords = self.y_grid[valid_mask]

        ones = cp.ones_like(valid_disp)
        points_hom = cp.stack((x_coords, y_coords, valid_disp, ones), axis=1)
        XYZW = points_hom @ self.Q_cp.T
        XYZ = XYZW[:, :3] / XYZW[:, 3:4]

        return self.voxel_downsample(XYZ)

    def get_normalized_disparity_image(self, disparity_gpu: cp.ndarray) -> np.ndarray:
        filtered = self.temporal_filter.filter(disparity_gpu)
        disp_min = filtered.min()
        disp_max = filtered.max()
        disp_norm_gpu = ((filtered - disp_min) / (disp_max - disp_min)) * 255.0
        return cp.asnumpy(disp_norm_gpu.astype(cp.uint8))
    
    def get_raw_disparity_image(self, disparity_gpu: cp.ndarray) -> np.ndarray:
        filtered = self.temporal_filter.filter(disparity_gpu)
        disp_min = filtered.min()
        disp_max = filtered.max()
        # print(disp_max)
        # print(disp_min)
        # disp_norm_gpu = ((filtered - disp_min) / (disp_max - disp_min)) * 255.0
        return cp.asnumpy(filtered.astype(cp.float32))

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
        # ------------------------------------



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

cuda.init()
device = cuda.Device(0)
# cuda.Device(0).make_context() 
context = device.make_context()  # Push context on stack
context.pop()
class StereoShowNode(Node):
    def __init__(self):
        global context
        super().__init__('stereo_show_node')
        self.get_logger().info('Starting StereoShowNode...')
        self.kernel_code = r'''
        extern "C" __global__
        void stereo_sgbm_like(const float* left, const float* right, float* disparity,
                            int width, int height, int max_disp, int window, int step,
                            float uniqueness_ratio, float P1, float P2) {

            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            int half_window = window / 2;
            if (x >= width || y >= height) return;

            // Get the value of the current pixel in the left image
            float center_left_val = left[y * width + x];
            if (center_left_val == 0.0f) {
                disparity[y * width + x] = 0.0f;
                return;
            }

            // Get the previous disparity from the left neighbor (if it exists)
            int prev_disp = (x > 0) ? (int)disparity[y * width + (x - 1)] : 0;

            float best_cost = 1e10;
            float second_best_cost = 1e10;
            int best_disp = 0;

            for (int d = 0; d < max_disp; d++) {
                float cost = 0.0;

                // --- Calculate matching cost (SAD) ---
                for (int wy = -half_window; wy <= half_window; wy++) {
                    for (int wx = -half_window; wx <= half_window; wx++) {
                        int xx = x + wx * step;
                        int yy = y + wy * step;

                        if (xx >= 0 && xx < width && yy >= 0 && yy < height && (xx - d) >= 0) {
                            float l = left[yy * width + xx];
                            float r = right[yy * width + xx - d];
                            cost += fabsf(l - r);
                        }
                    }
                }

                // --- Add the smoothness penalty ---
                float penalty = 0.0f;
                if (d != prev_disp) {
                    if (fabsf((float)d - (float)prev_disp) == 1.0f) {
                        // Small change in disparity, apply small penalty (P1)
                        penalty = P1;
                    } else {
                        // Large change in disparity, apply large penalty (P2)
                        penalty = P2;
                    }
                }

                float total_cost = cost + penalty;

                // Track best and second best for uniqueness check
                if (total_cost < best_cost) {
                    second_best_cost = best_cost;
                    best_cost = total_cost;
                    best_disp = d;
                } else if (total_cost < second_best_cost) {
                    second_best_cost = total_cost;
                }
            }

            if (best_cost < second_best_cost * uniqueness_ratio) {
                disparity[y * width + x] = (float)best_disp;
            } else {
                disparity[y * width + x] = 0.0f;
            }
        }


        '''
        self.temporal_filter = TemporalDisparityFilter(threshold=2.0)
        # ----------------Custom cuda Stereo Pipeline Parameters-------------------------
        self.height = 480
        self.width = 640
        self.block_size = (16, 16)
        self.grid_size = ((self.width + self.block_size[0] - 1) // self.block_size[0],
                    (self.height + self.block_size[1] - 1) // self.block_size[1])

        self.max_disp = 128       # You can tune this
        self.window = 7           # Odd window size like 5, 7, etc.
        self.step = 1             # Sparse block step (1 = dense, >1 = sparse)
        self.p1 = 0
        self.p2 = 0
        
        '''
        uniqueness_ratio = 0.8 → allows a ~20% cost gap between best and second-best match.

        Lower → more aggressive filtering (more occlusions detected).

        Tune depending on baseline and texture.
        '''
        self.uniqueness_ratio = 1.0
        self.stereo_kernel = RawKernel(self.kernel_code, 'stereo_sgbm_like')
        
        # ------------------ SGBM VPI Pipeline Params------------------------
        self.max_disp_vpi = 128

        # -------------------------------------------------------------------
        # Load calibration & generate warp maps
        calib_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"
        # self.disparity_gpu = cp.zeros((self.height, self.width), dtype=cp.float32)
        warpl, warpr , Q = load_and_generate_warp_maps(calib_dir)
        # print(Q)
        with cp.cuda.Device(0):
            self.left_gpu = cp.empty((self.height, self.width), dtype=cp.float32)
            self.right_gpu = cp.empty((self.height, self.width), dtype=cp.float32)
            self.disparity_gpu = cp.zeros((self.height, self.width), dtype=cp.float32)
            # For 3D coordinate
            self.x_coords = cp.arange(640, dtype=cp.float32)
            self.y_coords = cp.arange(480, dtype=cp.float32)
            self.x_grid, self.y_grid = cp.meshgrid(self.x_coords, self.y_coords)
            self.Q_cp = cp.asarray(Q)
        self.Q_np = Q
        self.f = 1003.024       #1003.024
        self.cx = 261.901
        self.cy = 278.561
        self.point_cloud_generator = PointCloudGenerator(self.Q_cp)
        with vpi.Backend.CUDA:
            self.streamLeft = vpi.Stream()
            self.streamRight = vpi.Stream()
            self.streamStereo = self.streamLeft
        engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/yolov8n640x640.engine"
        # video_path = '/path/to/your/video.mp4'
        self.cap = cv2.VideoCapture("/home/deen/ros2_ws/src/automama/automama/perception/run3.mp4")
        # Initialize camera handlers with warp maps
        self.cam_left = JetsonCameraHandler(sensor_id=1, warp_map=warpl)
        self.cam_right = JetsonCameraHandler(sensor_id=0, warp_map=warpr)
        context.push()
        self.trt_infer = TensorRTInference(engine_path)
        context.pop()

        # Create a periodic timer (30 FPS)
        self.timer = self.create_timer(1.0 / 30, self.timer_callback)
        # self.temporal_filter = TemporalDisparityFilter(threshold=20.0)
        self.should_shutdown = False

        # Camera intrinsics and baseline (fill in your values)
        # self.fx = 632.14
        # self.fy = 847.5
        # self.cx = 315.05
        # self.cy = 266.0
        # self.baseline = 0.17

        self.visualizer = Open3DVisualizer()
    def voxel_downsample(self,points, voxel_size):
        """
        GPU voxel downsampling for Nx3 CuPy point cloud.
        
        Args:
            points (cp.ndarray): (N, 3) point cloud
            voxel_size (float): size of each voxel cube

        Returns:
            cp.ndarray: downsampled points (M, 3)
        """

        # Convert to voxel indices
        voxel_indices = cp.floor(points / voxel_size).astype(cp.int32)

        # Create a unique key per voxel
        keys = voxel_indices[:, 0] * 73856093 ^ \
            voxel_indices[:, 1] * 19349663 ^ \
            voxel_indices[:, 2] * 83492791

        # Get unique voxels
        unique_keys, unique_indices = cp.unique(keys, return_index=True)

        # Select one point per voxel
        downsampled_points = points[unique_indices]
        return downsampled_points
    def compute_disparity(self):
        self.stereo_kernel(
            self.grid_size,
            self.block_size,
            (
                self.left_gpu, self.right_gpu, self.disparity_gpu,
                np.int32(self.width), np.int32(self.height),
                np.int32(self.max_disp), np.int32(self.window), np.int32(self.step),
                np.float32(self.uniqueness_ratio),np.float32(self.p1),
                np.float32(self.p2)
                
            )
        )
        return self.disparity_gpu


    def timer_callback(self):
        global context

        if self.should_shutdown:
            return  # Skip processing if shutdown triggered
        # ret, frame = self.cap.read()    #video check
        left_frame = self.cam_left.capture_and_process()    #camera feed
        right_frame = self.cam_right.capture_and_process()  #camera feed

        # if left_frame is not None:
        #     cv2.imshow("Left Camera", left_frame)
        # else:
        #     self.get_logger().warn('Left camera returned None.')

        # if right_frame is not None:
        #     cv2.imshow("Right Camera", right_frame)
        # else:
        #     self.get_logger().warn('Right camera returned None.')
        # Buffer right frame here (store in variable)
        buffered_right = right_frame.copy()

        #------------- Run inference on left frame only-------------------

        # left_mask = self.trt_infer.infer(left_frame,[0])

        left_mask = mask_red_pixels(left_frame,100)

# --------------------Create the masked left frame on GPU--------------------------
        with cp.cuda.Device(0):
            # Transfer the left frame and the color mask to the GPU
            left_frame_cp = cp.asarray(left_frame)
            color_mask_cp = cp.asarray(left_mask)

            # Create a single-channel binary mask from the color mask
            # We check if any of the R, G, or B channels are non-zero.
            binary_mask_cp = cp.any(color_mask_cp > 0, axis=-1)
            
            # Create a 3-channel binary mask for element-wise multiplication with the RGB frame
            binary_mask_3ch = cp.stack([binary_mask_cp, binary_mask_cp, binary_mask_cp], axis=-1)
            
            # Apply the mask to the left frame
            # Pixels where the mask is 0 become black, otherwise they keep their original value
            masked_left_frame_cp = left_frame_cp * binary_mask_3ch
            
            # Get the result back to the CPU for display and stereo
            masked_left_frame_np = masked_left_frame_cp.get()

        # Show the final masked frame
        # cv2.imshow("Masked Left Frame", masked_left_frame_np)
        # --------------------Stereo on camera (custom)-------------------------------
       
        # with cp.cuda.Device(0):
        #     left_grey = cv2.cvtColor(masked_left_frame_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
        #     right_grey = cv2.cvtColor(buffered_right, cv2.COLOR_BGR2GRAY).astype(np.float32)
        #     self.left_gpu.set(left_grey)   # reupload without reallocating
        #     self.right_gpu.set(right_grey)
        #     self.right_gpu = self.right_gpu/ 255.0
        #     self.left_gpu = self.left_gpu/ 255.0
        #     self.compute_disparity()

        #     # Generate point cloud
        #     # downsampled = self.point_cloud_generator.generate_point_cloud(self.disparity_gpu)
        #     # self.visualizer.update(cp.asnumpy(downsampled))

        #     # Get raw disparity
        #     disparity_img = self.point_cloud_generator.get_raw_disparity_image(self.disparity_gpu).astype(np.float32)
        #     # Create a proper valid mask first
        #     valid_disparity_mask = disparity_img > 0  # Only positive disparities are valid

        #     # Avoid division by zero more elegantly
        #     disparity_safe = disparity_img.copy()
        #     disparity_safe[~valid_disparity_mask] = 1.0  # Will be filtered out anyway

        #     # Compute depth
        #     depth_map = (self.f * 0.1741) / disparity_safe  # meters, float32
        #     # print(f"min depth :{np.min(depth_map)} max depth: {np.max(depth_map)}")
        #     # Apply reasonable depth limits (adjust these based on your scene)
        #     min_depth = 0.1  # 10cm minimum
        #     max_depth = 5.0  # 10m maximum
        #     valid_depth_mask = (depth_map >= min_depth) & (depth_map <= max_depth)

        #     # Combined valid mask
        #     final_valid_mask = valid_disparity_mask & valid_depth_mask
            
        #     # Visualization (unchanged)
        #     depth_map_vis = np.clip(depth_map, 0.1, 5.0)
        #     depth_map_vis = ((depth_map_vis / 5.0) * 255.0).astype(np.uint8)
        #     # cv2.imshow("Disparity Map", disparity_img.astype(np.uint8))
        #     cv2.imshow("Depth Map (Clipped)", depth_map_vis)

        #     # Backproject to 3D
        #     height, width = depth_map.shape
        #     u, v = np.meshgrid(np.arange(width), np.arange(height))

        #     # Only compute 3D points for valid pixels
        #     valid_pixels = final_valid_mask.reshape(-1)

        #     u_valid = u.reshape(-1)[valid_pixels]
        #     v_valid = v.reshape(-1)[valid_pixels]
        #     z_valid = depth_map.reshape(-1)[valid_pixels]
        #     print(z_valid)
        #     # print(f"min depth :{np.min(z_valid)} max depth: {np.max(z_valid)}")
        #     # print(z_valid[len(z_valid)//2])
        #     # Compute 3D coordinates
        #     x_valid = (u_valid - self.cx) * z_valid / self.f
        #     y_valid = (v_valid - self.cy) * z_valid / self.f

        #     points = np.stack((x_valid, -y_valid, -z_valid), axis=-1)
        #     # print(points.shape)
        #     # Optional: add color
        #     # colors = left_frame.reshape(-1, 3).astype(np.float32) / 255.0
        #     # colors = colors[valid_mask]

        #     # Update point cloud
        #     self.visualizer.update(points)

        # ------------------stereo with SGBM VPI -----------------------

        with vpi.Backend.CUDA:
            with self.streamLeft:
                # print(type(left_frame))
                # left_image = vpi.asimage(left_frame)
                left = vpi.asimage(left_frame).convert(vpi.Format.U8).eqhist().convert(vpi.Format.Y16_ER).convert(vpi.Format.Y16_ER_BL,backend=vpi.Backend.VIC )
            #         # .rescale((256*3, 256*2), interp=vpi.Interp.LINEAR)
            with self.streamRight:
                # right_image = vpi.asimage(right_frame)
                right = vpi.asimage(right_frame).convert(vpi.Format.U8).eqhist().convert(vpi.Format.Y16_ER).convert(vpi.Format.Y16_ER_BL,backend=vpi.Backend.VIC)
                            # .rescale((256*3, 256*2), interp=vpi.Interp.LINEAR)
            self.streamLeft.sync()
            self.streamRight.sync()
        
        # Estimate stereo disparity.
        with self.streamStereo, vpi.Backend.OFA :
            output = vpi.stereodisp(left,right, window=5, maxdisp=self.max_disp_vpi,
                                    p1=5, p2=180,     # p1 5, p2 180
                                    p2alpha=0, 
                                    # mindisp=minDisparity,
                                    # uniqueness=uniq,
                                    includediagonals=3, 
                                    numpasses=1,
                                    # confthreshold=55535     #max limit 55535
                                    )

        with self.streamStereo, vpi.Backend.CUDA:
            output= output.convert(vpi.Format.S16, backend=vpi.Backend.VIC)\
                        .convert(vpi.Format.U16, scale=1.0/(32*self.max_disp_vpi)*255)\
                        # .rescale((1280*3//4, 720*3//4), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            with output.rlock_cuda() as cuda_buffer:
                # print(cuda_buffer.__cuda_array_interface__)
                filtered = self.temporal_filter.filter(cp.asarray(CudaArrayWrapper(cuda_buffer.__cuda_array_interface__)))
                # print(filtered[240,:])
                
                # print(smooth_map)
                # disparity_img = cp.asnumpy(filtered)


                valid_disparity_mask = filtered > 0  # Only positive disparities are valid

                # Avoid division by zero more elegantly
                disparity_safe = filtered.copy()
                disparity_safe[~valid_disparity_mask] = 1.0  # Will be filtered out anyway

                # Compute depth #0.1741 = baselength
                depth_map = (2400 * 0.1741) / disparity_safe  # meters, float32

                # --- CORRECTED MASKING PIPELINE ---

                # 1. Create a mask for valid depth limits
                min_depth = 0.7
                max_depth = 50.0
                valid_depth_mask = (depth_map >= min_depth) & (depth_map <= max_depth)

                # 2. Get the boolean mask from your gradient function
                valid_gradient_mask = create_valid_mask_from_depth_concentration(depth_map,min_depth=0.7,max_depth=50,step_size=0.2,min_points_threshold=2600)

                # 3. Combine all valid masks into a single final mask
                final_valid_mask = valid_depth_mask & valid_gradient_mask

                # Do NOT overwrite the final_valid_mask with the float depth map here.
                # Keep final_valid_mask as a boolean array.

                # Apply the mask to a copy of the depth map for visualization and other uses
                filtered_depth_map = depth_map.copy()
                filtered_depth_map[~final_valid_mask] = None
                # print(filtered_depth_map)
                # --- VISUALIZATION ---
                # Use the filtered depth map for visualization to show the result of the filtering
                display_max_depth = 10.0 # Use a smaller max for better visual contrast
                depth_map_vis = cp.clip(filtered_depth_map, 0.0, display_max_depth)
                depth_map_vis = ((depth_map_vis / display_max_depth) * 255.0).astype(np.uint8)

                cv2.imshow("Depth Map (Clipped)", cp.asnumpy(depth_map_vis))

                # --- BACKPROJECT TO 3D ---
                # Convert relevant data to NumPy arrays once for efficiency
                filtered_depth_map_np = cp.asnumpy(filtered_depth_map)
                final_valid_mask_np = cp.asnumpy(final_valid_mask)

                height, width = filtered_depth_map_np.shape
                u, v = np.meshgrid(np.arange(width), np.arange(height))

                # Only compute 3D points for valid pixels
                valid_pixels = final_valid_mask_np.reshape(-1)

                # These lines now correctly use the boolean valid_pixels to index the arrays
                u_valid = u.reshape(-1)[valid_pixels]
                v_valid = v.reshape(-1)[valid_pixels]
                z_valid = filtered_depth_map_np.reshape(-1)[valid_pixels]

               # Only compute 3D points for valid pixels

                valid_pixels = final_valid_mask_np.reshape(-1)


                # These lines now correctly use the boolean valid_pixels to index the arrays

                u_valid = u.reshape(-1)[valid_pixels]

                v_valid = v.reshape(-1)[valid_pixels]

                z_valid = filtered_depth_map_np.reshape(-1)[valid_pixels]


                # Compute 3D coordinates

                x_valid = (u_valid - self.cx) * z_valid / self.f

                y_valid = (v_valid - self.cy) * z_valid / self.f

                print(f"Minimum = {np.min(z_valid)})")

                print(f"Maximum = {np.max(z_valid)})")

                points = np.stack((x_valid, -y_valid, -z_valid), axis=-1) 
                # Optional: add color
                # colors = left_frame.reshape(-1, 3).astype(np.float32) / 255.0
                # colors = colors[valid_mask]

                # Update point cloud
                self.visualizer.update(points)


        

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit key pressed. Initiating shutdown...")
            self.should_shutdown = True

    def cleanup(self):
        self.get_logger().info("Cleaning up resources...")
        try:
            self.cam_left.close()   # Use the close() method you defined in the handler
            self.cam_right.close()
            self.visualizer.close()
        except Exception as e:
            self.get_logger().error(f"Error closing cameras: {e}")
        self.timer.cancel()
        cv2.destroyAllWindows()
        self.destroy_node()


def main(args=None):
    # cuda.init()
    rclpy.init(args=args)
    node = StereoShowNode()

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received.")
    finally:
        context.detach()
        node.cleanup()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
