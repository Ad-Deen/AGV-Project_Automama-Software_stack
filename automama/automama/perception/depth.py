
import rclpy
from rclpy.node import Node
import numpy as np
import cupy as cp
import vpi
import cv2
import yaml
import os
import open3d as o3d
import jetson_utils
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Define your ROS2 Node
class StereoPointCloudNode(Node):
    def __init__(self):
        super().__init__('stereo_pointcloud_node')
        self.image_pub = self.create_publisher(Image, '/disparity', 10)
        self.bridge = CvBridge()
        # Load calibration parameters
        self.load_calibration()

        # Setup cameras
        self.camera1 = jetson_utils.gstCamera(1280, 720, '1')
        self.camera2 = jetson_utils.gstCamera(1280, 720, '0')
        self.camera1.Open()
        self.camera2.Open()

        # Setup displays (optional)
        self.display1 = jetson_utils.glDisplay()

        # Setup VPI streams
        self.streamLeft = vpi.Stream()
        self.streamRight = vpi.Stream()
        self.streamStereo = self.streamLeft

        # Temporal filter
        self.temporal_filter = TemporalDisparityFilter(threshold=5.0)

        # Start timer
        self.timer = self.create_timer(0.033, self.process_stereo)
# /home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib/camera_calibration1.yaml
    def load_calibration(self):
        calib_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"
        
        with open(os.path.join(calib_dir, "camera_calibration1.yaml")) as f:
            left_data = yaml.safe_load(f)
        with open(os.path.join(calib_dir, "camera_calibration0.yaml")) as f:
            right_data = yaml.safe_load(f)

        K1 = np.array(left_data["camera_matrix"], dtype=np.float64)
        D1 = np.array(left_data["distortion_coefficients"], dtype=np.float64)
        K2 = np.array(right_data["camera_matrix"], dtype=np.float64)
        D2 = np.array(right_data["distortion_coefficients"], dtype=np.float64)

        if D1.size != 5:
            D1 = np.zeros(5)
        if D2.size != 5:
            D2 = np.zeros(5)

        with open(os.path.join(calib_dir, "stereo_calibration.yaml")) as f:
            data = yaml.safe_load(f)

        R = np.array(data["R"], dtype=np.float64).reshape((3, 3))
        T = np.array(data["T"], dtype=np.float64).reshape((3, 1))

        image_size = (1280, 720)
        R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0)

        self.Q_cpmat = cp.asarray(Q)
        map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

        self.warpl = vpi.WarpMap(vpi.WarpGrid(image_size))
        self.warpr = vpi.WarpMap(vpi.WarpGrid(image_size))

        wxl, wyl = np.asarray(self.warpl).transpose(2,1,0)
        wxr, wyr = np.asarray(self.warpr).transpose(2,1,0)

        wxl[:] = map1_left.transpose(1,0)
        wyl[:] = map2_left.transpose(1,0)
        wxr[:] = map1_right.transpose(1,0)
        wyr[:] = map2_right.transpose(1,0)

    def process_stereo(self):
        frame1, width, height = self.camera1.CaptureRGBA(zeroCopy=1)
        frame2, _, _ = self.camera2.CaptureRGBA(zeroCopy=1)

        array_cp1 = cp.asarray(frame1, cp.uint8)[:, :, :3]
        array_cp2 = cp.asarray(frame2, cp.uint8)[:, :, :3]

        with vpi.Backend.CUDA:
            with self.streamLeft:
                left = vpi.asimage(cp.asnumpy(array_cp1)).remap(self.warpl).convert(vpi.Format.Y16_ER)
            with self.streamRight:
                right = vpi.asimage(cp.asnumpy(array_cp2)).remap(self.warpr).convert(vpi.Format.Y16_ER)

        self.streamLeft.sync()
        self.streamRight.sync()

        with vpi.Backend.VIC:
            with self.streamLeft:
                left = left.convert(vpi.Format.Y16_ER_BL).rescale((768, 512), interp=vpi.Interp.LINEAR)
            with self.streamRight:
                right = right.convert(vpi.Format.Y16_ER_BL).rescale((768, 512), interp=vpi.Interp.LINEAR)

        self.streamLeft.sync()
        self.streamRight.sync()

        maxdisp = 256
        with self.streamStereo, vpi.Backend.OFA:
            output = vpi.stereodisp(left, right, window=5, maxdisp=maxdisp,
                                    p1=10, p2=190, p2alpha=0, includediagonals=3, numpasses=3)

        with self.streamStereo, vpi.Backend.CUDA:
            output = output.convert(vpi.Format.S16, backend=vpi.Backend.VIC).convert(
                vpi.Format.RGB8, scale=1.0 / (32 * maxdisp) * 255)

            with output.rlock_cuda() as cuda_buffer:
                disparity = self.temporal_filter.filter(cp.asarray(CudaArrayWrapper(cuda_buffer.__cuda_array_interface__)))
                msg = self.bridge.cv2_to_imgmsg(cp.asnumpy(disparity), encoding='bgr8')
                self.image_pub.publish(msg)
        disparity_np = cp.asnumpy(disparity)
        self.display1.RenderOnce(jetson_utils.cudaFromNumpy(disparity_np), width, height)
        self.display1.SetTitle(f"Disparity | {width}x{height} | {self.display1.GetFPS():.1f} FPS")


# --- Wrappers and filters from original code ---
class TemporalDisparityFilter:
    def __init__(self, threshold=1.0):
        self.prev_frame = None
        self.threshold = threshold

    def filter(self, current_frame: cp.ndarray) -> cp.ndarray:
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            return current_frame.copy()
        diff = cp.abs(current_frame - self.prev_frame)
        stable_mask = diff < self.threshold
        filtered_frame = cp.where(stable_mask, current_frame, 0)
        self.prev_frame = current_frame.copy()
        return filtered_frame

class CudaArrayWrapper:
    def __init__(self, cuda_iface: dict):
        self.__cuda_array_interface__ = {
            'shape': tuple(cuda_iface['shape']),
            'strides': tuple(cuda_iface['strides']) if 'strides' in cuda_iface else None,
            'typestr': cuda_iface['typestr'],
            'data': cuda_iface['data'],
            'version': cuda_iface['version']
        }

# --- Main ROS2 Entry Point ---
def main(args=None):
    rclpy.init(args=args)
    node = StereoPointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
