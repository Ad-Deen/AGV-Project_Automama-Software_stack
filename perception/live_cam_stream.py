import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import os
import yaml
import numpy as np
import vpi
import jetson_utils
import cupy as cp

class JetsonCameraHandler:
    def __init__(self, node, sensor_id, topic_name, warp_map):
        self.node = node
        self.sensor_id = sensor_id
        self.topic_name = topic_name
        self.bridge = CvBridge()
        self.publisher = node.create_publisher(Image, topic_name, 10)

        self.camera = jetson_utils.gstCamera(1280, 720, str(sensor_id))
        self.camera.Open()

        self.stream = vpi.Stream()
        self.warp_map = warp_map

    def capture_and_publish(self):
        with vpi.Backend.CUDA:
            with self.stream:
                # Capture frame
                frame, _, _ = self.camera.CaptureRGBA(zeroCopy=1)
                array = cp.asarray(frame, dtype=cp.uint8)
                rgb_array = array[:, :, :3]

                # Convert to VPI image
                vpi_img = vpi.asimage(cp.asnumpy(rgb_array))

                # Apply remap (unwrap)
                rectified = vpi_img.remap(self.warp_map).convert(vpi.Format.RGB8)

                # Download to host for visualization
                rectified_numpy = rectified.cpu()

        # Show using OpenCV
        cv2.imshow(f"Frame Sensor {self.sensor_id}", rectified_numpy)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
                
        self.stream.sync()

        # Move to CPU and publish
        rectified_np = rectified.cpu().copy()
        msg = self.bridge.cv2_to_imgmsg(rectified_np, encoding='bgr8')
        self.publisher.publish(msg)


class CSIDualCameraPublisher(Node): #main loop
    def __init__(self):
        super().__init__('jetson_dual_camera_vpi_rectifier')
        self.get_logger().info('Initializing Jetson VPI stereo camera node...')

        # Load and generate warp maps
        warpl, warpr = self.load_and_generate_warp_maps()

        # Setup camera handlers
        self.cam_left = JetsonCameraHandler(self, sensor_id=1, topic_name='/csi_cam_left', warp_map=warpl)
        self.cam_right = JetsonCameraHandler(self, sensor_id=0, topic_name='/csi_cam_right', warp_map=warpr)

        # Publish at 30 FPS
        self.timer = self.create_timer(1.0 / 30, self.timer_callback)

    def timer_callback(self):
        self.cam_left.capture_and_publish()
        self.cam_right.capture_and_publish()

    def load_and_generate_warp_maps(self):
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
            stereo_data = yaml.safe_load(f)

        R = np.array(stereo_data["R"], dtype=np.float64).reshape((3, 3))
        T = np.array(stereo_data["T"], dtype=np.float64).reshape((3, 1))

        image_size = (1280, 720)
        R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0)

        map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

        # Create VPI warp maps
        warpl = vpi.WarpMap(vpi.WarpGrid(image_size))
        warpr = vpi.WarpMap(vpi.WarpGrid(image_size))

        wxl, wyl = np.asarray(warpl).transpose(2, 1, 0)
        wxr, wyr = np.asarray(warpr).transpose(2, 1, 0)

        wxl[:] = map1_left.transpose(1, 0)
        wyl[:] = map2_left.transpose(1, 0)
        wxr[:] = map1_right.transpose(1, 0)
        wyr[:] = map2_right.transpose(1, 0)

        return warpl, warpr


def main(args=None):
    rclpy.init(args=args)
    node = CSIDualCameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt - shutting down')
    finally:
        JetsonCameraHandler.camera.Close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
