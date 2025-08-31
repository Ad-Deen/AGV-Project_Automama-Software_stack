import rclpy
from rclpy.node import Node
import cv2
from daddyutils.camera_handler import JetsonCameraHandler
from daddyutils.warp_utils import load_and_generate_warp_maps


class StereoShowNode(Node):
    def __init__(self):
        super().__init__('stereo_show_node')
        self.get_logger().info('Starting StereoShowNode...')

        # Load calibration & generate warp maps
        calib_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"
        warpl, warpr = load_and_generate_warp_maps(calib_dir)

        # Initialize camera handlers with warp maps
        self.cam_left = JetsonCameraHandler(self, sensor_id=1, topic_name='/csi_cam_left', warp_map=warpl)
        self.cam_right = JetsonCameraHandler(self, sensor_id=0, topic_name='/csi_cam_right', warp_map=warpr)

        # Create a periodic timer (30 FPS)
        self.timer = self.create_timer(1.0 / 30, self.timer_callback)

        self.should_shutdown = False

    def timer_callback(self):
        if self.should_shutdown:
            return  # Skip processing if shutdown triggered

        left_frame = self.cam_left.capture_and_publish()
        right_frame = self.cam_right.capture_and_publish()

        if left_frame is not None:
            cv2.imshow("Left Camera", left_frame)
        else:
            self.get_logger().warn('Left camera returned None.')

        if right_frame is not None:
            cv2.imshow("Right Camera", right_frame)
        else:
            self.get_logger().warn('Right camera returned None.')

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit key pressed. Initiating shutdown...")
            self.should_shutdown = True

    def cleanup(self):
        self.get_logger().info("Cleaning up resources...")
        try:
            self.cam_left.camera.Close()
            self.cam_right.camera.Close()
        except Exception as e:
            self.get_logger().error(f"Error closing cameras: {e}")
        self.timer.cancel()
        cv2.destroyAllWindows()
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoShowNode()

    try:
        while rclpy.ok() and not node.should_shutdown:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt received.")
    finally:
        node.cleanup()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
