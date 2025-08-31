import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
import time # For simulating a capture time

class RectifyInputPublisher(Node):
    def __init__(self):
        super().__init__('rectify_input_publisher')

        # Declare parameters for topics and frame IDs (matching your launch file)
        self.declare_parameter('image_topic', '/camera/left/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/left/camera_info')
        self.declare_parameter('camera_frame_id', 'camera_0_optical_frame') # Matching your C++ node's output

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_frame_id = self.get_parameter('camera_frame_id').get_parameter_value().string_value

        self.get_logger().info(f"Publishing images to: {self.image_topic}")
        self.get_logger().info(f"Publishing camera info to: {self.camera_info_topic}")
        self.get_logger().info(f"Using frame_id: {self.camera_frame_id}")

        # Publishers
        self.image_publisher = self.create_publisher(Image, self.image_topic, 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo, self.camera_info_topic, 10)

        self.bridge = CvBridge()

        # Timer to publish messages at a fixed rate (e.g., 30 FPS)
        self.timer_period = 1.0 / 30.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.publish_data)

        # --- Dummy Image Setup (Replace with your actual camera capture) ---
        self.image_width = 960
        self.image_height = 576
        # Create a dummy BGR image (blue rectangle on black background)
        self.dummy_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        cv2.rectangle(self.dummy_image, (self.image_width // 4, self.image_height // 4),
                      (self.image_width * 3 // 4, self.image_height * 3 // 4),
                      (255, 0, 0), -1) # Blue rectangle

        # --- CameraInfo Setup (CRITICAL: REPLACE WITH YOUR ACTUAL CALIBRATION) ---
        self.camera_info_msg = CameraInfo()
        self.camera_info_msg.width = self.image_width
        self.camera_info_msg.height = self.image_height

        # Example Intrinsic Matrix (K) - REPLACE WITH YOUR CALIBRATION
        # fx, 0, cx
        # 0, fy, cy
        # 0, 0, 1
        self.camera_info_msg.k = [
            700.0, 0.0, self.image_width / 2.0,
            0.0, 700.0, self.image_height / 2.0,
            0.0, 0.0, 1.0
        ]

        # Example Distortion Coefficients (D) - REPLACE WITH YOUR CALIBRATION
        # For a rectified camera, D might be all zeros.
        # For a distorted camera, these are typically 5 or 8 values.
        self.camera_info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0] # Example: No distortion

        # Distortion Model - REPLACE WITH YOUR CALIBRATION (e.g., "plumb_bob", "rational_polynomial")
        self.camera_info_msg.distortion_model = "plumb_bob"

        # Projection Matrix (P) - REPLACE WITH YOUR CALIBRATION
        # fx', 0, cx', Tx
        # 0, fy', cy', Ty
        # 0, 0, 1, 0
        # For stereo, Tx and Ty are related to baseline and disparity.
        # For a single camera, Tx, Ty are often 0.
        self.camera_info_msg.p = [
            700.0, 0.0, self.image_width / 2.0, 0.0,
            0.0, 700.0, self.image_height / 2.0, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]
        # Rectification Matrix (R) - For a single camera, typically identity.
        self.camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


    def publish_data(self):
        # Get current time for message headers
        current_ros_time = self.get_clock().now().to_msg()

        # --- Publish Image Message ---
        image_msg = self.bridge.cv2_to_imgmsg(self.dummy_image, encoding='bgr8')
        image_msg.header.stamp = current_ros_time
        image_msg.header.frame_id = self.camera_frame_id
        self.image_publisher.publish(image_msg)

        # --- Publish CameraInfo Message ---
        camera_info_msg = self.camera_info_msg
        camera_info_msg.header.stamp = current_ros_time # IMPORTANT: Same timestamp as image
        camera_info_msg.header.frame_id = self.camera_frame_id # IMPORTANT: Same frame_id as image
        self.camera_info_publisher.publish(camera_info_msg)

        self.get_logger().info(
            f'Published Image (stamp: {image_msg.header.stamp.sec}.{image_msg.header.stamp.nanosec}, '
            f'frame_id: {image_msg.header.frame_id}) and CameraInfo.'
        )

def main(args=None):
    rclpy.init(args=args)
    node = RectifyInputPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()