import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os

from message_filters import Subscriber, ApproximateTimeSynchronizer

class StereoRectifier(Node):
    def __init__(self):
        super().__init__('stereo_rectifier_node')
        self.get_logger().info("StereoRectifier Node has been started.")

        # Declare a parameter for the calibration file path
        # You would typically place your 'stereo_calibration.yaml' in a 'resource'
        # directory within your ROS2 package, and set up your package's setup.py
        # to install it to the share directory.
        # Example: 'install/share/your_package_name/resource/stereo_calibration.yaml'
        self.declare_parameter(
            'calibration_file_path',
            '/home/deen/ros2_ws/src/stereo_calibration_results.yaml'
        ) # Default relative path
        self.calibration_file_path = self.get_parameter('calibration_file_path').get_parameter_value().string_value

        self.bridge = CvBridge()

        # Members to store rectification maps (initialized once)
        self.map1_x, self.map1_y = None, None
        self.map2_x, self.map2_y = None, None
        self.img_width, self.img_height = None, None

        # Load calibration data immediately on node creation
        if not self._load_calibration_data():
            self.get_logger().error("Failed to load calibration data. Please check the file path and content. Exiting node.")
            # It's good practice to ensure core setup is done before continuing.
            # In a real application, you might want to wait for the file, or retry.
            return # Node won't be fully functional without calibration

        # Define QoS profile for image topics
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # Or ReliabilityPolicy.RELIABLE for guaranteed delivery
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # Keep a small buffer
        )

        # Subscribers for raw left and right image topics
        self.sub_right = Subscriber(self, Image, '/csi_cam_right', qos_profile=qos_profile)
        self.sub_left = Subscriber(self, Image, '/csi_cam_left', qos_profile=qos_profile)

        # Approximate Time Synchronizer for image pairs
        # queue_size: How many messages to store in the queue for synchronization
        # slop: Maximum acceptable time difference (seconds) between messages to be synchronized
        self.ts = ApproximateTimeSynchronizer([self.sub_right, self.sub_left], queue_size=10, slop=0.05)
        self.ts.registerCallback(self.image_callback)

        # Publishers for rectified image topics
        self.pub_right_rect = self.create_publisher(Image, '/right_rect_img', qos_profile)
        self.pub_left_rect = self.create_publisher(Image, '/left_rect_img', qos_profile)

        self.get_logger().info("Subscribing to /csi_cam_right and /csi_cam_left. Publishing to /right_rect_img and /left_rect_img.")
        self.get_logger().info("Waiting for first image pair to determine resolution and initialize rectification maps...")

    def _load_calibration_data(self):
        """Loads camera calibration matrices from a YAML file."""
        # This assumes your YAML file 'stereo_calibration.yaml' is either in the
        # current working directory when you run the node, or you pass its full path
        # via the 'calibration_file_path' parameter.
        if not os.path.exists(self.calibration_file_path):
            self.get_logger().error(f"Calibration file not found at: '{self.calibration_file_path}'")
            return False

        try:
            with open(self.calibration_file_path, 'r') as file:
                calib_data = yaml.safe_load(file)

            # Access the 'camera_matrices' key, which contains all your matrices
            matrices = calib_data.get('camera_matrices', {})
            if not matrices:
                self.get_logger().error("YAML file does not contain 'camera_matrices' key or it is empty.")
                return False

            # Convert loaded lists to NumPy arrays with float64 dtype
            self.M1 = np.array(matrices.get('M1'), dtype=np.float64)
            self.D1 = np.array(matrices.get('D1'), dtype=np.float64)
            self.M2 = np.array(matrices.get('M2'), dtype=np.float64)
            self.D2 = np.array(matrices.get('D2'), dtype=np.float64)
            self.R1 = np.array(matrices.get('R1'), dtype=np.float64)
            self.R2 = np.array(matrices.get('R2'), dtype=np.float64)
            self.P1 = np.array(matrices.get('P1'), dtype=np.float64)
            self.P2 = np.array(matrices.get('P2'), dtype=np.float64)

            # Optional: Add checks to ensure matrices are not None and have correct shapes
            if any(m is None for m in [self.M1, self.D1, self.M2, self.D2, self.R1, self.R2, self.P1, self.P2]):
                self.get_logger().error("One or more required calibration matrices are missing or malformed in the YAML file.")
                return False

            self.get_logger().info("Calibration data loaded successfully.")
            # Uncomment below to print loaded matrices for debugging:
            # self.get_logger().info(f"M1:\n{self.M1}")
            # self.get_logger().info(f"D1:\n{self.D1}")
            # self.get_logger().info(f"M2:\n{self.M2}")
            # self.get_logger().info(f"D2:\n{self.D2}")
            # self.get_logger().info(f"R1:\n{self.R1}")
            # self.get_logger().info(f"R2:\n{self.R2}")
            # self.get_logger().info(f"P1:\n{self.P1}")
            # self.get_logger().info(f"P2:\n{self.P2}")

            return True

        except Exception as e:
            self.get_logger().error(f"Error loading calibration data from '{self.calibration_file_path}': {e}")
            return False

    def _init_rectification_maps(self, img_width, img_height):
        """
        Initializes the undistortion and rectification mapping matrices.
        This should be called once after determining the image dimensions.
        """
        self.img_width = img_width
        self.img_height = img_height

        self.get_logger().info(f"Initializing rectification maps for image size: {img_width}x{img_height}")

        # cv2.initUndistortRectifyMap calculates the pixel transformations needed
        # It takes:
        # 1. Original Camera Matrix (e.g., M1)
        # 2. Original Distortion Coefficients (e.g., D1)
        # 3. Rectification Rotation Matrix (e.g., R1)
        # 4. New Projection Matrix (e.g., P1)
        # 5. Desired new image size (width, height)
        # 6. Map type (cv2.CV_32FC1 is common for floating point maps)
        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(
            self.M1, self.D1, self.R1, self.P1,
            (self.img_width, self.img_height), cv2.CV_32FC1
        )
        self.map2_x, self.map2_y = cv2.initUndistortRectifyMap(
            self.M2, self.D2, self.R2, self.P2,
            (self.img_width, self.img_height), cv2.CV_32FC1
        )
        self.get_logger().info("Rectification maps successfully initialized.")

    def image_callback(self, right_img_msg, left_img_msg):
        """
        Callback function for synchronized left and right image messages.
        Performs rectification and publishes rectified images.
        """
        # Initialize rectification maps on the first received frame
        if self.map1_x is None or self.map2_x is None:
            # Check if image dimensions are available and valid
            if right_img_msg.width > 0 and right_img_msg.height > 0:
                self._init_rectification_maps(right_img_msg.width, right_img_msg.height)
            else:
                self.get_logger().warn("Received image with zero dimensions. Skipping map initialization.")
                return

            if self.map1_x is None: # If init_rectification_maps still failed
                self.get_logger().error("Rectification maps are still not initialized. Skipping current frame.")
                return

        try:
            # Convert ROS Image messages to OpenCV format (BGR8 for color images)
            # Adjust encoding if your camera publishes grayscale (e.g., 'mono8')
            cv_right_img = self.bridge.imgmsg_to_cv2(right_img_msg, desired_encoding='bgr8')
            cv_left_img = self.bridge.imgmsg_to_cv2(left_img_msg, desired_encoding='bgr8')

            # Ensure current image dimensions match the dimensions used for map initialization
            # This is important if your camera resolution might change or if the first frame was an outlier
            if cv_right_img.shape[1] != self.img_width or cv_right_img.shape[0] != self.img_height:
                self.get_logger().warn(f"Image dimension mismatch! Expected {self.img_width}x{self.img_height}, got {cv_right_img.shape[1]}x{cv_right_img.shape[0]}. Re-initializing maps.")
                # Re-initialize maps for the new resolution
                self._init_rectification_maps(cv_right_img.shape[1], cv_right_img.shape[0])
                if self.map1_x is None: # Check if re-initialization was successful
                    self.get_logger().error("Rectification maps re-initialization failed. Skipping current frame.")
                    return

            # Apply the rectification maps to the images
            # cv2.remap performs undistortion and rectification in one go
            rect_right_img = cv2.remap(cv_right_img, self.map1_x, self.map1_y, cv2.INTER_LINEAR)
            rect_left_img = cv2.remap(cv_left_img, self.map2_x, self.map2_y, cv2.INTER_LINEAR)

            # Convert rectified OpenCV images back to ROS Image messages
            rect_right_msg = self.bridge.cv2_to_imgmsg(rect_right_img, encoding='bgr8')
            rect_left_msg = self.bridge.cv2_to_imgmsg(rect_left_img, encoding='bgr8')

            # Preserve the original header (timestamp and frame_id) for rectified images
            rect_right_msg.header = right_img_msg.header
            rect_left_msg.header = left_img_msg.header

            # Publish the rectified images
            self.pub_right_rect.publish(rect_right_msg)
            self.pub_left_rect.publish(rect_left_msg)

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = StereoRectifier() # Create an instance of the StereoRectifier node
    try:
        rclpy.spin(node) # Keep the node alive and processing callbacks
    except KeyboardInterrupt:
        pass # Allow clean shutdown on Ctrl+C
    finally:
        node.destroy_node() # Clean up the node
        rclpy.shutdown() # Shut down rclpy

if __name__ == '__main__':
    main()