import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import re

VIDEO_DIR = '/home/deen/ros2_ws/src/automama/automama/interface/'
VIDEO_PREFIX = 'video'
VIDEO_SUFFIX = '.mp4'
RECORD_DURATION_SEC = 180  # 3 minutes

class VideoRecorderNode(Node):
    def __init__(self):
        super().__init__('video_recorder_node')
        self.bridge = CvBridge()
        self.video_writer = None
        self.frame_size = None
        self.fps = 30.0
        self.output_file = self.get_next_output_filename()
        self.recording = True
        self.start_time = None

        self.subscription = self.create_subscription(
            Image,
            '/csi_cam_0',
            self.image_callback,
            10
        )

        self.get_logger().info(f"Recording started: {self.output_file}. Press 'q' to save and stop early.")

    def get_next_output_filename(self):
        existing_files = os.listdir(VIDEO_DIR)
        max_index = 0
        pattern = re.compile(f"{VIDEO_PREFIX}(\\d+){VIDEO_SUFFIX}")
        for filename in existing_files:
            match = pattern.match(filename)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
        next_index = max_index + 1
        return os.path.join(VIDEO_DIR, f"{VIDEO_PREFIX}{next_index}{VIDEO_SUFFIX}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # frame = cv2.flip(frame, 0)

            if self.video_writer is None:
                self.frame_size = (frame.shape[1], frame.shape[0])
                self.video_writer = cv2.VideoWriter(
                    self.output_file,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    self.fps,
                    self.frame_size
                )
                self.start_time = time.time()

            # Write frame
            if self.recording:
                self.video_writer.write(frame)

            # Show frame
            cv2.imshow("Recording from /csi_cam_0", frame)
            key = cv2.waitKey(1) & 0xFF

            # Stop if 3 minutes passed or 'q' pressed
            if time.time() - self.start_time >= RECORD_DURATION_SEC or key == ord('q'):
                self.get_logger().info("Stopping recording...")
                self.stop_recording()

        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")

    def stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.video_writer:
            node.video_writer.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
