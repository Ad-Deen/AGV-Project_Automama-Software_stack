import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class StereoRollingBufferNode(Node):
    def __init__(self):
        super().__init__('stereo_rolling_buffer_node')
        self.bridge = CvBridge()

        self.buffer = [None, None]  # index 0 = left, 1 = right

        self.latest_left = None
        self.latest_right = None

        self.show_index = 0  # Which buffer index to show next (0 or 1)
        self.loop_counter = 0

        self.create_subscription(Image, '/csi_cam_left', self.left_callback, 10)
        self.create_subscription(Image, '/csi_cam_right', self.right_callback, 10)

        self.create_timer(1/30, self.timer_callback)  # 2 Hz for demo, change as needed

    def left_callback(self, msg):
        try:
            self.latest_left = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Left image decode failed: {e}")

    def right_callback(self, msg):
        try:
            self.latest_right = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Right image decode failed: {e}")

    def timer_callback(self):
        self.loop_counter += 1

        # Insert new frames only on odd loops (1,3,5,...)
        if self.loop_counter % 2 == 1:
            if self.latest_left is not None and self.latest_right is not None:
                self.buffer[0] = self.latest_left.copy()
                self.buffer[1] = self.latest_right.copy()
                # self.get_logger().info("Inserted new left & right frames into buffer")

        # Show current buffer frame
        frame_to_show = self.buffer[self.show_index]
        
        #============== Inference here===========================
        


        #==========================================================
        if frame_to_show is not None:
            cv2.imshow("Rolling Stereo Buffer Frame", frame_to_show)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                rclpy.shutdown()
                return

        # Alternate index for next display
        self.show_index = 1 - self.show_index


def main(args=None):
    rclpy.init(args=args)
    node = StereoRollingBufferNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
