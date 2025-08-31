import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import vpi
import cv2

class StereoDisparityFromMasks(Node):
    def __init__(self):
        super().__init__('stereo_disparity_from_masks')

        self.bridge = CvBridge()
        self.left_mask = None
        self.right_mask = None

        self.create_subscription(Image, '/left_mask', self.left_mask_callback, 10)
        self.create_subscription(Image, '/right_mask', self.right_mask_callback, 10)

        self.timer = self.create_timer(0.03, self.process_stereo)

    def left_mask_callback(self, msg):
        try:
            self.left_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Left mask conversion failed: {e}")

    def right_mask_callback(self, msg):
        try:
            self.right_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Right mask conversion failed: {e}")

    def process_stereo(self):
        if self.left_mask is None or self.right_mask is None:
            return

        try:
            # Convert to VPI images from numpy (CPU-host memory)
            with vpi.Backend.CUDA:
                left_vpi = vpi.asimage(self.left_mask)
                right_vpi = vpi.asimage(self.right_mask)

                # Perform stereo disparity
                disparity_vpi = left_vpi.stereodisp(right_vpi, window=9, max_disparity=64)

                # Convert VPI image back to NumPy
                disparity_np = disparity_vpi.cpu().asnumpy()

                # Normalize for display
                disp_vis = cv2.normalize(disparity_np, None, 0, 255, cv2.NORM_MINMAX)
                disp_vis = np.uint8(disp_vis)

                cv2.imshow("Stereo Disparity (mask-based)", disp_vis)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Stereo processing failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = StereoDisparityFromMasks()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
