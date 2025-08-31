import vpi
import cv2
import cupy as cp
import numpy as np
import jetson_utils
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class JetsonCameraHandler:
    def __init__(self, sensor_id, warp_map):
        # self.node = node
        self.sensor_id = sensor_id
        # self.topic_name = topic_name
        # self.bridge = CvBridge()
        # self.publisher = node.create_publisher(Image, topic_name, 10)

        self.camera = jetson_utils.gstCamera(640, 480, str(sensor_id))
        self.camera.Open()

        self.stream = vpi.Stream()
        self.warp_map = warp_map

    def capture_and_process(self):
        try:
            with vpi.Backend.CUDA:
                with self.stream:
                    frame, _, _ = self.camera.CaptureRGBA(zeroCopy=1)
                    if frame is None:
                        self.node.get_logger().warn(f"Camera {self.sensor_id} returned no frame")
                        return None

                    array = cp.asarray(frame, dtype=cp.uint8)
                    rgb_array = array[:, :, :3]
                    #dor VPI SGBM vpi.Format.Y16_ER_BL
                    # custom stereo vpi.Format.U8
                    vpi_img = vpi.asimage(cp.asnumpy(rgb_array))
                    rectified = vpi_img.remap(self.warp_map)\
                        .convert(vpi.Format.BGR8)\
                        # .eqhist()\
                        # .rescale((640,480), interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)

                    self.stream.sync()  # Ensure VPI ops complete

                    rectified_cpu = rectified.cpu()  # Transfer to CPU for ROS or OpenCV

                    # Publish ROS Image message
                    # msg = self.bridge.cv2_to_imgmsg(rectified_cpu, encoding='bgr8')
                    # self.publisher.publish(msg)

                    return np.asarray(rectified_cpu)  # Return for display or further processing

        except Exception as e:
            self.node.get_logger().error(f"Error in capture_and_publish: {e}")
            return None

    def close(self):
        try:
            self.camera.Close()
        except Exception as e:
            self.node.get_logger().error(f"Error closing camera {self.sensor_id}: {e}")
