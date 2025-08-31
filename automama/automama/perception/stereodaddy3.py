import rclpy
from rclpy.node import Node
import cv2
from daddyutils.camera_handler import JetsonCameraHandler
from daddyutils.warp_utils import load_and_generate_warp_maps
from daddyutils.tensorrt_infer2 import TensorRTInference
# import pycuda.autoinit
import cupy as cp
import pycuda.driver as cuda
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

        # Load calibration & generate warp maps
        calib_dir = "/home/deen/ros2_ws/src/automama/automama/perception/stereo_vision_test/single_cam_KD_callib"
        warpl, warpr = load_and_generate_warp_maps(calib_dir)
        engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/PID net models/engine/pidnet_L_480x640_32f.engine"
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

        self.should_shutdown = False

    def timer_callback(self):
        global context

        if self.should_shutdown:
            return  # Skip processing if shutdown triggered
        ret, frame = self.cap.read()    #video check
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
        context.push()
        mask = self.trt_infer.infer(frame) 
        context.pop()
        frame = cv2.resize(frame,(640,480))
        cv2.imshow("Vid Camera", frame)
        with cp.cuda.Device(0):
            # with cp.cuda.Stream(null=True):
            masked_frame = cp.where(cp.asarray(mask)[..., None] == 11, cp.asarray(frame), 0)
            np_mask = cp.asnumpy(masked_frame)
        cv2.imshow("mask Camera", np_mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit key pressed. Initiating shutdown...")
            self.should_shutdown = True

    def cleanup(self):
        self.get_logger().info("Cleaning up resources...")
        try:
            self.cam_left.close()   # Use the close() method you defined in the handler
            self.cam_right.close()
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
