import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from collections import namedtuple

CITYSCAPES_COLORMAP = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
], dtype=np.uint8)

HostDeviceMem = namedtuple('HostDeviceMem', 'host device')

class TensorRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.input_shape = None
        self.output_shape = None
        self.get_tensor_info()
        self.set_tensor_addresses_once()
    
    def set_tensor_addresses_once(self):
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])

    def load_engine(self):
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(self.trt_logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        return engine
    
    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    
    def get_tensor_info(self):
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_shape = tensor_shape
            else:
                self.output_shape = tensor_shape
    
    def preprocess_frame(self, frame):
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = frame_rgb.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)
    
    def postprocess(self, output_data, original_frame_shape):
        output_reshaped = output_data.reshape(self.output_shape)  # (1, 1, H, W)
        pred = output_reshaped[0, 0]
        color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for label in range(len(CITYSCAPES_COLORMAP)):
            color_mask[pred == label] = CITYSCAPES_COLORMAP[label]
        resized_mask = cv2.resize(color_mask, (original_frame_shape[1], original_frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        return resized_mask

    def infer(self, frame):
        input_data = self.preprocess_frame(frame)
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        self.stream.synchronize()
        mask = self.postprocess(self.outputs[0].host, frame.shape)
        return mask


class StereoRollingBufferTRTNode(Node):
    def __init__(self):
        super().__init__('stereo_rolling_buffer_trt_node')
        self.bridge = CvBridge()

        # Buffer holds left (index 0) and right (index 1) frames
        self.buffer = [None, None]

        self.latest_left = None
        self.latest_right = None
        self.leftmask_publisher = self.create_publisher(Image, '/left_mask', 10)
        self.rightmask_publisher = self.create_publisher(Image, '/right_mask', 10)
        self.state = 'waiting_for_frames'  # states: waiting_for_frames, processing_left, processing_right

        # Subscriptions
        self.create_subscription(Image, '/csi_cam_left', self.left_callback, 10)
        self.create_subscription(Image, '/csi_cam_right', self.right_callback, 10)

        # TensorRT model
        self.engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/PID net models/engine/pidnet_L_480x640_32f.engine"
        self.trt_infer = TensorRTInference(self.engine_path)

        self.frame_idx = 0
        self.total_time = 0

        # Initial timer period (very small to check for frames)
        self.timer_period = 0.01
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

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
        if self.state == 'waiting_for_frames':
            if self.latest_left is not None and self.latest_right is not None:
                self.buffer[0] = self.latest_left.copy()
                self.buffer[1] = self.latest_right.copy()
                self.latest_left = None
                self.latest_right = None
                self.state = 'processing_left'

        elif self.state == 'processing_left':
            frame = self.buffer[0]
            if frame is not None:
                start = time.time()
                mask = self.trt_infer.infer(frame)
                inf_time = time.time() - start
                self.total_time += inf_time

                # Convert color mask to grayscale if needed
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                masked_frame_left = cv2.bitwise_and(frame, frame, mask=mask)

                # Show frame
                # cv2.imshow("TRT Segmentation (Left)", masked_frame_left)
                # key = cv2.waitKey(1)
                # if key == ord('q'):
                #     cv2.destroyAllWindows()
                #     rclpy.shutdown()
                #     return

                # Publish
                msgleft = self.bridge.cv2_to_imgmsg(masked_frame_left, encoding='bgr8')
                self.leftmask_publisher.publish(msgleft)

                self.frame_idx += 1
                self.state = 'processing_right'
                self.timer_period = max(inf_time, 0.001)
                self.timer.reset()

        elif self.state == 'processing_right':
            frame = self.buffer[1]
            if frame is not None:
                start = time.time()
                mask = self.trt_infer.infer(frame)
                inf_time = time.time() - start
                self.total_time += inf_time

                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                masked_frame_right = cv2.bitwise_and(frame, frame, mask=mask)

                # cv2.imshow("TRT Segmentation (Right)", masked_frame_right)
                # key = cv2.waitKey(1)
                # if key == ord('q'):
                #     cv2.destroyAllWindows()
                #     rclpy.shutdown()
                #     return

                # Publish
                msgright = self.bridge.cv2_to_imgmsg(masked_frame_right, encoding='bgr8')
                self.rightmask_publisher.publish(msgright)

                self.frame_idx += 1
                self.state = 'waiting_for_frames'
                self.buffer = [None, None]
                self.timer_period = max(inf_time, 0.001)
                self.timer.reset()


def main(args=None):
    rclpy.init(args=args)
    node = StereoRollingBufferTRTNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user, shutting down...')
    finally:
        cv2.destroyAllWindows()
        if node.frame_idx > 0:
            avg_fps = node.frame_idx / node.total_time
            node.get_logger().info(f"Average inference FPS: {avg_fps:.2f}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
