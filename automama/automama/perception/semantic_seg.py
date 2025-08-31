
import rclpy
from rclpy.node import Node
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os
from collections import namedtuple

# -------------------Color map ------------------
CITYSCAPES_COLORMAP = np.array([
    [128, 64,128],  # road
    [244, 35,232],  # sidewalk
    [ 70, 70, 70],  # building
    [102,102,156],  # wall
    [190,153,153],  # fence
    [153,153,153],  # pole
    [250,170, 30],  # traffic light
    [220,220,  0],  # traffic sign
    [107,142, 35],  # vegetation
    [152,251,152],  # terrain
    [ 70,130,180],  # sky
    [220, 20, 60],  # person
    [255,  0,  0],  # rider
    [  0,  0,142],  # car
    [  0,  0, 70],  # truck
    [  0, 60,100],  # bus
    [  0, 80,100],  # train
    [  0,  0,230],  # motorcycle
    [119, 11, 32]   # bicycle
], dtype=np.uint8)

HostDeviceMem = namedtuple('HostDeviceMem', 'host device')

class TensorRTInference:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        self.input_tensor_name = None
        self.output_tensor_name = None
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
        inputs = []
        outputs = []
        bindings = []
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
                self.input_tensor_name = tensor_name
                self.input_shape = tensor_shape
            else:
                self.output_tensor_name = tensor_name
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
        pred = output_reshaped[0, 0]  # (H, W)
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


class TRTSegmentationNode(Node):
    def __init__(self):
        super().__init__('trt_segmentation_node')

        self.engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/PID net models/engine/pidnet_L_480x640_32f.engine"
        self.video_path = "/home/deen/ros2_ws/src/automama/automama/perception/killo_road.mp4"

        self.trt_inference = TensorRTInference(self.engine_path)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {self.video_path}")
            rclpy.shutdown()
            return

        self.timer_period = 0.03  # ~30 FPS
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        self.frame_idx = 0
        self.total_time = 0

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video reached or failed to read frame.")
            self.cap.release()
            rclpy.shutdown()
            return
        
        start_time = time.time()
        mask = self.trt_inference.infer(frame)
        inference_time = time.time() - start_time
        self.total_time += inference_time
        
        blended = cv2.addWeighted(frame, 0.2, mask, 0.8, 0)
        fps_text = f"FPS: {1.0/inference_time:.1f} | Frame: {self.frame_idx+1}"
        cv2.putText(blended, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.imshow("TensorRT Segmentation", blended)
        self.frame_idx += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("User requested exit")
            self.cap.release()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TRTSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user, shutting down...')
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.get_logger().info(f"Processed {node.frame_idx} frames")
        if node.frame_idx > 0:
            avg_fps = node.frame_idx / node.total_time
            node.get_logger().info(f"Average FPS: {avg_fps:.2f}")

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
