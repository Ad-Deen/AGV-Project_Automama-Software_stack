import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import os
from collections import namedtuple

# Manual CUDA context
cuda.init()
_device = cuda.Device(0)
_cuda_context = _device.make_context()  # Only one context per process/thread
_cuda_context.pop()  # Keep it dormant until used

HostDeviceMem = namedtuple('HostDeviceMem', 'host device')

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


def get_colormap_for_labels(labels, full_colormap):
    return {label: full_colormap[label] for label in labels if 0 <= label < len(full_colormap)}

class TensorRTInference:
    def __init__(self, engine_path):
        global _cuda_context
        _cuda_context.push()
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
        _cuda_context.pop()

    # def __enter__(self):
    #     _cuda_context.push()
    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     _cuda_context.pop()

    def load_engine(self):
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(self.trt_logger)
        return runtime.deserialize_cuda_engine(engine_data)

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def set_tensor_addresses_once(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

    def get_tensor_info(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_tensor_name = name
                self.input_shape = shape
            else:
                self.output_tensor_name = name
                self.output_shape = shape

    def preprocess_frame(self, frame):
        h, w = self.input_shape[2], self.input_shape[3]
        frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = frame_rgb.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def postprocess(self, output_data, original_shape, labels=None):
        pred = output_data.reshape(self.output_shape)[0, 0]
        if labels is None:
            labels = list(range(len(CITYSCAPES_COLORMAP)))
        colormap = get_colormap_for_labels(labels, CITYSCAPES_COLORMAP)
        mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for label in labels:
            mask[pred == label] = colormap[label]
        return cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    def infer(self, frame):
        global _cuda_context
        _cuda_context.push()
        input_data = self.preprocess_frame(frame)
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        self.stream.synchronize()
        _cuda_context.pop()
        return self.postprocess(self.outputs[0].host, frame.shape)
    
