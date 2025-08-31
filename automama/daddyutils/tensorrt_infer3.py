import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import os
from collections import namedtuple
import time
import cupy as cp
# Manual CUDA context
cuda.init()
_device = cuda.Device(0)
_cuda_context = _device.make_context()  # Only one context per process/thread
_cuda_context.pop()  # Keep it dormant until used

HostDeviceMem = namedtuple('HostDeviceMem', 'host device')

# -------------------Color map ------------------
SEGMENTATION_CLASSES = {
    0: 'person',
    1: 'auto_rickshaw',
    2: 'bicycle',
    3: 'bus',
    4: 'car',
    5: 'cart_vehicle',
    6: 'construction_vehicle',
    7: 'motorbike',
    8: 'priority_vehicle',
    9: 'three_wheeler',
    10: 'train',
    11: 'truck',
    12: 'road'
}

# A suggested new colormap for your 13 classes. 
# You can customize these RGB values as you see fit.
SEGMENTATION_COLORMAP = [
    (0,   0, 255),    # 0 - person (blue)
    (255, 0, 0),      # 1 - auto_rickshaw (red)
    (0, 255, 0),      # 2 - bicycle (green)
    (128, 0, 128),    # 3 - bus (purple)
    (255, 255, 0),    # 4 - car (yellow)
    (0, 255, 255),    # 5 - cart_vehicle (cyan)
    (128, 128, 0),    # 6 - construction_vehicle (olive)
    (255, 0, 255),    # 7 - motorbike (magenta)
    (0, 128, 255),    # 8 - priority_vehicle (sky blue)
    (255, 128, 0),    # 9 - three_wheeler (orange)
    (128, 0, 0),      # 10 - train (maroon)
    (0, 128, 0),      # 11 - truck (dark green)
    (128, 128, 128)   # 12 - road (gray)
]


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

        self.inputs, self.outputs, self.bindings = self.allocate_buffers()

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
        # stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            print(f"Tensor {i}: {name}, Shape: {shape}, Dtype: {dtype}")
            

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings

    def set_tensor_addresses_once(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            print(f"tensor_names {name}") 
            self.context.set_tensor_address(name, self.bindings[i])
            print(f"ptr address_names {self.bindings[i]}")

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
        """Preprocess frame for model input"""
        # Get target dimensions from input shape (NCHW format)
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        
        # Resize frame to target size
        frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = frame_rgb.astype(np.float32) / 255.0
        
        # Transpose HWC → CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension: CHW → NCHW
        img = np.expand_dims(img, axis=0)
        
        return np.ascontiguousarray(img)

    def postprocess(self, detection_output, seg_proto, original_frame_shape, labels_to_show=None, confidence_threshold=0.2):
        """
        Postprocess the detection and segmentation output to create a class-based mask.
        
        Args:
            detection_output: The output0 tensor (1, 49, 8400)
            seg_proto: The output1 tensor (1, 32, 160, 160)
            original_frame_shape: Shape of the original frame (H, W, C)
            labels_to_show: Optional list of labels to include in the mask.
            confidence_threshold: Minimum confidence to consider a detection.

        Returns:
            color_mask: A (H, W, 3) RGB image with color-mapped segmentation
        """
        
        # # Transpose detection_output for easier access
        # detections = detection_output.transpose(0, 2, 1)[0] # Shape: (8400, 49)
        
        # num_classes = len(SEGMENTATION_CLASSES)
        
        # # Extract scores and class IDs
        # scores = detections[:, 4:4 + num_classes]
        # max_scores = np.max(scores, axis=1)
        # class_ids = np.argmax(scores, axis=1)
        
        # # Extract mask coefficients
        # mask_coeffs = detections[:, 4 + num_classes:]

        # # Filter detections based on confidence and labels_to_show
        # valid_indices = (max_scores > confidence_threshold)
        # if labels_to_show is not None:
        #     valid_indices &= np.isin(class_ids, labels_to_show)
        
        # valid_class_ids = class_ids[valid_indices]
        # valid_mask_coeffs = mask_coeffs[valid_indices]
        
        # # If no valid detections, return an empty mask
        # if valid_mask_coeffs.shape[0] == 0:
        #     return np.zeros((original_frame_shape[0], original_frame_shape[1], 3), dtype=np.uint8)
        
        # # --- Start of the refactored, semantic mask generation ---
        
        # # 1. Combine all mask coefficients with the prototypes to get instance logits
        # prototypes = seg_proto[0].reshape(32, -1)
        # instance_masks_logits = (valid_mask_coeffs @ prototypes).reshape(-1, 160, 160)
        
        # # 2. Apply sigmoid and binarize all masks at once
        # sigmoid_masks = 1 / (1 + np.exp(-instance_masks_logits))
        # binary_masks = (sigmoid_masks > 0.55).astype(np.uint8)
        
        # # 3. Create a single final colored mask at low resolution
        # low_res_mask = np.zeros((160, 160, 3), dtype=np.uint8)
        
        # # Iterate through the unique classes, combine masks, and apply colors in one pass
        # for class_id in np.unique(valid_class_ids):
        #     # Get all binary masks for the current class
        #     class_mask_indices = np.where(valid_class_ids == class_id)
        #     masks_for_class = binary_masks[class_mask_indices]
            
        #     # Combine all masks for this class using logical OR
        #     if masks_for_class.shape[0] > 0:
        #         combined_mask = np.any(masks_for_class, axis=0).astype(np.uint8)
                
        #         # Apply the color to the low_res_mask immediately
        #         color = SEGMENTATION_COLORMAP[class_id]
        #         low_res_mask[combined_mask > 0] = color
        # final_color_mask = cv2.resize(low_res_mask, 
        #                             (original_frame_shape[1], original_frame_shape[0]), 
        #                             interpolation=cv2.INTER_NEAREST)
        """
        GPU-accelerated postprocessing using CuPy to create a class-based mask.
        Includes explicit synchronization to prevent hangs.
        """
        with cp.cuda.Device(0):
            # Transfer input data to GPU
            detections = cp.asarray(detection_output.transpose(0, 2, 1)[0]) # Shape: (8400, 49)
            prototypes = cp.asarray(seg_proto[0].reshape(32, -1))
            
            num_classes = len(SEGMENTATION_CLASSES)
            
            scores = detections[:, 4:4 + num_classes]
            max_scores = cp.max(scores, axis=1)
            class_ids = cp.argmax(scores, axis=1)
            mask_coeffs = detections[:, 4 + num_classes:]

            valid_indices = (max_scores > confidence_threshold)
            if labels_to_show is not None:
                cp_labels_to_show = cp.asarray(labels_to_show)
                valid_indices &= cp.isin(class_ids, cp_labels_to_show)
            
            valid_class_ids = class_ids[valid_indices]
            valid_mask_coeffs = mask_coeffs[valid_indices]
            
            if valid_mask_coeffs.shape[0] == 0:
                return np.zeros((original_frame_shape[0], original_frame_shape[1], 3), dtype=np.uint8)
            
            # --- GPU-accelerated mask generation ---
            
            instance_masks_logits = (valid_mask_coeffs @ prototypes).reshape(-1, 160, 160)
            sigmoid_masks = 1 / (1 + cp.exp(-instance_masks_logits))
            binary_masks = (sigmoid_masks > 0.65).astype(cp.uint8)
            
            low_res_mask = cp.zeros((160, 160, 3), dtype=cp.uint8)
            
            for class_id in cp.unique(valid_class_ids).get():
                class_mask_indices = cp.where(valid_class_ids == class_id)
                masks_for_class = binary_masks[class_mask_indices]
                
                if masks_for_class.shape[0] > 0:
                    combined_mask = cp.any(masks_for_class, axis=0).astype(cp.uint8)
                    
                    color = cp.asarray(SEGMENTATION_COLORMAP[class_id])
                    low_res_mask[combined_mask > 0] = color

            # --- IMPORTANT: Explicit synchronization to prevent hangs ---
            # This call forces the CPU to wait for all CuPy operations on the
            # default stream to complete before proceeding.
            cp.cuda.Stream.null.synchronize()

            # Transfer the final mask back to the CPU for OpenCV
            final_color_mask_np = low_res_mask.get()
        final_color_mask = cv2.resize(final_color_mask_np, 
                                  (original_frame_shape[1], original_frame_shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)


        

        return final_color_mask

    
    def infer(self, frame, labels_to_show=[0,1,2,3,4,5,6,7,8,9,10,11,12]):
        """Run inference on a single frame and return selected segmentation mask."""
        global _cuda_context
        
        # --- Preprocess ---
        input_data = self.preprocess_frame(frame)
        
        # Start timer for GPU operations (preprocess + inference + memory copy)
        # gpu_start_time = time.perf_counter()

        _cuda_context.push()
        np.copyto(self.inputs[0].host, input_data.ravel())
        
        stream = cuda.Stream()
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, stream)
        
        # --- Run inference ---
        self.context.execute_async_v3(stream_handle=stream.handle)
        
        # --- Copy BOTH outputs ---
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, stream)
        cuda.memcpy_dtoh_async(self.outputs[1].host, self.outputs[1].device, stream)
        
        # Wait for all GPU operations to finish
        stream.synchronize()
        _cuda_context.pop()
        
        # gpu_duration = time.perf_counter() - gpu_start_time
        # print(f"GPU time         : {gpu_duration:.4f} seconds")

        # --- Get segmentation data & post-process ---
        # post_process_start_time = time.perf_counter()
        
        detection_output = self.outputs[0].host.reshape(1, 49, 8400)
        seg_proto = self.outputs[1].host.reshape(1, 32, 160, 160)
        
        mask = self.postprocess(detection_output, seg_proto, frame.shape, labels_to_show)
        
        # post_process_duration = time.perf_counter() - post_process_start_time
        # print(f"Post process time: {post_process_duration:.4f} seconds")
        
        return mask
    # people --> 11 (BGR color palate)
    #road --> 0

