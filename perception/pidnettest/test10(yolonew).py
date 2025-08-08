import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os
from collections import namedtuple
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


def get_colormap_for_labels(labels_to_show, full_colormap, invalid_color=(0, 0, 0)):
    """
    Generate a custom colormap where only the specified labels are shown.
    All other labels are set to `invalid_color`.
    If labels_to_show is empty, return the full colormap unchanged.
    """
    if not labels_to_show:  # Empty list: show all labels
        return full_colormap

    colormap = []
    for idx in range(len(full_colormap)):
        if idx in labels_to_show:
            colormap.append(full_colormap[idx])
        else:
            colormap.append(invalid_color)

    return colormap
def softmax(x, axis=0):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ---------------- Helper Classes ----------------
HostDeviceMem = namedtuple('HostDeviceMem', 'host device')

'''TensorRT version: 10.3.0
Loading engine from: /home/deen/ros2_ws/src/automama/automama/perception/PID net models/engine/pidnet_L_480x640_32f.engine
Tensor 0: input, Shape: (1, 3, 480, 640), Dtype: <class 'numpy.float32'>
Tensor 1: output, Shape: (1, 1, 480, 640), Dtype: <class 'numpy.int64'>
Input tensor: input, Shape: (1, 3, 480, 640)
Output tensor: output, Shape: (1, 1, 480, 640)
tensor_names input
ptr address_names 8860753920
tensor_names output
ptr address_names 8866897920
TensorRT engine loaded successfully'''

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
        self.set_tensor_addresses_once()  # <- Set once here
    
    def set_tensor_addresses_once(self):
        """Set tensor device addresses once for fixed input/output"""
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)        #input, ptr addr= ******
            print(f"tensor_names {tensor_name}")                #output, ptraddr =******
            # print(f"tensor_names {tensor_name}")
            self.context.set_tensor_address(tensor_name, self.bindings[i])
            print(f"ptr address_names {self.bindings[i]}")



        
    def load_engine(self):
        """Load TensorRT engine from file"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine file not found: {self.engine_path}")
        
        try:
            with open(self.engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
            
            return engine
            
        except Exception as e:
            print(f"Error loading engine: {e}")
            raise
    
    def allocate_buffers(self):
        """Allocate all buffers required for the engine using new API"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            size = trt.volume(tensor_shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            print(f"Tensor {i}: {tensor_name}, Shape: {tensor_shape}, Dtype: {dtype}")
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer address to bindings
            bindings.append(int(device_mem))
            
            # Append to appropriate input/output list
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def get_tensor_info(self):
        """Get input and output tensor information"""
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_tensor_name = tensor_name
                self.input_shape = tensor_shape
                print(f"Input tensor: {tensor_name}, Shape: {tensor_shape}")
            else:
                self.output_tensor_name = tensor_name
                self.output_shape = tensor_shape
                print(f"Output tensor: {tensor_name}, Shape: {tensor_shape}")
    
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
    
    def postprocess(self, detection_output, seg_proto, original_frame_shape, labels_to_show=None, confidence_threshold=0.1):
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
        
        # Transpose detection_output for easier access
        detections = detection_output.transpose(0, 2, 1)[0] # Shape: (8400, 49)
        
        # Define the number of classes from your dictionary
        num_classes = len(SEGMENTATION_CLASSES)
        
        # YOLOv8-seg output format: [bbox(4), scores(num_classes), mask_coeffs(32)]
        # The number of classes is 34 based on your SEGMENTATION_CLASSES dict.
        num_classes = len(SEGMENTATION_CLASSES)
        
        # Extract scores and class IDs
        # Scores are from index 4 to 4 + num_classes - 1
        scores = detections[:, 4:4 + num_classes]
        
        # Find the maximum score for each detection and its corresponding class ID
        max_scores = np.max(scores, axis=1)
        # print(max_scores)

        class_ids = np.argmax(scores, axis=1)
        # print(len(class_ids))
        # Extract mask coefficients
        # Coefficients are from index 4 + num_classes to the end (49 - 1)
        mask_coeffs = detections[:, 4 + num_classes:]
        # print(mask_coeffs)
        # Filter detections based on confidence and labels_to_show
        valid_indices = (max_scores > confidence_threshold)
        if labels_to_show is not None:
            valid_indices &= np.isin(class_ids, labels_to_show)
        
        # valid_detections = detections[valid_indices]
        valid_class_ids = class_ids[valid_indices]
        # print(valid_class_ids)
        # print(valid_indices)
        valid_mask_coeffs = mask_coeffs[valid_indices]
        # If no valid detections, return an empty mask
        if valid_mask_coeffs.shape[0] == 0:
            return np.zeros((original_frame_shape[0], original_frame_shape[1], 3), dtype=np.uint8)
        # Initialize the final color mask
        # color_mask = np.zeros((original_frame_shape[0], original_frame_shape[1], 3), dtype=np.uint8)
        
        # --- Start of the refactored, vectorized mask generation ---
        
        # 1. Combine all mask coefficients with the prototypes
        # Reshape prototypes to (32, 160*160) for matrix multiplication
        prototypes = seg_proto[0].reshape(32, -1)
        
        # Combine coeffs and prototypes
        # valid_mask_coeffs shape: (N, 32)
        # prototypes shape: (32, 160*160)
        # result shape: (N, 160*160) where N is the number of valid detections
        instance_masks_logits = (valid_mask_coeffs @ prototypes).reshape(-1, 160, 160)
        
        # Reshape back to (N, 160, 160)
        # instance_masks_logits = instance_masks_logits
        
        # 2. Apply sigmoid and binarize all masks at once
        # Apply sigmoid to all logits
        sigmoid_masks = 1 / (1 + np.exp(-instance_masks_logits))
        print(sigmoid_masks.shape)
        # Binarize all masks
        binary_masks = (sigmoid_masks > 0.75).astype(np.uint8)
        
        # 3. Create a single combined mask at low resolution
        # Get the class IDs and corresponding colors
        valid_colors = np.array([SEGMENTATION_COLORMAP[cid] for cid in valid_class_ids])
        
        # We create a single output mask at the low resolution (160, 160, 3)
        # and fill it by layering the individual masks
        low_res_mask = np.zeros((160, 160, 3), dtype=np.uint8)
        
        # Iterate through each mask and color it. This loop is fast because it's only for coloring,
        # and not for the expensive matrix multiplication or sigmoid ops.
        for i in range(binary_masks.shape[0]):
            low_res_mask[binary_masks[i] > 0] = valid_colors[i]

        # 4. Upsample the single combined mask to the original frame size
        final_color_mask = cv2.resize(low_res_mask, 
                                    (original_frame_shape[1], original_frame_shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)

        return final_color_mask

    
    def infer(self, frame, labels_to_show=[0,1,2,3,4,5,6,7,8,9,10,11,12]):
        """Run inference on a single frame and return selected segmentation mask."""
        
        # --- Preprocess ---
        input_data = self.preprocess_frame(frame)
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        
        # --- Run inference ---
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # --- Copy BOTH outputs ---
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)  # <-- ADD THIS LINE for output0
        cuda.memcpy_dtoh_async(self.outputs[1].host, self.outputs[1].device, self.stream)  # <-- This line is for output1
        self.stream.synchronize()
        
        # --- Get segmentation data ---
        # Reshape and get both outputs from the host buffers
        detection_output = self.outputs[0].host.reshape(1, 49, 8400) # Output0
        seg_proto = self.outputs[1].host.reshape(1, 32, 160, 160)    # Output1
        
        # Pass BOTH outputs to the new postprocess function
        mask = self.postprocess(detection_output, seg_proto, frame.shape, labels_to_show)

        return mask

# ---------------- Main Video Processing ----------------
def main():
    # Initialize TensorRT inference
    engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/yolov8n640x640.engine"
    video_path = "/home/deen/ros2_ws/src/automama/automama/perception/run3.mp4"
    
    print(f"TensorRT version: {trt.__version__}")
    print(f"Loading engine from: {engine_path}")
    
    try:
        # Initialize inference engine
        trt_inference = TensorRTInference(engine_path)
        print("TensorRT engine loaded successfully")
        
        # Open video
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Failed to open video!")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {frame_count} frames at {fps:.2f} FPS")
        
        frame_idx = 0
        total_time = 0
        
        print("Starting inference... Press 'q' to quit")
        # np.set_printoptions(threshold=1000)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            start_time = time.time()
            
            # frame= cv2.resize(frame,(1280,720))
            # print(frame.shape)
            # Run inference
            mask = trt_inference.infer(frame)
            # print(np.array2string(mask))
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Create overlay
            blended = cv2.addWeighted(frame, 0.4, mask, 0.6, 0)
            
            # Add performance info
            fps_text = f"FPS: {1.0/inference_time:.1f} | Frame: {frame_idx+1}/{frame_count}"
            cv2.putText(blended, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display results
            cv2.imshow("TensorRT Segmentation", blended)
            
            # Optional: Show mask separately
            # cv2.imshow("Mask", mask)
            
            frame_idx += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        # Print performance stats
        if 'frame_idx' in locals() and frame_idx > 0:
            avg_fps = frame_idx / total_time
            print(f"\nPerformance Summary:")
            print(f"Processed {frame_idx} frames")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Average inference time: {total_time/frame_idx*1000:.1f} ms per frame")

if __name__ == "__main__":
    main()