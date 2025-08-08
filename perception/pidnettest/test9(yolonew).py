import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
import os
from collections import namedtuple

# Vehicle classes for YOLO model
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

SEGMENTATION_COLORMAP = [
    (0, 0, 70),      # 0 - person
    (244, 35, 232),  # 1 - auto_rickshaw
    (70, 70, 70),    # 2 - bicycle
    (153, 153, 153), # 3 - bus
    (250, 170, 30),  # 4 - car
    (220, 220, 0),   # 5 - cart_vehicle
    (107, 142, 35),  # 6 - construction_vehicle
    (255, 0, 0),     # 7 - motorbike
    (0, 0, 230),     # 8 - priority_vehicle
    (200, 128, 50),  # 9 - three_wheeler
    (128, 0, 255),   # 10 - train
    (128, 0, 0),     # 11 - truck
    (81, 0, 81)      # 12 - road
]

HostDeviceMem = namedtuple('HostDeviceMem', 'host device')

class YOLOTensorRTInference:
    def __init__(self, engine_path, conf_threshold=0.25, iou_threshold=0.45, mask_threshold=0.5):
        self.engine_path = engine_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.mask_threshold = mask_threshold
        self.num_classes = 13  # Based on your class list
        
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()
        
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.get_tensor_info()
        self.set_tensor_addresses_once()
    
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
        """Allocate all buffers required for the engine"""
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
            
            bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
    
    def set_tensor_addresses_once(self):
        """Set tensor device addresses once"""
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(tensor_name, self.bindings[i])
            print(f"Set tensor {tensor_name} address: {self.bindings[i]}")
    
    def get_tensor_info(self):
        """Get input and output tensor information"""
        self.input_shapes = {}
        self.output_shapes = {}
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_shape = self.engine.get_tensor_shape(tensor_name)
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_shapes[tensor_name] = tensor_shape
                print(f"Input tensor: {tensor_name}, Shape: {tensor_shape}")
            else:
                self.output_shapes[tensor_name] = tensor_shape
                print(f"Output tensor: {tensor_name}, Shape: {tensor_shape}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for YOLO model input"""
        # Assuming input shape is (1, 3, 640, 640)
        input_shape = list(self.input_shapes.values())[0]
        target_h, target_w = input_shape[2], input_shape[3]
        
        original_shape = frame.shape[:2]  # (height, width)
        
        # Calculate scale factor to maintain aspect ratio
        scale = min(target_w / original_shape[1], target_h / original_shape[0])
        
        # Calculate new dimensions
        new_width = int(original_shape[1] * scale)
        new_height = int(original_shape[0] * scale)
        
        # Resize image
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_width = target_w - new_width
        pad_height = target_h - new_height
        
        # Pad to target size (pad on bottom and right)
        padded = cv2.copyMakeBorder(
            resized, 0, pad_height, 0, pad_width, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert BGR to RGB and normalize
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        preprocessed = np.transpose(normalized, (2, 0, 1))
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        return np.ascontiguousarray(preprocessed), scale, (pad_width, pad_height), original_shape
    
    def postprocess_yolo_seg(self, predictions, proto_masks, scale_factor, padding, original_shape):
        """Postprocess YOLO segmentation outputs"""
        # Remove batch dimension
        predictions = predictions[0]  # [49, 8400]
        proto_masks = proto_masks[0]  # [32, 160, 160]
        
        # Transpose predictions for easier processing
        predictions = predictions.T  # [8400, 49]
        
        # Extract components
        boxes = predictions[:, :4]  # center_x, center_y, width, height
        class_scores = predictions[:, 4:4+self.num_classes]  # class probabilities
        mask_coefficients = predictions[:, 4+self.num_classes:]  # mask coefficients
        
        # Get maximum class scores and indices
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence threshold
        valid_indices = max_scores >= self.conf_threshold
        
        if not np.any(valid_indices):
            return np.zeros((*original_shape, 3), dtype=np.uint8)
        
        # Filter predictions
        filtered_boxes = boxes[valid_indices]
        filtered_scores = max_scores[valid_indices]
        filtered_class_ids = class_ids[valid_indices]
        filtered_mask_coeffs = mask_coefficients[valid_indices]
        
        # Convert center format to corner format
        x_center, y_center, width, height = filtered_boxes.T
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        corner_boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Apply Non-Maximum Suppression
        indices = self.nms(corner_boxes, filtered_scores, self.iou_threshold)
        
        if len(indices) == 0:
            return np.zeros((*original_shape, 3), dtype=np.uint8)
        
        # Final filtered results
        final_boxes = corner_boxes[indices]
        final_scores = filtered_scores[indices]
        final_class_ids = filtered_class_ids[indices]
        final_mask_coeffs = filtered_mask_coeffs[indices]
        
        # Scale boxes back to original image coordinates
        final_boxes = self.scale_boxes(final_boxes, scale_factor, padding, original_shape)
        
        # Generate masks
        final_masks = self.generate_masks(final_mask_coeffs, proto_masks, final_boxes, original_shape)
        
        # Create color mask
        color_mask = np.zeros((*original_shape, 3), dtype=np.uint8)
        
        for i, (class_id, mask) in enumerate(zip(final_class_ids, final_masks)):
            if class_id < len(SEGMENTATION_COLORMAP):
                color = SEGMENTATION_COLORMAP[class_id]
                color_mask[mask > 0] = color
        
        return color_mask
    
    def scale_boxes(self, boxes, scale_factor, padding, original_shape):
        """Scale boxes back to original image coordinates"""
        pad_width, pad_height = padding
        
        # Scale back to original size
        boxes /= scale_factor
        
        # Clip to image boundaries
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])
        
        return boxes
    
    def generate_masks(self, mask_coeffs, proto_masks, boxes, original_shape):
        """Generate final masks from coefficients and prototypes"""
        masks = []
        
        for i in range(len(mask_coeffs)):
            # Linear combination of prototype masks
            mask = np.dot(mask_coeffs[i], proto_masks.reshape(32, -1))
            mask = mask.reshape(160, 160)
            
            # Apply sigmoid activation
            mask = 1 / (1 + np.exp(-mask))
            
            # Resize mask to original image size
            mask_resized = cv2.resize(mask, 
                                    (original_shape[1], original_shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
            
            # Crop mask to bounding box region
            x1, y1, x2, y2 = boxes[i].astype(int)
            mask_cropped = np.zeros_like(mask_resized)
            if x2 > x1 and y2 > y1:  # Valid box
                mask_cropped[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]
            
            # Apply threshold
            binary_mask = (mask_cropped > self.mask_threshold).astype(np.uint8)
            masks.append(binary_mask)
        
        return masks
    
    @staticmethod
    def nms(boxes, scores, iou_threshold):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            order = order[1:][iou <= iou_threshold]
        
        return keep
    
    def infer(self, frame, labels_to_show=None):
        """Run inference on a single frame and return segmentation mask"""
        # Preprocess
        input_data, scale, padding, original_shape = self.preprocess_frame(frame)
        
        # Copy input to GPU
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        
        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy outputs from GPU
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output.host, output.device, self.stream)
        self.stream.synchronize()
        
        # Reshape outputs based on expected shapes
        output_shapes = list(self.output_shapes.values())
        predictions = self.outputs[0].host.reshape(output_shapes[0])  # [1, 49, 8400]
        proto_masks = self.outputs[1].host.reshape(output_shapes[1])  # [1, 32, 160, 160]
        
        # Postprocess
        mask = self.postprocess_yolo_seg(predictions, proto_masks, scale, padding, original_shape)
        
        return mask

def main():
    # Initialize TensorRT inference  
    engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/yolov8n640x640.engine"
    video_path = "/home/deen/ros2_ws/src/automama/automama/perception/killo_road.mp4"
    
    print(f"TensorRT version: {trt.__version__}")
    print(f"Loading engine from: {engine_path}")
    
    try:
        # Initialize inference engine
        trt_inference = YOLOTensorRTInference(engine_path)
        print("YOLO TensorRT engine loaded successfully")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Failed to open video!")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {frame_count} frames at {fps:.2f} FPS")
        
        frame_idx = 0
        total_time = 0
        
        print("Starting inference... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame")
                break
            
            start_time = time.time()
            
            # Run inference
            mask = trt_inference.infer(frame)
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Create overlay
            blended = cv2.addWeighted(frame, 0., mask, 0.3, 0)
            
            # Add performance info
            fps_text = f"FPS: {1.0/inference_time:.1f} | Frame: {frame_idx+1}/{frame_count}"
            cv2.putText(blended, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display results
            cv2.imshow("YOLO Segmentation", blended)
            
            frame_idx += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        if 'frame_idx' in locals() and frame_idx > 0:
            avg_fps = frame_idx / total_time
            print(f"\nPerformance Summary:")
            print(f"Processed {frame_idx} frames")
            print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main()