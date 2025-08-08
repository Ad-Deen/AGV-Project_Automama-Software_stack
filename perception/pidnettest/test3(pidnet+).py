import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import cv2
import time
import os
from collections import namedtuple

cuda.init()
device = cuda.Device(0)
# cuda.Device(0).make_context() 
context = device.make_context()  # Push context on stack
context.pop()
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

def get_colormap_for_labels(labels, full_colormap):
    colormap = {}
    for label in labels:
        if 0 <= label < len(full_colormap):
            colormap[label] = full_colormap[label]
    return colormap

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
        # global context
        # context.push()   
        self.engine_path = engine_path
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine()
        # context = device.make_context()  # Push context on stack
        
        self.context = self.engine.create_execution_context()
        
        self.inputs, self.outputs, self.bindings = self.allocate_buffers()
        
        self.input_tensor_name = None
        self.output_tensor_name = None
        self.input_shape = None
        self.output_shape = None

        self.get_tensor_info()
        self.set_tensor_addresses_once()  # <- Set once here
        # context.pop()
    
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
        # stream = cuda.Stream()
        
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
        
        return inputs, outputs, bindings
    
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
    
    def postprocess(self, output_data, original_frame_shape, labels_to_show=None):
        """
        Postprocess the segmentation output and return a color-mapped image
        showing only specified label(s). If labels_to_show is None, show all.
        """

        # Reshape the model output to (H, W)
        output_reshaped = output_data.reshape(self.output_shape)  # (1, 1, H, W)
        pred = output_reshaped[0, 0]  # (H, W)

        # Use all labels if not specified
        if labels_to_show is None:
            labels_to_show = list(range(len(CITYSCAPES_COLORMAP)))

        # Prepare selective colormap
        selective_colormap = get_colormap_for_labels(labels_to_show, CITYSCAPES_COLORMAP)

        # Apply color to selected labels, others are black
        color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for label in labels_to_show:
            color_mask[pred == label] = selective_colormap[label]

        # Resize to match original frame shape
        resized_mask = cv2.resize(color_mask, (original_frame_shape[1], original_frame_shape[0]), interpolation=cv2.INTER_NEAREST)

        return resized_mask
    
    def infer(self, frame):
        self.stream = cuda.Stream()
        # global context
        """Run inference on a single frame"""
        # Preprocess
        input_data = self.preprocess_frame(frame)
        # context.push()
        # Copy input data to host buffer
        np.copyto(self.inputs[0].host, input_data.ravel())
        
        # Transfer data from host to device
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)
        
        # Run inference using new API
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Transfer results from device to host
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        
        # Synchronize the stream
        self.stream.synchronize()
        # context.pop()
        # Postprocess
        mask = self.postprocess(self.outputs[0].host, frame.shape)
        
        return mask

# ---------------- Main Video Processing ----------------
def main():
    # Initialize TensorRT inference
    engine_path = "/home/deen/ros2_ws/src/automama/automama/perception/PID net models/engine/pidnet_L_480x640_32f.engine"
    video_path = "/home/deen/ros2_ws/src/automama/automama/perception/killo_road.mp4"
    
    print(f"TensorRT version: {trt.__version__}")
    print(f"Loading engine from: {engine_path}")
    
    try:
        global context
        context.push() 
        # Initialize inference engine
        trt_inference = TensorRTInference(engine_path)
        print("TensorRT engine loaded successfully")
        context.pop() 
        
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
            context.push() 
            mask = trt_inference.infer(frame)
            context.pop() 
            # print(np.array2string(mask))
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Create overlay
            blended = cv2.addWeighted(frame, 0.2, mask, 0.8, 0)
            
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
        # context.detach()
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