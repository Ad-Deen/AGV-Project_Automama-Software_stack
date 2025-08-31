from ultralytics import YOLO
import yaml

# Load class names from your data.yaml
# with open('data.yaml', 'r') as f:
#     data = yaml.safe_load(f)
# class_names = data['names']

# Step 1: Export the YOLOv8s PyTorch model to TensorRT format
# You only need to do this once. Comment it out after first run.
# model = YOLO("yolo11s-seg.pt")
# model.export(format="engine")  # Creates yolov8s.engine

# Step 2: Load the TensorRT-optimized segmentation model
tensorrt_model = YOLO("yolo11s-seg.engine", task="segment")

# Step 3: Assign class names to the model (to avoid KeyError)
# tensorrt_model.names = {i: name for i, name in enumerate(class_names)}

# Step 4: Run inference on the video with stream=True
results = tensorrt_model.predict(source="killo_road.mp4", stream=True)

# Step 5: Iterate over frames and print masks
# Iterate results to force inference execution and print output
for r in results:
    print(f"Frame: {r.path}")            # path to the current frame
    # print(f"Detected classes: {r.names}")  # class name map
    # print(f"Boxes: {r.boxes}")           # bounding boxes
    # print(f"Masks: {r.masks.data}")           # segmentation masks
#     break
