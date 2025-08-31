import os
import urllib.request

# URLs
model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11s-seg.pt"
yaml_url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco-seg.yaml"

# Filenames
model_filename = "yolo11s-seg.pt"
yaml_filename = "coco-seg.yaml"

# Download model
print(f"Downloading {model_filename}...")
urllib.request.urlretrieve(model_url, model_filename)
print("✅ Model downloaded.")

# Download YAML
print(f"Downloading {yaml_filename}...")
urllib.request.urlretrieve(yaml_url, yaml_filename)
print("✅ YAML downloaded.")

# Show location
print(f"\n📁 Saved files in: {os.getcwd()}")
