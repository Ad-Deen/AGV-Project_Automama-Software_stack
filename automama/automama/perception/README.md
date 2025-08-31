# Perception, Stereo Depth, and Navigation Stack

This package handles stereo vision, object segmentation, depth estimation, and local path planning for the AGV. It runs fully on the GPU using NVIDIA CUDA, VPI, and CuPy for real-time performance.

---

## Features

1. **Stereo Camera Rectification & Calibration**
   - Uses left and right Jetson cameras.
   - Warp maps are loaded from the calibration directory.
   - Rectification ensures accurate disparity computation.

2. **Object Segmentation**
   - YOLOv8n model (TensorRT engine) runs inference on the left frame.
   - Segments the road, dynamic objects, and other semantic classes.
   - Segmentation masks are transferred to GPU for further processing.

3. **Depth Estimation**
   - **Custom CUDA Kernel Stereo Matching**
     - Implements a SGBM-like disparity calculation directly on the GPU.
     - Temporal filtering ensures stable disparity maps.
     - Outputs disparity and filtered disparity images.
   - **Depth Map Generation**
     - Converts disparity to depth using camera intrinsics and baseline.
     - Applies masking and valid-depth filtering to remove outliers.

4. **Occupancy Grid Mapping**
   - Processes the depth map and segmentation masks.
   - Generates:
     - Fixed bird’s-eye view occupancy grid
     - Dynamic costmaps for navigation
   - GPU-accelerated via CuPy kernels for real-time performance.

5. **Path Planning (NavStack)**
   - Receives occupancy/costmap data.
   - Computes local steering and throttle commands.
   - Publishes actuator commands to `/actuator_cmds` ROS2 topic.

6. **Point Cloud Generation**
   - Converts disparity map into 3D points using Q matrix.
   - Optional voxel downsampling for visualization.
   - Visualized using Open3D.

---

## Usage

1. **Start the Perception Node**
```bash
ros2 run automama navstack
```

---

## Main Loop

- Captures left and right frames from stereo cameras.
- Runs YOLOv8n segmentation on the left frame.
- Generates masks for road and dynamic objects.
- Computes disparity map (using custom CUDA kernel or VPI).
- Filters depth map and generates occupancy grid.
- Computes local steering and throttle using NavStack.
- Publishes `[throttle, steering, brake]` to `/actuator_cmds`.

---

## Visualization

- Segmentation masks, depth map, costmaps, and bird’s-eye occupancy maps can be displayed using OpenCV.
- 3D point cloud can be visualized using Open3D.

---

## Notes

- Uses `CuPy` arrays for GPU-accelerated computations.
- Supports temporal disparity filtering for smooth depth maps.
- Voxel downsampling can reduce point cloud size for visualization.
- All computation (segmentation, depth estimation, occupancy map) is performed on GPU for real-time performance.

---

## Dependencies

- ROS2 Foxy / Humble
- NVIDIA Jetson drivers + CUDA
- OpenCV, CuPy, PyCUDA
- VPI (Vision Programming Interface)
- Open3D (optional for point cloud visualization)
- TensorRT (for YOLOv8n inference)
