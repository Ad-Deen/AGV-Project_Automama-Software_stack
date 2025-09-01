# 🚙 Automama – AGV Software Stack

This repository contains the software stack for **Project Automama (AGV)**.  
It integrates custom perception, stereo vision, and autonomous navigation pipelines with low-level actuation through an **ESP32 microcontroller**.

---

## 🧩 System Overview

- **Camera Perception** – Stereo image capture, depth estimation, and segmentation for obstacle detection.
- **Stereo Vision Pipeline** – Custom disparity → depth → 3D point cloud reconstruction, accelerated with GPU and visualized in real time with VisPy.
- **Navigation Stack (ROS2)** – Localization, costmaps, and path planning modules for autonomous driving.
- **Control & Actuation** – ROS2 communicates with **ESP32** to control:
  - Steering  
  - Throttle  
  - Braking  
- **Manual Override** – Python-Arduino bridge for teleoperation and debugging.

<p align="left">
  <img src="gifs/Autonomous_run.gif" alt="AGV 1" width="500" height="480"/>
  <img src="gifs/automama_run.gif" alt="AGV 2" width="150" height="480"/>
</p>

---

## 📂 Project Automama – Directory Structure

```text
automama
├── arduino codes/                # Arduino control scripts
│   ├── Automama_control_test_DualCore/
│   │   └── *.ino                 # Thruster, steering, brake, comms control
│   └── py_control.py             # Python-side Arduino control interface
│
├── automama/
│   ├── callab_data/              # Camera calibration files (stereo + intrinsic)
│   ├── control/                  # Manual + autonomous control scripts
│   ├── interface/                # Data logging and interface scripts
│   ├── navigation/               # Costmaps, GPS tests, VO modules
│   ├── perception/               # Stereo vision, depth estimation, segmentation
│   └── test/                     # Unit tests for communication & control
│
├── daddyutils/                   # Utility modules (camera handling, GPU, SLAM utils)
├── launch/                       # ROS2 launch files for stereo pipeline & nodes
├── resource/                     # ROS2 resource files
├── test/                         # Code quality & style tests
│
├── output_video.mp4              # Sample output (stereo pipeline demo)
├── stereo.rviz                   # RViz config for visualization
├── sust_campus_map.html          # Campus map visualization
│
├── setup.py                      # Package setup
├── setup.cfg
├── pyproject.toml
└── package.xml                   # ROS2 package manifest
```

---
## ⚙️ Technology Stack

- **ROS2 (Foxy/Humble)** – Middleware for modular control and communication  
- **Python + OpenCV** – Image processing & stereo disparity  
- **VisPy (GPU)** – Real-time 3D point cloud visualization  
- **ESP32 + Arduino** – Low-level actuation (steering, brake, throttle)  
- **RViz2** – Simulation & visualization  

---

## 🚀 Features

- Custom stereo vision → depth → 3D point cloud pipeline  
- Autonomous navigation with ROS2 planners & costmaps  
- Real-time actuation via ESP32 bridge  
- Simulation + real-world testing support  
- Modular design for future extensions  

---
## 🧠 Perception & Navigation Pipeline

In this module, we designed a **GPU-accelerated stereo vision + semantic perception stack** optimized for real-time operation on the Jetson Orin Nano.

### Workflow

1. **Stereo Camera Input** → Capture and rectify left/right frames.  
2. **Semantic Segmentation** → Apply YOLOv8n-Seg on the left frame to extract class-wise segmentation masks.  
3. **Depth Estimation**  
   - **Road Mask** → Use static camera projection for depth estimation.  
   - **Dynamic Objects** → Apply custom stereo disparity + 3D reconstruction for object depth.  
4. **Occupancy Grid Mapping** → Fuse road and object depth into a unified grid map.  
5. **Path Planning** → Run **Gap Follow Algorithm** for real-time dynamic path planning.  

### Optimization

- Entire pipeline runs on **GPU** for real-time performance.  
- Integrated with **NVIDIA VPI**, **CUDA**, and **CuPy** to handle:  
  - GPU memory context management  
  - Custom CUDA kernel insertions  
  - Concurrent processing of segmentation + stereo vision  
- Achieved **real-time inference + planning** on **Jetson Orin Nano**.


```mermaid
graph TD
    A[Stereo Camera Input] --> B[Rectification]
    B --> C[Left Frame: YOLOv8n-Seg]
    C --> D[Segmentation Masks]
    D --> E{Mask Type?}
    E -->|Road| F[Static Camera Projection → Depth]
    E -->|Dynamic Objects| G[Stereo Vision Pipeline → Object Depth]
    F --> H[Occupancy Grid Mapping]
    G --> H
    H --> I[Gap Follow Algorithm]
    I --> J[Real-Time Path Planning]
    style A fill:#d9f0ff,stroke:#333,stroke-width:1px
    style B fill:#ffe6cc,stroke:#333,stroke-width:1px
    style C fill:#e6ccff,stroke:#333,stroke-width:1px
    style D fill:#fff0b3,stroke:#333,stroke-width:1px
    style F fill:#ccffcc,stroke:#333,stroke-width:1px
    style G fill:#ffcccc,stroke:#333,stroke-width:1px
    style H fill:#ffd9e6,stroke:#333,stroke-width:1px
    style I fill:#cce6ff,stroke:#333,stroke-width:1px
    style J fill:#cce2ff,stroke:#333,stroke-width:1px
```
---
### 🛣️ Gap Follow Algorithm (AGV Path Planning)

After generating the occupancy costmap from perception data, the road pixels often had discontinuities due to sensor noise or sparse depth data:

- **Near-road pixels** were smooth and reliable  
- **Distant-road pixels** exhibited discontinuous scan lines  

To handle this, we implemented a **GPU-based interpolation pipeline** using **CuPy custom kernels in C++**, fully leveraging the Jetson Orin Nano GPU for real-time performance.

#### Workflow:

1. **Costmap Interpolation**  
   - Fill gaps in distant road pixels  
   - GPU kernels perform fast 2D interpolation

2. **Boundary Classification**  
   - Free space → `0`  
   - Obstacles → `255`  
   - Intermediate border → `245`  

3. **Euclidean Distance Transform (EDT)**  
   - Compute distances from borders using GPU-accelerated custom kernels

4. **Low-Cost Path Convergence**  
   - Pixels converge along the path of least cost  
   - Vehicle constraints (turning radius, width) applied  
   - Occupancy grid ensures collision-free navigation

5. **Real-Time Control Output**  
   - Gap-following trajectory sent to ROS2 navigation stack  
   - ESP32 receives commands for steering, throttle, and braking

#### Workflow Diagram

```mermaid
graph TD
    A[Raw Costmap] --> B[GPU Interpolation CuPy C++ Kernel]
    B --> C[Boundary Classification]
    C --> D[Euclidean Distance Transform]
    D --> E[Low-Cost Path Convergence]
    E --> F[Apply Vehicle Constraints & Occupancy Grid]
    F --> G[Gap Follow Trajectory Output to ROS2 & ESP32]

    style A fill:#d9f0ff,stroke:#333,stroke-width:1px
    style B fill:#ffe6cc,stroke:#333,stroke-width:1px
    style C fill:#e6ccff,stroke:#333,stroke-width:1px
    style D fill:#fff0b3,stroke:#333,stroke-width:1px
    style E fill:#ccffcc,stroke:#333,stroke-width:1px
    style F fill:#ffcccc,stroke:#333,stroke-width:1px
    style G fill:#cce6ff,stroke:#333,stroke-width:1px
```
---
### 📦 How to Use the Package

The AGV software stack is modular and provides multiple interfaces for control and perception. Follow these steps to get started:

---

#### 1. Manual Control
- Start the manual control interface using ROS2:

```bash
ros2 run automama manual_control
```
- This allows you to operate the AGV via keyboard inputs.
- For detailed instructions on installation, key bindings, and script usage, refer to the **[Control Stack README](automama/automama/control/README.md)**.
- Topic reference: Manual Control
#### 2. Perception & Autonomous Navigation
- Start the perception stack along with local map generation, path planning, and actuation commands:

```bash
ros2 run automama navstack
```
The system performs:

- Stereo camera rectification & calibration
- Object segmentation and depth estimation
- Occupancy grid mapping
- Gap Follow path planning
- Steering, throttle, and brake actuation via ESP32

Within the manual steering interface, press **`o`** to toggle between:

- **Manual override** – user controls the AGV
- **Autonomous mode** – AGV follows the planned path automatically
- For detailed instructions on stereo setup, calibration, and perception, refer to the **[Perception Stack README](automama/automama/perception/README.md)**.
---
💡 **Important Notes**

- Ensure your NVIDIA Jetson device has GPU drivers installed.
- Make sure all Python/C++ dependencies (CuPy, OpenCV, VisPy, ROS2 packages) are installed and correctly configured.
- Test the manual control first to verify communication with the ESP32 before running autonomous navigation.
