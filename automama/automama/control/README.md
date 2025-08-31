# AGV Control Interface (Keyboard + ESP32)

This script provides a **manual and autonomous control interface** for the AutoMAMA AGV, communicating with the ESP32 microcontroller over serial. It allows controlling throttle, steering, and brake while also supporting autonomous commands from the navigation stack.

---

## Features

- **Manual Control via Keyboard**
  - W/S → Increase / Decrease Throttle
  - A/D → Steer Left / Right
  - B/V → Adjust Brake Level
  - E → Center steering
  - Space → Emergency Stop
  - O → Toggle between Manual and Autonomous mode
  - Q / ESC → Exit the interface

- **Autonomous Control Integration**
  - Subscribes to `/actuator_cmds` topic (`Float32MultiArray`)
  - Expects `[throttle, steering, brake]` values
  - Overrides manual input when Auto mode is active

- **Serial Communication**
  - Protocol with start (`0xAA`) and end (`0x55`) bytes
  - Steering values are encoded with an offset for safe transmission
  - Sends packets: `[START_BYTE, Throttle, Encoded Steering, Brake, END_BYTE]`
  - Receives packets from ESP32 for telemetry and verification

- **Live Console Display**
  - Shows current mode, throttle, steering, brake, and last received ESP32 data
  - Updates in real-time every 0.1 seconds

---

## Getting Started

### Requirements
- NVIDIA Jetson / Linux device
- ROS2 (Foxy/Humble) installed
- Python dependencies: `rclpy`, `pynput`, `serial`
- ESP32 firmware flashed for AutoMAMA communication

### Running the Interface
1. Start ROS2
2. Run the manual control node:
   ```bash
   ros2 run automama manual_control
   ```
### While Running

- Press keys to manually control the AGV
- Press `O` to toggle to autonomous commands (received from navstack)
- Press `Q` or `ESC` to exit

---

### Control Modes

| Mode   | Description |
|--------|-------------|
| MANUAL | Keyboard directly controls throttle, steering, and brake |
| AUTO   | Commands received from `/actuator_cmds` ROS2 topic override manual input |

---
### Serial Transmission Protocol

**Packet Structure (5 bytes):**

| Byte | Value                                |
|------|--------------------------------------|
| 0    | `START_BYTE` (0xAA)                  |
| 1    | Throttle (0-150)                     |
| 2    | Encoded Steering (-25 to +25 + offset) |
| 3    | Brake Level (0-40)                   |
| 4    | `END_BYTE` (0x55)                    |

---

### Notes

- `STEERING_OFFSET = 30` ensures all steering values are positive during transmission.
- The node decodes received steering bytes to verify actual ESP32 feedback.

### Key Points

- Ensure the serial port (`/dev/ttyTHS1`) and baud rate (`115200`) match your ESP32 setup.
- Always test manual control first to verify hardware communication before enabling autonomous mode.
- Emergency stop (`SPACE`) sets throttle and steering to zero and max brake level.

