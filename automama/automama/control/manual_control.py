#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pynput import keyboard
import serial
import time
import os
from std_msgs.msg import Float32MultiArray
# ---------------------- Protocol Definitions ----------------------
START_BYTE = 0xAA
END_BYTE = 0x55
STEERING_OFFSET = 30

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def encode_steering(value):
    return int(value + STEERING_OFFSET)

def decode_steering(encoded_byte):
    return int(encoded_byte - STEERING_OFFSET)

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


class ESP32KeyboardController(Node):
    def __init__(self):
        super().__init__('esp32_keyboard_controller')

        # Serial configuration
        self.serial_port = '/dev/ttyTHS1'
        self.baud_rate = 115200
        self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
        time.sleep(2)

        self.get_logger().info(f"🔌 Connected to {self.serial_port}")
        self.get_logger().info("🎮 Use keyboard to control ESP32 — Press Q or ESC to exit.")
        self.auto_sub = self.create_subscription(
            Float32MultiArray,
            '/actuator_cmds',
            self.auto_cmd_callback,
            10
        )
        # Store the latest auto commands
        self.auto_throttle = 0
        self.auto_steering = 0
        self.auto_brake = 0
        # Control variables
        self.throttle = 0
        self.steering = 0
        self.brake_levels = [40, 20, 10, 0]
        self.brake_level_index = 0
        self.brake = self.brake_levels[self.brake_level_index]
        self.last_response = None
        self.pressed_keys = set()
        self.exit_flag = False
        self.auto_mode = False   # Starts in manual mode
        self.mode = "MANUAL"
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        # ROS timer for main loop
        self.timer = self.create_timer(0.1, self.main_loop)

    def auto_cmd_callback(self, msg):
    # Expecting msg.data = [throttle, steering, brake]
        if len(msg.data) >= 3:
            self.auto_throttle = 70 + int(round(float(msg.data[0])*35))

            self.auto_steering = int(round(float(msg.data[1]) * 25))
            
            # self.auto_brake = int(round(float(msg.data[2])))
            print(f"steering: {self.auto_steering}, throttle: {self.auto_throttle}, brake:{self.brake}")
    # ---------------------- Keyboard Events ----------------------
    def on_press(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = key.name

        if k in ['w', 's', 'a', 'd','e', 'b', 'v', 'o', 'space', 'q', 'esc']:
            self.pressed_keys.add(k)

            if k == 'w':
                self.throttle = clamp(self.throttle + 5, 0, 150)
            elif k == 's':
                self.throttle = clamp(self.throttle - 5, 0, 150)
            elif k == 'd':
                self.steering = clamp(self.steering + 1, -25, 25)
            elif k == 'a':
                self.steering = clamp(self.steering - 1, -25, 25)
            elif k == 'b':
                self.brake_level_index = clamp(self.brake_level_index + 1, 0, len(self.brake_levels) - 1)
                self.brake = self.brake_levels[self.brake_level_index]
            elif k == 'v':
                self.brake_level_index = clamp(self.brake_level_index - 1, 0, len(self.brake_levels) - 1)
                self.brake = self.brake_levels[self.brake_level_index]
            elif k == 'e':
                self.steering = 0
            elif k == 'space':
                self.throttle = 0
                self.steering = 0
                self.brake = self.brake_levels[-1]
                self.get_logger().warn("🚨 EMERGENCY STOP! 🚨")
                time.sleep(0.5)
            elif k in ['q', 'esc']:
                self.exit_flag = True
                self.throttle = 0
                self.steering = 0
            elif k == 'o':  # toggle auto/manual
                self.auto_mode = not self.auto_mode
                self.mode = "AUTO" if self.auto_mode else "MANUAL"
                self.get_logger().info(f"Mode switched to {self.mode}")

    def on_release(self, key):
        try:
            k = key.char.lower()
        except AttributeError:
            k = key.name
        if k in self.pressed_keys:
            self.pressed_keys.remove(k)

    # ---------------------- Main Loop ----------------------
    def main_loop(self):
        if self.exit_flag:
            rclpy.shutdown()
            return

        if self.auto_mode:
            # ------------------- AUTO MODE -------------------
            # Here, replace these example values with your navstack / autopilot outputs
            throttle = self.auto_throttle       # Example auto throttle
            steering = self.auto_steering         # Example auto steering
            if self.auto_throttle < 55:          # Example auto brake
                throttle = 0
                brake = self.brake_levels[-1]
            else: 
                brake = self.brake_levels[0]
        else:   
            # ------------------- MANUAL MODE ----------------- 
            steering = self.steering
            brake = self.brake
            throttle = self.throttle

        # Encode & send to ESP32
        steering_encoded = encode_steering(steering)
        packet = bytes([START_BYTE, throttle, steering_encoded, brake, END_BYTE])
        self.ser.write(packet)
        time.sleep(0.05)

        # Process incoming serial data
        while self.ser.in_waiting >= 5:
            if self.ser.read(1) == bytes([START_BYTE]):
                response_data = self.ser.read(4)
                if response_data[3] == END_BYTE:
                    self.last_response = list(response_data[0:3])

        received_steering = decode_steering(self.last_response[1]) if self.last_response else "None"

        # Console output
        clear_console()
        print(f"🎮 Live AutoMAMA Control ({self.serial_port}) — Press Q/ESC to exit")
        print(f"========== Mode switched to {self.mode} !! ==============")
        print("Press 'O' to change between modes")
        print("--------------------------------------------------")
        print(f"Throttle : {self.throttle:3} | Steering : {steering:+3} | Brake : {self.brake:3} (Level {self.brake_level_index})")
        print(f"Sent     : [Throttle: {throttle}, Steering: {steering:+3}, Brake: {brake}]")

        if self.last_response:
            print(f"Received : [Throttle: {self.last_response[0]}, Decoded Steering: {received_steering:+3}, Brake: {self.last_response[2]}]")
            print(f"           -> Raw Steering Byte from ESP32: {self.last_response[1]}")
        else:
            print("Received : None")

        print("--------------------------------------------------")
        print("Keys: [W/S]=Throttle  [A/D]=Steering  [B/V]=Brake Level")
        print("      [E]=Center      [SPACE]=E-Stop   [Q/ESC]=Exit")

    def destroy_node(self):
        self.listener.stop()
        self.ser.close()
        self.get_logger().info(f"✅ Serial {self.serial_port} closed.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ESP32KeyboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
