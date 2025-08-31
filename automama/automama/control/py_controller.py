from pynput import keyboard
import serial
import time
import os

# ---------------------- Serial Configuration ----------------------
SERIAL_PORT = '/dev/ttyTHS1'       # Adjust to your correct port
BAUD_RATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
time.sleep(2)

# ---------------------- Protocol Definitions ----------------------
START_BYTE = 0xAA
END_BYTE = 0x55
STEERING_OFFSET = 30

# ---------------------- Control Variables -------------------------
throttle = 0
steering = 0
last_response = None

brake_levels = [40, 20, 10, 0]
brake_level_index = 0
brake = brake_levels[brake_level_index]

pressed_keys = set()
exit_flag = False

# ---------------------- Utility Functions -------------------------
def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def encode_steering(value):
    return int(value + STEERING_OFFSET)

def decode_steering(encoded_byte):
    return int(encoded_byte - STEERING_OFFSET)

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# ---------------------- Key Event Handlers -------------------------
def on_press(key):
    global throttle, steering, brake, brake_level_index, exit_flag

    try:
        k = key.char.lower()
    except AttributeError:
        k = key.name

    if k in ['w', 's', 'a', 'd', 'b', 'v', 'o', 'space', 'q', 'esc']:
        pressed_keys.add(k)

        if k == 'w':
            throttle = clamp(throttle + 5, 0, 150)
        elif k == 's':
            throttle = clamp(throttle - 5, 0, 150)
        elif k == 'd':
            steering = clamp(steering + 1, -25, 25)
        elif k == 'a':
            steering = clamp(steering - 1, -25, 25)
        elif k == 'b':
            brake_level_index = clamp(brake_level_index + 1, 0, len(brake_levels) - 1)
            brake = brake_levels[brake_level_index]
        elif k == 'v':
            brake_level_index = clamp(brake_level_index - 1, 0, len(brake_levels) - 1)
            brake = brake_levels[brake_level_index]
        elif k == 'o':
            steering = 0
        elif k == 'space':
            # throttle = 0
            steering = 5
            # brake_level_index = len(brake_levels) - 1
            brake = brake_levels[-1]
            print("\n🚨 EMERGENCY STOP! 🚨")
            time.sleep(0.5)
        elif k in ['q', 'esc']:
            exit_flag = True
            throttle = 0
            steering = 0

def on_release(key):
    try:
        k = key.char.lower()
    except AttributeError:
        k = key.name
    if k in pressed_keys:
        pressed_keys.remove(k)

# ---------------------- UI Header -------------------------
print(f"🔌 Connected to {SERIAL_PORT}")
print("🎮 Use keyboard to control ESP32")
print("Press Q or ESC to exit.\n")

# ---------------------- Main Control Loop -------------------------
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    while not exit_flag:
        # Send data if keys are actively being pressed
        if pressed_keys:
            steering_encoded = encode_steering(steering)
            packet = bytes([START_BYTE, throttle, steering_encoded, brake, END_BYTE])
            ser.write(packet)
            time.sleep(0.05)

        # Process incoming serial response
        while ser.in_waiting >= 5:
            if ser.read(1) == bytes([START_BYTE]):
                response_data = ser.read(4)
                if response_data[3] == END_BYTE:
                    last_response = list(response_data[0:3])

        received_steering = decode_steering(last_response[1]) if last_response else "None"

        clear_console()
        print(f"🎮 Live AutoMAMA Control ({SERIAL_PORT}) — Press Q/ESC to exit")
        print("--------------------------------------------------")
        print(f"Throttle : {throttle:3} | Steering : {steering:+3} | Brake : {brake:3} (Level {brake_level_index})")
        print(f"Sent     : [Throttle: {throttle}, Steering: {steering:+3}, Brake: {brake}]")

        if last_response:
            print(f"Received : [Throttle: {last_response[0]}, Decoded Steering: {received_steering:+3}, Brake: {last_response[2]}]")
            print(f"           -> Raw Steering Byte from ESP32: {last_response[1]}")
        else:
            print("Received : None")

        print("--------------------------------------------------")
        print("Keys: [W/S]=Throttle  [A/D]=Steering  [B/V]=Brake Level")
        print("      [O]=Center      [SPACE]=E-Stop   [Q/ESC]=Exit")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n❌ Interrupted")

finally:
    listener.stop()
    ser.close()
    print(f"✅ Serial {SERIAL_PORT} closed.")
