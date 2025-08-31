import serial
import time

# Open the serial port
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Update port if needed
time.sleep(2)  # Wait for Arduino reset

throttle = 100
steering = 30
brake = 50

try:
    while True:
        # Send 3-byte command
        msg = bytes([throttle, steering, brake])
        ser.write(msg)

        # Read response (3 bytes)
        response = ser.read(3)
        if len(response) == 3:
            rx = list(response)
            print(f"Sent: [{throttle}, {steering}, {brake}] | Received: {rx}")
        else:
            print("Incomplete response")

        # Change values for testing
        throttle = (throttle + 1) % 256
        steering = (steering - 1) % 256

        time.sleep(0.05)  # 10 Hz

except KeyboardInterrupt:
    print("Stopped")
finally:
    ser.close()
