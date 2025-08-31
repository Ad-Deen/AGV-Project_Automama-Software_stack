import serial

PORT = '/dev/ttyCH341USB0'  # or ttyUSB0 if using mainline driver
BAUD = 115200  # default for WTGPS-300

try:
    ser = serial.Serial(PORT, BAUD, timeout=2)
    print(f"Reading from {PORT} at {BAUD} baud...")
    
    while True:
        line = ser.readline().decode('ascii', errors='replace').strip()
        if line:
            print(line)
            print("==============================")

except Exception as e:
    print("Error:", e)
