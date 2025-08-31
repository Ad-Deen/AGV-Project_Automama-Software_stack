import serial
import pynmea2

# Change this to your actual serial port (e.g., /dev/ttyUSB0 for Linux or COMx for Windows)
SERIAL_PORT = '/dev/ttyCH341USB0'
BAUD_RATE = 115200  # WTGPS300 default baud rate

def parse_gps():
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print("Listening to WTGPS300...")
            while True:
                line = ser.readline().decode('ascii', errors='replace').strip()
                
                if line.startswith('$'):
                    try:
                        msg = pynmea2.parse(line)
                        
                        if isinstance(msg, pynmea2.GGA):
                            print(f"[GGA] Time: {msg.timestamp}, Lat: {msg.latitude} {msg.lat_dir}, "
                                  f"Lon: {msg.longitude} {msg.lon_dir}, Alt: {msg.altitude}m")
                        
                        elif isinstance(msg, pynmea2.RMC):
                            print(f"[RMC] Time: {msg.timestamp}, Lat: {msg.latitude}, Lon: {msg.longitude}, "
                                  f"Speed: {msg.spd_over_grnd} knots, Date: {msg.datestamp}")
                        
                        # Add more message types if needed
                        
                    except pynmea2.nmea.ParseError as e:
                        print(f"NMEA Parse Error: {e}")

    except serial.SerialException as e:
        print(f"Serial Error: {e}")

if __name__ == "__main__":
    parse_gps()
