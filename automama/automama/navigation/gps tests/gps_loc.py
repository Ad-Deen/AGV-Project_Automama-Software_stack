import serial
import pynmea2
import pyproj
import time
import os

# --- Configuration ---
# Change this to your actual serial port
SERIAL_PORT = '/dev/ttyCH341USB0'
BAUD_RATE = 115200

# Global variables to store the initial state for the transformation
initial_lat, initial_lon, initial_alt = None, None, None
origin_x, origin_y = None, None
transformer = None

def get_real_time_position():
    """
    Reads GPS data from a serial port and prints real-time position updates in meters.
    """
    global initial_lat, initial_lon, initial_alt, origin_x, origin_y, transformer

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Listening to GPS on {SERIAL_PORT}...")
            
            ser.flushInput()

            while True:
                line = ser.readline().decode('ascii', errors='replace').strip()
                
                if line.startswith('$'):
                    try:
                        msg = pynmea2.parse(line)
                        
                        if isinstance(msg, pynmea2.GGA):
                            current_lat = msg.latitude
                            current_lon = msg.longitude
                            current_alt = msg.altitude

                            if initial_lat is None:
                                if msg.is_valid:
                                    initial_lat, initial_lon, initial_alt = current_lat, current_lon, current_alt
                                    
                                    utm_zone = int((initial_lon + 180) / 6) + 1
                                    
                                    # --- CORRECTION ---
                                    # We use pyproj.CRS.from_dict, which is the modern and robust way
                                    # to define a projection for the transformer.
                                    proj_utm_dict = {
                                        "proj": "utm",
                                        "zone": utm_zone,
                                        "ellps": "WGS84",
                                        "units": "m",
                                        "no_defs": True
                                    }
                                    proj_utm = pyproj.CRS.from_dict(proj_utm_dict)
                                    
                                    from_crs = "EPSG:4326"
                                    transformer = pyproj.Transformer.from_crs(from_crs, proj_utm, always_xy=True)
                                    
                                    origin_x, origin_y = transformer.transform(initial_lon, initial_lat)
                                    
                                    print("-" * 50)
                                    print("GPS Fix Acquired! Initializing coordinate system.")
                                    print(f"Origin (Lat, Lon): ({initial_lat:.6f}, {initial_lon:.6f})")
                                    print(f"Origin Altitude: {initial_alt:.2f} meters")
                                    print("-" * 50)
                                else:
                                    print("Waiting for a valid GPS fix...")
                            
                            else:
                                if msg.is_valid:
                                    current_x, current_y = transformer.transform(current_lon, current_lat)
                                    
                                    relative_x = current_x - origin_x
                                    relative_y = current_y - origin_y
                                    relative_z = current_alt - initial_alt

                                    print(f"Position (meters): x={relative_x:.2f}, y={relative_y:.2f}, z={relative_z:.2f}")
                                else:
                                    print("GPS signal lost or not valid. Waiting for fix...")
                                

                    except pynmea2.nmea.ParseError as e:
                        pass
                    
    except serial.SerialException as e:
        print(f"\nFATAL: Serial port error. Check port '{SERIAL_PORT}' is correct and not in use.")
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_real_time_position()