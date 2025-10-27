"""
main_json_output.py
-------------------
Reads packets from an LD19 LiDAR, decodes them with CalcLidarData.py,
and prints each full 360° sweep as a JSON object to stdout.

Use cases:
 - Forwarding LiDAR data to another process
 - Capturing scans for logging or analysis
 - Quick debugging without the web viewer

Each output line is a JSON object:
{
    "timestamp": 1730000000.123,      # epoch seconds
    "num_points": 432,
    "points": [
        [angle_deg, distance_m, intensity],
        ...
    ]
}
"""

import json
import time
import serial
import math
import laspy
import numpy as np

# -------------------------------
# --- USER CONFIGURATION ----
# -------------------------------
SERIAL_PORT = "/dev/ttyUSB0"      # Change this to your LiDAR port
BAUD_RATE = 230400
TIMEOUT = 2.0

CURRENT_SCAN = []
# ----------------------------------------------------------
# LidarData class
# ----------------------------------------------------------
class LidarData:
    """
    Simple container for a decoded LiDAR packet.
    Stores all per-packet fields and lists of computed values.
    """
    def __init__(self, FSA, LSA, CS, Speed, TimeStamp,
                 Degree_angle, Angle_i, Distance_i, Intensity_i):
        self.FSA = FSA                    # First Sample Angle (°)
        self.LSA = LSA                    # Last Sample Angle (°)
        self.CS = CS                      # Checksum (not used)
        self.Speed = Speed                # Motor speed (°/s)
        self.TimeStamp = TimeStamp        # Packet time [ms]
        self.Degree_angle = Degree_angle  # List of point angles [°]
        self.Angle_i = Angle_i            # List of point angles [radians]
        self.Distance_i = Distance_i      # List of distances [m]
        self.Intensity_i = Intensity_i    # List of intensity values [0–255]

    def __repr__(self):
        """Readable summary when printed."""
        return (
            f"<LidarData {len(self.Distance_i)} pts "
            f"FSA={self.FSA:.2f}° LSA={self.LSA:.2f}° "
            f"Speed={self.Speed:.2f}°/s>"
        )

# ----------------------------------------------------------
# CalcLidarData function
# ----------------------------------------------------------
def CalcLidarData(str_):
    """
    Convert a full 90-character hex packet string into a LidarData object.

    Parameters
    ----------
    str_ : str
        The raw hexadecimal string (e.g. "54 2C 7F 00 ...") for one packet.

    Returns
    -------
    LidarData
        Parsed data containing lists of per-point angle, distance, intensity.
    """

    # --- Step 1: Clean up input string (remove spaces) ---
    s = str_.replace(' ', '')

    # --- Step 2: Parse packet header fields ---
    # Speed (2 bytes, little-endian) -> degrees per second
    Speed = int(s[2:4] + s[0:2], 16) / 100

    # First Sample Angle (FSA): starting angle of the packet [0.01° units]
    FSA = float(int(s[6:8] + s[4:6], 16)) / 100

    # Last Sample Angle (LSA): ending angle of the packet [0.01° units]
    LSA = float(int(s[-8:-6] + s[-10:-8], 16)) / 100

    # Timestamp (2 bytes)
    TimeStamp = int(s[-4:-2] + s[-6:-4], 16)

    # Checksum (last byte)
    CS = int(s[-2:], 16)

    # --- Step 3: Initialize arrays for decoded point data ---
    Degree_angle = []  # point angles in degrees
    Angle_i = []       # point angles in radians
    Distance_i = []    # point distances in meters
    Intensity_i = []   # per-point return intensity

    # --- Step 4: Compute angular step between points ---
    # 12 points per packet; handle wrap-around (e.g., 358°→2°)
    if LSA - FSA > 0:
        step = (LSA - FSA) / 12
    else:
        step = (LSA + 360 - FSA) / 12

    # Helper lambda: keep angle values in [0,360)
    circle = lambda d: d - 360 if d >= 360 else d

    # --- Step 5: Decode all 12 data points (3 bytes each) ---
    for i in range(0, 6 * 12, 6):
        # Distance (2 bytes, little-endian) → meters
        dist_mm = int(s[8+i+2 : 8+i+4] + s[8+i : 8+i+2], 16)
        Distance_i.append(dist_mm / 1000.0)

        # Intensity (1 byte)
        Intensity_i.append(int(s[8+i+4 : 8+i+6], 16))

        # Compute absolute angle for this point
        angle_deg = circle(step * (i // 6) + FSA)
        Degree_angle.append(angle_deg)
        Angle_i.append(math.radians(angle_deg))

    # --- Step 6: Package results into a LidarData object ---
    return LidarData(
        FSA, LSA, CS, Speed, TimeStamp,
        Degree_angle, Angle_i, Distance_i, Intensity_i
    )
# -------------------------------
# --- MAIN READER LOOP ----
# -------------------------------
def read_lidar():
    # --- Укажите имя файла для сохранения ---
    OUTPUT_FILE = "lidar_scans.jsonl"  # .jsonl = JSON Lines

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"✅ Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"❌ Serial connection failed: {e}")
        return

    # Открываем файл ОДИН РАЗ в режиме добавления
    with open(OUTPUT_FILE, 'a') as f_out:
        tmp = ""
        flag2c = False
        last_angle = 0
        full_scan = []

        while True:
            b = ser.read()
            if not b:
                continue

            v = int.from_bytes(b, "big")

            if v == 0x54:
                tmp += b.hex() + " "
                flag2c = True
                continue

            elif v == 0x2c and flag2c:
                tmp += b.hex()
                raw_hex = tmp.replace(" ", "")
                if len(raw_hex) == 90:  # 45 байт = 90 hex-символов
                    try:
                        data = CalcLidarData(tmp)
                        pts = list(zip(data.Degree_angle, data.Distance_i, data.Intensity_i))

                        # Обнаружение завершения оборота
                        if data.FSA < last_angle and full_scan:
                            # Формируем скан
                            scan = {
                                "timestamp": time.time(),
                                "num_points": len(full_scan),
                                "points": full_scan,
                            }
                            # Записываем как ОДНУ СТРОКУ JSON
                            f_out.write(json.dumps(scan) + "\n")
                            f_out.flush()  # Гарантируем запись на диск
                            print(f"💾 Saved scan with {len(full_scan)} points")

                            full_scan = []

                        full_scan.extend(pts)
                        last_angle = data.FSA

                    except Exception as e:
                        print(f"⚠️ Decode error: {e}")

                tmp = ""
                flag2c = False

            else:
                tmp += b.hex() + " "
                flag2c = False

def convert_to_LAS(lidar_frame):
    laser_scan = []
    for point in lidar_frame['points']:
        angle_deg, distance_m, intensity = point
        angle_rad = math.radians(angle_deg)  # convert angle to radians
        # Compute Cartesian coordinates
        x = distance_m * math.cos(angle_rad)
        y = distance_m * math.sin(angle_rad)
        z = 0  # or some fixed value
        laser_scan.append([x, y, z, intensity])
    return laser_scan


if __name__ == "__main__":
    read_lidar()