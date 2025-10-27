# lidar_mjpeg_server.py
import json
import time
import serial
import math
import threading
import numpy as np
import cv2
from flask import Flask, Response

# -------------------------------
# --- USER CONFIGURATION ----
# -------------------------------
SERIAL_PORT = "/dev/ttyUSB0"  # Измените под ваш порт
BAUD_RATE = 230400
TIMEOUT = 2.0
OUTPUT_FILE = "lidar_scans.jsonl"
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800
MAX_DISTANCE_M = 10.0  # Макс. дистанция для масштабирования

# -------------------------------
# --- GLOBAL STATE (thread-safe) ---
# -------------------------------
current_frame = None  # Последний отрисованный кадр (для MJPEG)
frame_lock = threading.Lock()
shutdown_event = threading.Event()

# ----------------------------------------------------------
# LidarData class & CalcLidarData — без изменений
# ----------------------------------------------------------
class LidarData:
    def __init__(self, FSA, LSA, CS, Speed, TimeStamp,
                 Degree_angle, Angle_i, Distance_i, Intensity_i):
        self.FSA = FSA
        self.LSA = LSA
        self.CS = CS
        self.Speed = Speed
        self.TimeStamp = TimeStamp
        self.Degree_angle = Degree_angle
        self.Angle_i = Angle_i
        self.Distance_i = Distance_i
        self.Intensity_i = Intensity_i

def CalcLidarData(str_):
    s = str_.replace(' ', '')
    Speed = int(s[2:4] + s[0:2], 16) / 100
    FSA = float(int(s[6:8] + s[4:6], 16)) / 100
    LSA = float(int(s[-8:-6] + s[-10:-8], 16)) / 100
    TimeStamp = int(s[-4:-2] + s[-6:-4], 16)
    CS = int(s[-2:], 16)

    Degree_angle = []
    Angle_i = []
    Distance_i = []
    Intensity_i = []

    if LSA - FSA > 0:
        step = (LSA - FSA) / 12
    else:
        step = (LSA + 360 - FSA) / 12

    circle = lambda d: d - 360 if d >= 360 else d

    for i in range(0, 6 * 12, 6):
        dist_mm = int(s[8+i+2 : 8+i+4] + s[8+i : 8+i+2], 16)
        Distance_i.append(dist_mm / 1000.0)
        Intensity_i.append(int(s[8+i+4 : 8+i+6], 16))
        angle_deg = circle(step * (i // 6) + FSA)
        Degree_angle.append(angle_deg)
        Angle_i.append(math.radians(angle_deg))

    return LidarData(FSA, LSA, CS, Speed, TimeStamp,
                     Degree_angle, Angle_i, Distance_i, Intensity_i)

# ----------------------------------------------------------
# --- LiDAR READER THREAD ---
# ----------------------------------------------------------
def lidar_reader():
    global current_frame
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        print(f"✅ Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"❌ Serial connection failed: {e}")
        return

    tmp = ""
    flag2c = False
    last_angle = 0
    full_scan = []

    with open(OUTPUT_FILE, 'a') as f_out:
        while not shutdown_event.is_set():
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
                if len(raw_hex) == 90:
                    try:
                        data = CalcLidarData(tmp)
                        pts = list(zip(data.Degree_angle, data.Distance_i, data.Intensity_i))

                        if data.FSA < last_angle and full_scan:
                            # --- Сохраняем в JSON ---
                            scan = {
                                "timestamp": time.time(),
                                "num_points": len(full_scan),
                                "points": full_scan,
                            }
                            f_out.write(json.dumps(scan) + "\n")
                            f_out.flush()
                            print(f"💾 Saved scan with {len(full_scan)} points")

                            # --- Отрисовка ---
                            img = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 255  # белый фон
                            center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
                            scale = min(IMAGE_WIDTH, IMAGE_HEIGHT) / (2 * MAX_DISTANCE_M)

                            for angle_deg, dist, intensity in full_scan:
                                if dist == 0 or dist > MAX_DISTANCE_M:
                                    continue
                                angle_rad = math.radians(angle_deg)
                                x = int(center[0] + dist * math.cos(angle_rad) * scale)
                                y = int(center[1] - dist * math.sin(angle_rad) * scale)  # Y вверх — инверсия
                                # Цвет по интенсивности (серый)
                                color = int(np.clip(intensity, 0, 255))
                                cv2.circle(img, (x, y), 2, (color, color, color), -1)

                            # Обновляем глобальный кадр
                            with frame_lock:
                                current_frame = img.copy()

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

    ser.close()

# ----------------------------------------------------------
# --- MJPEG STREAM GENERATOR ---
# ----------------------------------------------------------
def generate_mjpeg():
    global current_frame
    while not shutdown_event.is_set():
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None

        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            # Показываем "ожидание" если данных нет
            blank = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8) * 200
            cv2.putText(blank, "Waiting for LiDAR data...", (100, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)  # ~20 FPS максимум

# ----------------------------------------------------------
# --- FLASK APP ---
# ----------------------------------------------------------
app = Flask(__name__)

@app.route('/video')
def video():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <head><title>LiDAR Viewer</title></head>
        <body>
            <h1>Live LiDAR Scan</h1>
            <img src="/video" width="800" height="800" />
            <p>Data saved to: lidar_scans.jsonl</p>
        </body>
    </html>
    """

# ----------------------------------------------------------
# --- MAIN ---
# ----------------------------------------------------------
if __name__ == '__main__':
    # Запускаем LiDAR в фоновом потоке
    lidar_thread = threading.Thread(target=lidar_reader, daemon=True)
    lidar_thread.start()

    try:
        print("🚀 Starting MJPEG server on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        shutdown_event.set()
        lidar_thread.join(timeout=2)