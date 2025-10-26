# pyorbbec/depth_viewer.py
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType

def main():
    print("here")
    config = Config()
    print("here")
    pipeline = Pipeline()  # ← без аргументов!

    # Включаем поток глубины
    print("here")
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    print("here")
    depth_profile = profile_list.get_default_video_stream_profile()
    print("here")
    config.enable_stream(depth_profile)

    # Включаем RGB
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if depth_frame:
                width = depth_frame.get_width()
                height = depth_frame.get_height()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((height, width))
                depth_img = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imshow("Depth", depth_img)

            if color_frame:
                width = color_frame.get_width()
                height = color_frame.get_height()
                color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                color_data = color_data.reshape((height, width, 3))
                color_img = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
                cv2.imshow("RGB", color_img)

            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()