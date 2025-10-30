import cv2 as cv
import os
import numpy as np
import json
from tqdm import tqdm


class LidarToVideo:
    def __init__(self, out_folder: str = "", max_distance: int = 12, resolution: int = 2400, suffix=""):
        """
        Инициализация объекта для записи данных лидара в видео.
        
        :param out_folder: Папка для сохранения видеофайлов.
        :param max_distance: Максимальная дальность лидара в мм.
        :param resolution: Размер выходного изображения (в пикселях).
        """
        self.name_id = 0
        os.makedirs(out_folder, exist_ok=True)
        self.out_folder = out_folder
        self.max_distance = max_distance
        self.resolution = resolution
        self.out = None
        self.center = resolution // 2  # Центр изображения
        self.scale = resolution / (2 * max_distance)  # Масштаб для перевода мм в пиксели
        self.suffix = suffix

    def new_video(self):
        """
        Создает новый файл для записи видео.
        """
        self.name_id += 1
        name = f"lidar_video_{self.name_id}_{self.suffix}.avi"
        full_path = os.path.join(self.out_folder, name)
        self.out = cv.VideoWriter(
            full_path,
            cv.VideoWriter_fourcc(*"FFV1"),  # Кодек без потерь
            10,  # Частота кадров
            (self.resolution, self.resolution)
        )

    def write(self, data: list):
        """
        Записывает данные лидара в текущий видеофайл.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
                     Угол задан в градусах, расстояние в мм, интенсивность в диапазоне [0, 255].
        """
        if self.out is None:
            print("No video output specified")
            return
        
        frame = self._points_data_to_frame(data)
        self.out.write(frame)

    def _points_data_to_frame(self, data: list):
        """
        Преобразует данные лидара в кадр изображения.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
        :return: Изображение в формате BGR (numpy array).
        """
        # Создаем черное изображение
        frame = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        
        for point in data:
            angle_degrees, distance, intensity = point
            
            # Переводим угол в радианы
            angle_radians = np.radians(angle_degrees)
            
            # Переводим полярные координаты в декартовы
            x = distance * np.cos(angle_radians)
            y = distance * np.sin(angle_radians)
            
            # Переводим координаты в пиксели
            pixel_x = int(self.center + x * self.scale)
            pixel_y = int(self.center - y * self.scale)
            
            # Проверяем, что точка находится в пределах изображения
            if 0 <= pixel_x < self.resolution and 0 <= pixel_y < self.resolution:
                # Нормализуем интенсивность (предполагая, что она в диапазоне [0, 255])
                intensity = max(0, min(255, int(intensity)))
                
                # Устанавливаем яркость пикселя
                frame[pixel_y, pixel_x] = [intensity, intensity, intensity]
                
        
        return frame

    def stop(self):
        """
        Останавливает запись видео и освобождает ресурсы.
        """
        if self.out is not None:
            self.out.release()
            self.out = None


# Пример использования
if __name__ == "__main__":
    # Создаем объект для записи видео
    lidar_to_video = LidarToVideo(out_folder="output_videos", max_distance=3, resolution=800, suffix="new_1")
    
    # Создаем новый видеофайл
    lidar_to_video.new_video()
    data = []

    with open("lidar_main_work/scan_output_1761760087.txt") as data_file:
        for line in data_file:
            if line[0] == "{":
                data.append(json.loads(line))
    
    
    # Записываем данные в видео
    for point in tqdm(data):  # 100 кадров
        points = point['p']
        lidar_to_video.write(points)
    
    # Останавливаем запись
    lidar_to_video.stop()