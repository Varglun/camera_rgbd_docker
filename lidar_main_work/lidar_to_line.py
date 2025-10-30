import numpy as np
import cv2 as cv

from opt import read_txt, split_and_merge_segment, filter_segments, split_on_gaps, polar_to_cartesian


data = read_txt("lidar_main_work/scan_output3.txt")

points = np.array(data[1]['points'])
points = polar_to_cartesian(points)


# Параметры
gap_threshold = 0.2   # разрыв > 20 см — считаем новым объектом
eps = 0.05           # точность аппроксимации
min_length = 0.4     # минимальная длина отрезка
min_points = 20      # минимальное число точек

# 1. Разбиваем на подпоследовательности по "дырам"
subsequences = split_on_gaps(points, gap_threshold=gap_threshold)

# 2. Обрабатываем каждую подпоследовательность отдельно
all_segments = []  # будет содержать (start_global, end_global)
global_offset = 0

for subseq in subsequences:
    if len(subseq) < min_points:
        global_offset += len(subseq)
        continue
    
    # Применяем Split-and-Merge к подпоследовательности
    local_segments = split_and_merge_segment(subseq, eps)
    
    # Фильтруем по длине и числу точек
    local_segments = filter_segments(
        subseq, local_segments,
        min_length=min_length,
        min_points=min_points
    )
    
    # Преобразуем локальные индексы в глобальные
    for start_local, end_local in local_segments:
        start_global = global_offset + start_local
        end_global = global_offset + end_local
        all_segments.append((start_global, end_global))
    
    global_offset += len(subseq)

print(all_segments)

resolution = 1200
max_distance = 12
center = resolution // 2  # Центр изображения
scale = resolution / (2 * max_distance)  # Масштаб для перевода мм в пиксели

frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)

print(len(points))
for point in points:
    x, y = point
    
    # Переводим координаты в пиксели
    pixel_x = int(center + x * scale)
    pixel_y = int(center - y * scale)
    
    # Проверяем, что точка находится в пределах изображения
    if 0 <= pixel_x < resolution and 0 <= pixel_y < resolution:
        
        # Устанавливаем яркость пикселя
        frame[pixel_y, pixel_x] = [255, 255, 255]

for segment in all_segments:
    x0, y0 = points[segment[0]]
    
    # Переводим координаты в пиксели
    pixel_x0 = int(center + x0 * scale)
    pixel_y0 = int(center - y0 * scale)    
    x1, y1 = points[segment[1]]
    
    # Переводим координаты в пиксели
    pixel_x1 = int(center + x1 * scale)
    pixel_y1 = int(center - y1 * scale)     
    cv.line(frame, (pixel_x0, pixel_y0), (pixel_x1, pixel_y1), (0, 255, 0), 1)

cv.imshow("image", frame)
cv.waitKey(0)

# Закройте все окна OpenCV
cv.destroyAllWindows()
