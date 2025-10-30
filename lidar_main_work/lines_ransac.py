import numpy as np
import math

from opt import read_txt, draw_points, polar_to_cartesian

def point_to_line_distance(p, line):
    """
    Вычисляет расстояние от точки p до прямой, заданной двумя точками.
    line: ((x1, y1), (x2, y2))
    p: (x, y)
    """
    (x1, y1), (x2, y2) = line
    px, py = p
    # Векторное произведение для расстояния
    num = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1)
    if den == 0:
        return float('inf')
    return num / den

def fit_line_from_points(p1, p2):
    """Возвращает параметры прямой как кортеж двух точек."""
    return (p1, p2)

def ransac_line_sequential(points, 
                           max_iter=1000,
                           distance_threshold=0.05,
                           min_inliers=10,
                           min_length=0.3,
                           max_lines=10):
    """
    Последовательный RANSAC для извлечения нескольких прямых линий.
    
    Параметры:
        points: массив (N, 2)
        max_iter: макс. итераций на одну линию
        distance_threshold: порог расстояния для inlier'ов
        min_inliers: минимальное число inlier'ов для принятия линии
        min_length: минимальная длина сегмента (в метрах)
        max_lines: макс. число линий для поиска

    Возвращает:
        lines: список кортежей (line_params, inlier_points)
               где line_params = ((x1, y1), (x2, y2)) — две крайние точки сегмента
    """
    points = np.array(points)
    remaining = np.arange(len(points))
    lines = []

    for _ in range(max_lines):
        if len(remaining) < min_inliers:
            break

        best_inliers = []
        best_line = None

        for _ in range(max_iter):
            # Случайный выбор двух различных индексов
            idxs = np.random.choice(len(remaining), size=2, replace=False)
            i1, i2 = remaining[idxs]
            p1, p2 = points[i1], points[i2]

            # Пропускаем, если точки совпадают
            if np.allclose(p1, p2):
                continue

            line = (p1, p2)
            inliers = []

            for idx in remaining:
                p = points[idx]
                d = point_to_line_distance(p, line)
                if d <= distance_threshold:
                    inliers.append(idx)

            if len(inliers) >= min_inliers and len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_line = line

        # Если не нашли подходящую линию — выходим
        if best_line is None:
            break

        # Преобразуем inliers в массив
        best_inliers = np.array(best_inliers)
        inlier_points = points[best_inliers]

        # Определяем концевые точки сегмента (максимально удалённые друг от друга)
        # Проецируем точки на направляющий вектор линии
        v = np.array(best_line[1]) - np.array(best_line[0])
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            continue
        v_unit = v / v_norm

        projections = np.dot(inlier_points - best_line[0], v_unit)
        i_start = np.argmin(projections)
        i_end = np.argmax(projections)
        segment_length = projections[i_end] - projections[i_start]

        if segment_length < min_length:
            # Удаляем эти точки и продолжаем
            remaining = np.setdiff1d(remaining, best_inliers, assume_unique=True)
            continue

        # Формируем финальную линию как отрезок между крайними inlier'ами
        final_line = (inlier_points[i_start], inlier_points[i_end])

        lines.append((final_line, inlier_points))

        # Удаляем использованные точки
        remaining = np.setdiff1d(remaining, best_inliers, assume_unique=True)

    return lines


data = read_txt("lidar_main_work/scan_output_1761760401.txt")
points = polar_to_cartesian(np.array(data[1]['p']))

# Запускаем алгоритм
lines = ransac_line_sequential(
    points,
    distance_threshold=0.02,
    min_inliers=8,
    min_length=0.1,
    max_iter=500,
    max_lines=5
)

# Визуализация (опционально)
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], s=5, color='gray', alpha=0.5)

colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, (line, inliers) in enumerate(lines):
    (x1, y1), (x2, y2) = line
    plt.plot([x1, x2], [y1, y2], color=colors[i % len(colors)], linewidth=2)
    plt.scatter(inliers[:, 0], inliers[:, 1], s=10, color=colors[i % len(colors)])

plt.axis('equal')
plt.title('RANSAC Line Segments from Lidar Points')
plt.show()