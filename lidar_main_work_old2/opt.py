import numpy as np
import open3d as o3d
import json
import cv2 as cv


def read_txt(file):
    data = []
    with open(file) as data_file:
        for line in data_file:
            if line[0] == "{":
                data.append(json.loads(line))
    return data

        

def polar_to_cartesian(scan, min_dist=0, max_dist=12):
    """
    scan: np.array of shape (N, 2) — [[angle, distance], ...]
    returns: np.array of shape (2, N) — [[x1, x2, ...], [y1, y2, ...]]
    """
    angles = np.radians(scan[:, 0]) * -1
    distances = scan[:, 1]
    valid = (distances > min_dist) & (distances < max_dist)
    angles = angles[valid]    
    distances = distances[valid]    
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return np.vstack([x, y]).T




def subsample(points, max_points=80):
    if points.shape[1] <= max_points:
        return points
    step = points.shape[1] // max_points
    return points[:, ::step]


def scan_to_pcd(pts2d):
    pts3d = np.vstack([pts2d, np.zeros((1, pts2d.shape[1]))]).T  # (N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    return pcd


def farthest_point_sampling(pcd, npoints):
    points = np.asarray(pcd.points)
    N = points.shape[0]
    if npoints >= N:
        return pcd

    sampled_indices = np.zeros(npoints, dtype=int)
    distances = np.full(N, np.inf)  # расстояния до ближайшей выбранной точки

    # Выбираем первую точку случайно (или можно взять первую)
    sampled_indices[0] = 0
    distances = np.linalg.norm(points - points[0], axis=1)

    for i in range(1, npoints):
        # Выбираем точку с максимальным расстоянием до уже выбранных
        farthest_idx = np.argmax(distances)
        sampled_indices[i] = farthest_idx
        # Обновляем расстояния
        new_distances = np.linalg.norm(points - points[farthest_idx], axis=1)
        distances = np.minimum(distances, new_distances)

    pcd_fps = pcd.select_by_index(sampled_indices.tolist())
    return pcd_fps

def icp_2d(scan_prev, scan_curr, max_corr_dist=0.5, voxel_down_sample=0.1, remove_points_prev=None):
    """
    scan_prev, scan_curr: np.array of shape (N, 2) — [[angle, dist], ...]
    returns: (theta, tx, ty)
    """
    # Преобразуем в облака
    target = scan_to_pcd(scan_prev)
    source = scan_to_pcd(scan_curr)

    if voxel_down_sample is not None:
        # (Опционально) прореживаем
        target = target.voxel_down_sample(voxel_down_sample)
        source = source.voxel_down_sample(voxel_down_sample)
    
    if remove_points_prev is not None:
        target = farthest_point_sampling(target, remove_points_prev)
    # Запускаем ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_corr_dist,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    # print(result.correspondence_set)
    T = np.array(result.transformation)  # 4x4

    # Извлекаем угол поворота из матрицы 2x2
    R = T[0:2, 0:2]
    theta = np.arctan2(R[1, 0], R[0, 0])  # atan2(sin, cos)

    # Нормализуем в (-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    tx, ty = T[0, 3], T[1, 3]

    return theta, tx, ty

def  robust_icp_2d(scan_prev, scan_curr, max_corr_dist=0.5):
    """
    scan_prev, scan_curr: np.array of shape (N, 2) — [[angle, dist], ...]
    returns: (theta, tx, ty)
    """
    # Преобразуем в облака
    target = scan_to_pcd(scan_prev)
    source = scan_to_pcd(scan_curr)

    # (Опционально) прореживаем
    # voxel_size = 0.1  # метров
    # target = target.voxel_down_sample(voxel_size)
    # source = source.voxel_down_sample(voxel_size)

    source = source.voxel_down_sample(voxel_size=0.05)
    target = target.voxel_down_sample(voxel_size=0.05)

    cl, ind = source.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    source = source.select_by_index(ind)

    cl, ind = target.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    target = target.select_by_index(ind)

    # Вычисление нормалей
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Выполнение ICP с робастной метрикой
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_corr_dist,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(
            kernel=o3d.pipelines.registration.TukeyLoss(k=0.1)  # Робастная ядро-функция
        ),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=200
        )
    )

    # print(result.correspondence_set)
    T = np.array(result.transformation)  # 4x4

    # Извлекаем угол поворота из матрицы 2x2
    R = T[0:2, 0:2]
    theta = np.arctan2(R[1, 0], R[0, 0])  # atan2(sin, cos)

    # Нормализуем в (-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    tx, ty = T[0, 3], T[1, 3]

    return theta, tx, ty


def transform_point_cloud(points, angle, translation):
    """
    Преобразует облако точек с учетом угла поворота и смещения.

    :param points: Исходное облако точек [(x1, y1), (x2, y2), ...].
    :param angle: Угол поворота в радианах.
    :param translation: Смещение (tx, ty).
    :return: Новое облако точек [(x1', y1'), (x2', y2'), ...].
    """
    # Разбиваем смещение на компоненты
    tx, ty = translation

    # Преобразуем угол в матрицу поворота
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Создаем матрицу поворота
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                                [sin_theta,  cos_theta]])

    # Преобразуем точки
    transformed_points = []
    for x, y in points:
        # Применяем поворот
        rotated_point = np.dot(rotation_matrix, np.array([x, y]))
        # Применяем смещение
        transformed_point = rotated_point + np.array([tx, ty])
        transformed_points.append(transformed_point)

    return np.array(transformed_points)

def draw_rect_at_center(cx, cy, img, square_size):
    top_left = (cx - square_size // 2, cy - square_size // 2)
    bottom_right = (cx + square_size // 2, cy + square_size // 2)
    cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), -1)    

def get_affine_matrix(theta, tx, ty):
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    M = np.array([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty],
        [0, 0, 1 ]
    ], dtype=np.float32)
    return M    

def apply_transform(points, T):
    """Применяет 3×3 трансформацию к точкам [N, 2]"""
    points_h = np.hstack([points, np.ones((len(points), 1))])
    return (T @ points_h.T).T[:, :2]

def compose_transformations(transform1, transform2):
    """
    Объединяет две трансформации в одну.

    :param transform1: Первая трансформация [theta1, [tx1, ty1]].
    :param transform2: Вторая трансформация [theta2, [tx2, ty2]].
    :return: Итоговая трансформация [theta, [tx, ty]].
    """
    theta1, (tx1, ty1) = transform1
    theta2, (tx2, ty2) = transform2

    # Создаем матрицы для каждой трансформации
    cos_theta1, sin_theta1 = np.cos(theta1), np.sin(theta1)
    cos_theta2, sin_theta2 = np.cos(theta2), np.sin(theta2)

    T1 = np.array([
        [cos_theta1, -sin_theta1, tx1],
        [sin_theta1, cos_theta1, ty1],
        [0, 0, 1]
    ])

    T2 = np.array([
        [cos_theta2, -sin_theta2, tx2],
        [sin_theta2, cos_theta2, ty2],
        [0, 0, 1]
    ])

    # Умножаем матрицы (композиция трансформаций)
    T_final = np.dot(T2, T1)

    # Извлекаем параметры из результирующей матрицы
    theta_final = np.arctan2(T_final[1, 0], T_final[0, 0])  # Угол поворота
    tx_final = T_final[0, 2]  # Смещение по X
    ty_final = T_final[1, 2]  # Смещение по Y

    return [theta_final, [tx_final, ty_final]]

import cv2

def combine_videos_side_by_side(video1_path, video2_path, output_path):
    """
    Объединяет два видео по горизонтали.

    :param video1_path: Путь к первому видео.
    :param video2_path: Путь ко второму видео.
    :param output_path: Путь для сохранения результирующего видео.
    """
    # Открываем видеозахват для обоих видео
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # Проверяем, что оба видео успешно открыты
    if not cap1.isOpened() or not cap2.isOpened():
        print("Ошибка: Не удалось открыть одно или оба видео.")
        return

    # Получаем параметры первого видео (ширина, высота, частота кадров)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # Получаем параметры второго видео
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Убедимся, что высота обоих видео одинакова (если нет, изменяем размеры)
    if height1 != height2:
        print(f"Высота видео разная: {height1} и {height2}. Изменяем размеры...")
        height = min(height1, height2)  # Выбираем минимальную высоту
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    else:
        height = height1

    # Создаем объект для записи нового видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width1 + width2, height))

    # Читаем кадры из обоих видео и объединяем их
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Если хотя бы одно видео закончилось, выходим из цикла
        if not ret1 or not ret2:
            break

        # Изменяем размеры кадров, если они не совпадают по высоте
        if frame1.shape[0] != frame2.shape[0]:
            frame1 = cv2.resize(frame1, (width1, height))
            frame2 = cv2.resize(frame2, (width2, height))

        # Объединяем кадры по горизонтали
        combined_frame = cv2.hconcat([frame1, frame2])

        # Записываем объединенный кадр в выходное видео
        out.write(combined_frame)

    # Освобождаем ресурсы
    cap1.release()
    cap2.release()
    out.release()
    print(f"Видео успешно сохранено: {output_path}")




def filter_points_by_radius(points, center=(0, 0), radius=0.2, inside=False):
    """
    Удаляет точки, находящиеся в пределах заданного радиуса от центра.

    :param points: Облако точек в формате numpy массива (N, 2).
    :param center: Координаты центра (x, y).
    :param radius: Радиус, внутри которого точки будут удалены.
    :return: Отфильтрованное облако точек.
    """
    # Вычисляем расстояние каждой точки от центра
    distances = np.linalg.norm(points - np.array(center), axis=1)
    
    # Фильтруем точки, оставляя только те, которые находятся за пределами радиуса
    if inside:
        filtered_points = points[distances < radius]
    else:
        filtered_points = points[distances >= radius]
    
    return filtered_points

import numpy as np

def perpendicular_distance(p, a, b):
    """Расстояние от точки p до прямой, проходящей через a и b."""
    if np.allclose(a, b):
        return np.linalg.norm(p - a)
    return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

def split_and_merge_segment(points, eps):
    """Возвращает список индексов сегментов: [(start0, end0), (start1, end1), ...]"""
    if len(points) < 2:
        return []
    segments = []
    _split_rec(points, 0, len(points) - 1, eps, segments)
    return segments

def _split_rec(points, first, last, eps, segments):
    if last - first < 1:
        return

    a, b = points[first], points[last]
    max_dist = 0.0
    max_idx = first

    for i in range(first + 1, last):
        d = perpendicular_distance(points[i], a, b)
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > eps:
        # Рекурсивно разбиваем
        _split_rec(points, first, max_idx, eps, segments)
        _split_rec(points, max_idx, last, eps, segments)
    else:
        print(max_dist)
        # Добавляем сегмент [first, last]
        segments.append((first, last))

def filter_segments(points, segments, min_length=0.0, min_points=2):
    filtered = []
    for start, end in segments:
        num_pts = end - start + 1
        length = np.linalg.norm(points[end] - points[start])
        
        if length >= min_length and num_pts >= min_points:
            filtered.append((start, end))
    return filtered

def fit_line(points_segment):
    # points_segment: массив точек (N, 2)
    x = points_segment[:, 0]
    y = points_segment[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]  # y = m*x + c
    return m, c

import numpy as np

def split_on_gaps(points, gap_threshold=0.2):
    """
    Разбивает упорядоченный массив точек на подпоследовательности,
    разрывая там, где расстояние между соседними точками > gap_threshold.
    
    Возвращает список numpy-массивов.
    """
    if len(points) == 0:
        return []
    
    subsequences = []
    current_subseq = [points[0]]
    
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i - 1])
        if dist > gap_threshold:
            # Завершаем текущую подпоследовательность
            subsequences.append(np.array(current_subseq))
            current_subseq = [points[i]]
        else:
            current_subseq.append(points[i])
    
    # Не забываем добавить последнюю
    if current_subseq:
        subsequences.append(np.array(current_subseq))
    
    return subsequences

def draw_points(points, resolution=600, max_distance=12, frame=None, color=[255, 255, 255], size=None):
    if frame is None:
        frame = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    
    center = resolution // 2  # Центр изображения
    scale = resolution / (2 * max_distance)  # Масштаб для перевода мм в пиксели    

    print(len(points))
    for point in points:
        x, y = point
        
        # Переводим координаты в пиксели
        pixel_x = int(center + x * scale)
        pixel_y = int(center - y * scale)
        
        # Проверяем, что точка находится в пределах изображения
        if 0 <= pixel_x < resolution and 0 <= pixel_y < resolution:
            
            # Устанавливаем яркость пикселя
            frame[pixel_y, pixel_x] = color   
            if size is not None:
                cv.circle(frame, [pixel_y, pixel_x], size, color, -1)
    return frame