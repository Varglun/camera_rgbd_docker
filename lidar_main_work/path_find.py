from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt


def shift_trajectory_away_from_obstacles(
    grid: np.ndarray,
    trajectory: list[tuple[float, float]],
    max_offset: int = 10
) -> list[tuple[float, float]]:
    """
    Смещает траекторию в сторону максимального расстояния от препятствий,
    но не дальше max_offset клеток от оригинальной точки.
    
    Параметры:
    ----------
    grid : np.ndarray
        Бинарная карта, где 255 = препятствие, 0 = свободно.
    trajectory : list of (x, y)
        Траектория в координатах (float или int).
    max_offset : int
        Максимальное смещение в клетках.

    Возвращает:
    ----------
    new_trajectory : list of (x, y)
        Скорректированная траектория.
    """
    # Преобразуем карту: 1 = препятствие, 0 = свободно
    occupied = (grid == 255).astype(np.uint8)
    distance_map = distance_transform_edt(1 - occupied)  # свободное пространство => расстояния

    h, w = grid.shape
    new_traj = []

    for x, y in trajectory:
        ix, iy = int(round(x)), int(round(y))

        # Ограничиваем окно поиска
        y1 = max(0, iy - max_offset)
        y2 = min(h, iy + max_offset + 1)
        x1 = max(0, ix - max_offset)
        x2 = min(w, ix + max_offset + 1)

        local_dist = distance_map[y1:y2, x1:x2]
        local_mask = (grid[y1:y2, x1:x2] == 0)  # только свободные клетки

        if not np.any(local_mask):
            # Нет свободных клеток в окрестности — оставляем как есть
            new_traj.append((x, y))
            continue

        # Применяем маску: запрещаем выбирать препятствия
        masked_dist = np.where(local_mask, local_dist, -1)

        # Находим координату с максимальным расстоянием
        local_best_idx = np.unravel_index(np.argmax(masked_dist), masked_dist.shape)
        best_y_local, best_x_local = local_best_idx

        best_x = x1 + best_x_local
        best_y = y1 + best_y_local

        new_traj.append((float(best_x), float(best_y)))

    return np.array(new_traj, dtype=np.int32)


def find_nearest_reachable_point(im, start, goal):
    grid = Grid(matrix=(im[:, :, 0] == 0).tolist())  # 0 — проходимо, 1 — препятствие
    start_node = grid.node(*start)
    
    # Создаем сетку для поиска ближайшей точки
    rows, cols = im.shape[:2]
    min_distance = float('inf')
    nearest_node = None
    
    # Проходим по всем точкам сетки, чтобы найти ближайшую достижимую
    for r in range(rows):
        for c in range(cols):
            if grid.nodes[r][c].walkable:  # Проверяем, что точка проходима
                distance = np.linalg.norm(np.array([r, c]) - np.array(goal))
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = grid.nodes[r][c]
    
    return nearest_node

def find_path_to_nearest(im, start, goal):
    grid = Grid(matrix=(im[:, :, 0] == 0).tolist())  # 0 — проходимо, 1 — препятствие
    start_node = grid.node(*start)
    
    # Находим ближайшую достижимую точку
    # nearest_node = find_nearest_reachable_point(im, start, goal)
    nearest_node = grid.node(*goal)
    
    if nearest_node is None:
        raise ValueError("Нет достижимых точек рядом с целью")
    
    # Ищем путь до ближайшей достижимой точки
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, _ = finder.find_path(start_node, nearest_node, grid)
    
    return path

def find_safest_target_around(
    grid: np.ndarray,
    my_pos: tuple[float, float],
    target_pos: tuple[float, float],
    max_angle_deg: float = 15.0,
    step_deg: float = 1.0,
    safety_radius: int = 3
) -> np.ndarray:
    """
    Находит скорректированную целевую точку в пределах ±max_angle_deg,
    проверяя:
      1. Проходимость прямой линии от текущей позиции до кандидата.
      2. Количество свободных ячеек в окрестности кандидата.

    Углы проверяются в порядке возрастания отклонения: 0°, +step, -step, +2*step, ...

    Параметры:
    ----------
    grid : np.ndarray
        Бинарная карта (0 — свободно, !=0 — занято), shape (H, W), индексация [y, x].
    my_pos : (x0, y0)
        Текущая позиция (float).
    target_pos : (xt, yt)
        Исходная целевая точка (float).
    max_angle_deg : float
        Максимальное отклонение (±) в градусах.
    step_deg : float
        Шаг перебора угла в градусах.
    safety_radius : int
        Радиус окрестности для оценки свободного пространства.

    Возвращает:
    ----------
    np.ndarray of shape (2,) with dtype=np.int32 — (x, y) скорректированной цели.
    """
    x0, y0 = my_pos
    xt, yt = target_pos

    dx = xt - x0
    dy = yt - y0
    dist = np.hypot(dx, dy)

    if dist == 0:
        return np.array([int(round(xt)), int(round(yt))], dtype=np.int32)

    base_angle = np.arctan2(dy, dx)
    h, w = grid.shape

    best_score = -1
    best_point = (xt, yt)

    # Генерация отклонений: 0, +step, -step, +2*step, -2*step, ...
    max_step = int(np.ceil(max_angle_deg / step_deg))
    deviations_deg = [0.0]
    for k in range(1, max_step + 1):
        dev = k * step_deg
        if dev > max_angle_deg:
            break
        deviations_deg.append(dev)
        deviations_deg.append(-dev)

    # Вспомогательная функция: проверка линии
    def is_line_free(p0, p1):
        x0l, y0l = p0
        x1l, y1l = p1
        d = np.hypot(x1l - x0l, y1l - y0l)
        if d == 0:
            return True
        n_samples = max(1, int(np.ceil(d)))
        xs = np.linspace(x0l, x1l, n_samples)
        ys = np.linspace(y0l, y1l, n_samples)
        for x, y in zip(xs, ys):
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= iy < h and 0 <= ix < w):
                continue  # или return False, если выход за карту = недопустимо
            if grid[iy, ix] != 0:
                return False
        return True

    # Перебор кандидатов
    for dev_deg in deviations_deg:
        ang = base_angle + np.deg2rad(dev_deg)
        nx = x0 + dist * np.cos(ang)
        ny = y0 + dist * np.sin(ang)

        # Проверка проходимости прямой линии
        if not is_line_free((x0, y0), (nx, ny)):
            continue

        # Проверка границ для оценки окрестности
        ix = int(round(nx))
        iy = int(round(ny))
        if not (0 <= ix < w and 0 <= iy < h):
            continue

        # Оценка окрестности
        y1 = max(0, iy - safety_radius)
        y2 = min(h, iy + safety_radius + 1)
        x1 = max(0, ix - safety_radius)
        x2 = min(w, ix + safety_radius + 1)

        patch = grid[y1:y2, x1:x2]
        free_count = np.sum(patch == 0)

        if free_count > best_score:
            best_score = free_count
            best_point = (nx, ny)

    # Возвращаем целочисленные координаты (как вы указали)
    return np.array([int(round(best_point[0])), int(round(best_point[1]))], dtype=np.int32)

def is_obstacle_in_front(data, robot_width, max_distance):
    """
    Проверяет наличие препятствия в прямоугольной зоне спереди робота.

    Параметры:
        data: np.ndarray формы (N, 2)
              data[:, 0] — углы в градусах [0, 360)
              data[:, 1] — расстояния (метры)
        robot_width: ширина робота (м)
        max_distance: глубина зоны проверки вперёд (м)

    Возвращает:
        bool: True, если есть препятствие в зоне
    """
    angles_deg = data[:, 0]
    ranges = data[:, 1]

    # Преобразуем углы: 0° = вперёд → ось X
    # В стандартной системе: x = r * cos(θ), y = r * sin(θ)
    # Но угол в радианах должен быть от оси X, против часовой стрелки
    angles_rad = np.deg2rad(angles_deg)

    x = ranges * np.cos(angles_rad)
    y = ranges * np.sin(angles_rad)

    # Фильтрация: только валидные расстояния (конечные и положительные)
    valid = np.isfinite(ranges) & (ranges > 0)

    # Условия зоны спереди:
    in_corridor = (
        valid &
        (x > 0) &
        (x <= max_distance) &
        (np.abs(y) <= robot_width / 2.0)
    )

    return bool(np.any(in_corridor))

def reverse_angle(angle):
    rev = angle + 180
    if rev > 180:
        rev -= 360
    return rev

def point_opposite_direction(A, B, d):
    """
    Возвращает точку C, лежащую на расстоянии d от B
    в направлении, противоположном вектору AB.
    
    A, B — массивы или кортежи (x, y)
    d — скалярное расстояние (>= 0)
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    v = B - A
    norm = np.linalg.norm(v)
    
    if norm == 0:
        raise ValueError("Точки A и B совпадают — направление не определено.")
    
    unit_opposite = -v / norm
    C = A + d * unit_opposite
    return C  # или C, если хотите numpy массив

def is_line_clear(im, start, end):
    """
    Проверяет, можно ли провести прямую линию между start и end без пересечения с препятствиями.
    """
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        # Если текущая точка находится на препятствии, возвращаем False
        if im[y0, x0] != 0:
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return True

def find_farthest_point_on_path(im, path, start):
    """
    Находит самую дальнюю точку на пути, до которой можно провести прямую линию от начальной точки.
    """
    if path is None or len(path) == 0:
        return None
    farthest_point = None
    for point in path:
        if is_line_clear(im[:, :, 0], start, point):
            farthest_point = point
        else:
            break  # Прекращаем проверку, если встретили препятствие
    if farthest_point is not None:
        return np.array(farthest_point)
    return None, None
