from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement
import cv2 as cv
import numpy as np


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
    farthest_point = None
    for point in path:
        if is_line_clear(im, start, point):
            farthest_point = point
        else:
            break  # Прекращаем проверку, если встретили препятствие
    return farthest_point