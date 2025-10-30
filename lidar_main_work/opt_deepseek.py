import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def find_2d_rotation(source_2d, target_2d, method='pca'):
    """
    Находит угол поворота между двумя 2D облаками точек
    """
    # Центрируем данные
    source_centered = source_2d - np.mean(source_2d, axis=0)
    target_centered = target_2d - np.mean(target_2d, axis=0)
    
    if method == 'pca':
        # Используем главные компоненты для определения ориентации
        source_cov = source_centered.T @ source_centered
        target_cov = target_centered.T @ target_centered
        
        # Собственные векторы дают направления главных осей
        _, source_vecs = np.linalg.eig(source_cov)
        _, target_vecs = np.linalg.eig(target_cov)
        
        # Угол между главными осями
        source_angle = np.arctan2(source_vecs[1, 0], source_vecs[0, 0])
        target_angle = np.arctan2(target_vecs[1, 0], target_vecs[0, 0])
        angle = target_angle - source_angle
        
    elif method == 'svd':
        # Метод SVD для 2D
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Извлекаем угол из матрицы поворота
        angle = np.arctan2(R[1, 0], R[0, 0])
    
    elif method == 'icp':
        # Итеративный метод для неточных данных
        angle = iterative_2d_rotation(source_2d, target_2d)
    
    return angle

def iterative_2d_rotation(source_2d, target_2d, max_iterations=20):
    """
    Итеративный метод для нахождения поворота в 2D
    """
    angle = 0.0
    best_error = float('inf')
    
    tree = KDTree(target_2d)
    
    for iteration in range(max_iterations):
        # Пробуем углы в диапазоне ±30 градусов с мелким шагом
        test_angles = np.linspace(angle - np.pi/6, angle + np.pi/6, 50)
        
        for test_angle in test_angles:
            # Матрица поворота 2D
            R = np.array([
                [np.cos(test_angle), -np.sin(test_angle)],
                [np.sin(test_angle), np.cos(test_angle)]
            ])
            
            # Применяем поворот
            rotated = (R @ source_2d.T).T
            
            # Находим ближайшие точки
            distances, _ = tree.query(rotated)
            error = np.median(distances)
            
            if error < best_error:
                best_error = error
                angle = test_angle
    
    return angle

def rotation_matrix_2d(angle):
    """Создает матрицу поворота 2D"""
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])


def find_2d_x_translation(source_2d, target_2d, method='histogram'):
    """
    Находит сдвиг по X между двумя 2D облаками точек
    """
    if method == 'histogram':
        # Гистограммный метод - очень эффективен для 2D
        x_source = source_2d[:, 0]
        x_target = target_2d[:, 0]
        
        # Создаем общие бины
        x_min = min(np.min(x_source), np.min(x_target))
        x_max = max(np.max(x_source), np.max(x_target))
        bins = np.linspace(x_min, x_max, 200)
        
        hist_source, _ = np.histogram(x_source, bins=bins, density=True)
        hist_target, _ = np.histogram(x_target, bins=bins, density=True)
        
        # Кросс-корреляция
        correlation = np.correlate(hist_source, hist_target, mode='full')
        shift_idx = np.argmax(correlation) - len(hist_source) + 1
        bin_width = bins[1] - bins[0]
        translation_x = shift_idx * bin_width
        
    elif method == 'median':
        # Простой медианный метод
        translation_x = np.median(target_2d[:, 0]) - np.median(source_2d[:, 0])
    
    elif method == 'robust':
        # Устойчивый метод с поиском соседей
        tree = KDTree(target_2d[:, 1].reshape(-1, 1))  # используем только Y для поиска соответствий
        
        # Для каждой точки ищем соседей по Y и усредняем разницу по X
        x_differences = []
        for point in source_2d:
            # Ищем точки с похожими Y координатами
            indices = tree.query_ball_point([point[1]], r=0.5)  # радиус можно настроить
            if indices:
                target_x = target_2d[indices[0], 0]
                x_diff = np.median(target_x) - point[0]
                x_differences.append(x_diff)
        
        translation_x = np.median(x_differences) if x_differences else 0.0
    
    return translation_x


def find_2d_rotation_with_nn(source_2d, target_2d, max_iterations=30, tolerance=1e-5):
    """
    Находит угол поворота используя итеративный подход с Nearest Neighbors
    """
    # Центрируем облака
    source_centroid = np.mean(source_2d, axis=0)
    target_centroid = np.mean(target_2d, axis=0)
    
    source_centered = source_2d - source_centroid
    target_centered = target_2d - target_centroid
    
    # Создаем KDTree для быстрого поиска ближайших соседей
    tree = KDTree(target_centered)
    
    angle = 0.0
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Матрица поворота для текущего угла
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Применяем поворот
        rotated_source = (R @ source_centered.T).T
        
        # Находим ближайшие точки в целевом облаке
        distances, indices = tree.query(rotated_source)
        
        # Создаем пары соответствующих точек
        corresponding_target = target_centered[indices]
        
        # Вычисляем оптимальный поворот для этих пар
        H = rotated_source.T @ corresponding_target
        U, S, Vt = np.linalg.svd(H)
        
        # Инкрементальная матрица поворота
        R_delta = Vt.T @ U.T
        
        # Извлекаем угол из матрицы поворота
        angle_delta = np.arctan2(R_delta[1, 0], R_delta[0, 0])
        angle += angle_delta
        
        # Проверяем сходимость
        current_error = np.mean(distances)
        if abs(prev_error - current_error) < tolerance:
            break
            
        prev_error = current_error
    
    return angle

# Альтернатива с sklearn NearestNeighbors
def find_2d_rotation_sklearn_nn(source_2d, target_2d):
    """
    Использует sklearn NearestNeighbors для лучшей производительности
    """
    from sklearn.neighbors import NearestNeighbors
    
    source_centered = source_2d - np.mean(source_2d, axis=0)
    target_centered = target_2d - np.mean(target_2d, axis=0)
    
    # Используем NearestNeighbors вместо KDTree
    nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nn.fit(target_centered)
    
    angle = 0.0
    best_error = float('inf')
    best_angle = 0.0
    
    # Поиск по сетке углов
    test_angles = np.linspace(-np.pi, np.pi, 72)  # шаг 5 градусов
    
    for test_angle in test_angles:
        R = np.array([
            [np.cos(test_angle), -np.sin(test_angle)],
            [np.sin(test_angle), np.cos(test_angle)]
        ])
        
        rotated = (R @ source_centered.T).T
        distances, _ = nn.kneighbors(rotated)
        error = np.median(distances)
        
        if error < best_error:
            best_error = error
            best_angle = test_angle
    
    # Уточнение вокруг лучшего угла
    refine_angles = np.linspace(best_angle - np.pi/36, best_angle + np.pi/36, 20)
    for test_angle in refine_angles:
        R = np.array([
            [np.cos(test_angle), -np.sin(test_angle)],
            [np.sin(test_angle), np.cos(test_angle)]
        ])
        
        rotated = (R @ source_centered.T).T
        distances, _ = nn.kneighbors(rotated)
        error = np.median(distances)
        
        if error < best_error:
            best_error = error
            best_angle = test_angle
    
    return best_angle

def find_2d_x_translation_with_nn(source_2d, target_2d):
    """
    Находит сдвиг по X используя Nearest Neighbors для поиска соответствий
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Создаем дерево для поиска по Y координатам
    # (предполагаем, что облака уже примерно выровнены по Y)
    nn_y = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn_y.fit(target_2d[:, 1].reshape(-1, 1))  # используем только Y координату
    
    x_differences = []
    
    for i, source_point in enumerate(source_2d):
        # Ищем точки в целевом облаке с похожими Y координатами
        distances, indices = nn_y.kneighbors([[source_point[1]]])
        
        if len(indices) > 0:
            # Берем ближайшие точки по Y
            target_points = target_2d[indices[0]]
            
            # Вычисляем разницу по X для этих точек
            x_diffs = target_points[:, 0] - source_point[0]
            
            # Используем медиану для устойчивости к выбросам
            median_x_diff = np.median(x_diffs)
            x_differences.append(median_x_diff)
    
    if not x_differences:
        return 0.0
    
    # Ищем наиболее вероятный сдвиг через гистограмму
    hist, bins = np.histogram(x_differences, bins=50, density=True)
    peak_bin = np.argmax(hist)
    translation_x = (bins[peak_bin] + bins[peak_bin + 1]) / 2
    
    return translation_x

def robust_x_translation_with_correspondences(source_2d, target_2d):
    """
    Еще более надежный метод с установлением соответствий
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Находим грубые соответствия по всем координатам
    nn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    nn.fit(target_2d)
    
    # Для каждой точки исходного облака находим ближайшие в целевом
    distances, indices = nn.kneighbors(source_2d)
    
    x_translations = []
    confidence_weights = []
    
    for i, (source_point, target_indices, dists) in enumerate(zip(source_2d, indices, distances)):
        target_points = target_2d[target_indices]
        
        # Вычисляем возможные сдвиги по X
        possible_translations = target_points[:, 0] - source_point[0]
        
        # Взвешиваем по обратному расстоянию (ближайшие точки важнее)
        weights = 1.0 / (dists + 1e-8)
        
        for trans, weight in zip(possible_translations, weights):
            x_translations.append(trans)
            confidence_weights.append(weight)
    
    if not x_translations:
        return 0.0
    
    # Взвешенная медиана
    sorted_indices = np.argsort(x_translations)
    sorted_translations = np.array(x_translations)[sorted_indices]
    sorted_weights = np.array(confidence_weights)[sorted_indices]
    
    cumulative_weights = np.cumsum(sorted_weights)
    median_weight = cumulative_weights[-1] / 2
    
    median_index = np.searchsorted(cumulative_weights, median_weight)
    translation_x = sorted_translations[median_index]
    
    return translation_x