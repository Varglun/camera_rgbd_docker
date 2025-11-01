import open3d as o3d
import numpy as np
from opt import draw_points, read_txt, polar_to_cartesian

# Создаем 2D облако точек
data = read_txt("lidar_main_work/scan_output_1761760401.txt")
points = np.array(data[0]['p'])
points = polar_to_cartesian(points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.hstack([points, np.zeros((points.shape[0], 1))]))

# Параметры для detect_planar_patches [[3]]
distance_threshold = 0.02   # максимальное расстояние от точки до плоскости
ransac_n = 3                # минимальное количество точек для формирования плоскости
num_iterations = 1000       # количество итераций RANSAC
voxel_size = 0.05           # размер вокселя для сегментации
normal_variance_threshold = 0.5  # порог для различия нормалей
min_plane_edge_length = 0.2 # минимальная длина ребра плоскости

search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)

# Обнаруживаем плоские участки
patches = pcd.detect_planar_patches(
    normal_variance_threshold_deg=60,
    coplanarity_deg=75,
    outlier_ratio=0.75,
    min_plane_edge_length=0.2,
    min_num_points=10,
    search_param=search_param
)

# patches содержит список OrientedBoundingBox объектов [[4]]
for patch in patches:
    # Получаем параметры каждой найденной прямой
    center = patch.center
    extent = patch.extent
    R = patch.R  # матрица поворота
    
    # Нормаль плоскости (направление прямой)
    normal = R[:,2]


import matplotlib.pyplot as plt
import numpy as np

# patches содержит список OrientedBoundingBox объектов
for patch in patches:
    center = patch.center[:2]  # берем только x,y координаты
    normal = patch.R[:,2][:2]  # берем только x,y компоненты нормали
    
    # Вычисляем две точки на прямой
    length = np.max(patch.extent)  # используем размер bounding box'а
    point1 = center + normal * length/2
    point2 = center - normal * length/2
    
    # Рисуем линию
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'r-', linewidth=2)

# Рисуем исходные точки лидара
plt.scatter(points[:,0], points[:,1], c='b', s=1)

# Настройка графика
plt.axis('equal')
plt.show()    