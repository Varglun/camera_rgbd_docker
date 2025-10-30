import numpy as np
import open3d as o3d

def polar_to_cartesian(scan, min_dist=0, max_dist=12):
    """
    scan: np.array of shape (N, 2) — [[angle, distance], ...]
    returns: np.array of shape (2, N) — [[x1, x2, ...], [y1, y2, ...]]
    """
    angles = scan[:, 0]
    distances = scan[:, 1]
    valid = (distances > min_dist) & (distances < max_dist)
    angles = angles[valid]    
    distances = distances[valid]    
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return np.vstack([x, y])




def subsample(points, max_points=80):
    if points.shape[1] <= max_points:
        return points
    step = points.shape[1] // max_points
    return points[:, ::step]


def scan_to_pcd(scan_polar):
    pts2d = polar_to_cartesian(scan_polar)  # (2, N)
    pts3d = np.vstack([pts2d, np.zeros((1, pts2d.shape[1]))]).T  # (N, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    return pcd


def icp_2d(scan_prev, scan_curr, max_corr_dist=0.5):
    """
    scan_prev, scan_curr: np.array of shape (N, 2) — [[angle, dist], ...]
    returns: (theta, tx, ty)
    """
    # Преобразуем в облака
    target = scan_to_pcd(scan_prev)
    source = scan_to_pcd(scan_curr)

    # (Опционально) прореживаем
    voxel_size = 0.1  # метров
    target = target.voxel_down_sample(voxel_size)
    source = source.voxel_down_sample(voxel_size)

    # Запускаем ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_corr_dist,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    T = np.array(result.transformation)  # 4x4

    # Извлекаем угол поворота из матрицы 2x2
    R = T[0:2, 0:2]
    theta = np.arctan2(R[1, 0], R[0, 0])  # atan2(sin, cos)

    # Нормализуем в (-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    tx, ty = T[0, 3], T[1, 3]

    return theta, tx, ty


# target — прореживаем сильнее
target = target.voxel_down_sample(voxel_size=0.15)

# source — прореживаем слабее или не трогаем
source = source.voxel_down_sample(voxel_size=0.05)  # или вообще не downsample

# Пример данных
scan_prev = np.array([[0.0, 2.0], [0.1, 2.1], [0.2, 2.05], ...])  # (N, 2)
scan_curr = np.array([[0.0, 1.9], [0.1, 2.0], [0.2, 1.95], ...])  # (M, 2)

theta, tx, ty = icp_2d(scan_prev, scan_curr, max_corr_dist=0.3)

print(f"Поворот: {np.degrees(theta):.2f}°")
print(f"Смещение: ({tx:.3f}, {ty:.3f}) м")

