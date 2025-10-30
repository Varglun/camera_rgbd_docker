import numpy as np
import cv2 as cv

from opt import read_txt, icp_2d, polar_to_cartesian, transform_point_cloud, draw_points

data = read_txt("lidar_main_work/scan_output_1761760087.txt")



# # target — прореживаем сильнее
# target = target.voxel_down_sample(voxel_size=0.15)

# # source — прореживаем слабее или не трогаем
# source = source.voxel_down_sample(voxel_size=0.05)  # или вообще не downsample

# Пример данных
scan_prev = np.array(data[4]['p'])  # (N, 2)
scan_curr = np.array(data[5]['p'])  # (N, 2)

scan_prev = polar_to_cartesian(scan_prev) + np.array([[0.066], [-0.03]]).T # (2, N)  np.array([[0.066], [-0.03]])  np.array([[-0.02], [0.01]])
scan_curr = polar_to_cartesian(scan_curr) + np.array([[0.066], [-0.03]]).T# (2, N)

theta, tx, ty = icp_2d(scan_prev.T, scan_curr.T, max_corr_dist=0.1)

frame = draw_points(scan_prev, 800, 12)
frame = draw_points([[0, 0]], 800, 12, frame, size=2, color=(0, 255, 0))

frame = draw_points(scan_curr, 800, 12, frame, color=[0, 0, 255])

theta = np.degrees(theta)
tx = 0
ty = 0

scan_curr_my = transform_point_cloud(scan_curr, np.radians(theta), [tx, ty])
new_center = transform_point_cloud([[0, 0]], np.radians(theta), [tx, ty])
frame = draw_points(scan_curr_my, 800, 12, frame, color=[255, 0, 0])
frame = draw_points(new_center, 800, 12, frame, size=2, color=[255, 0, 0])
cv.imshow("hello", frame)
cv.waitKey(0)
theta = np.radians(theta)

print(f"Поворот: {np.degrees(theta):.2f}°")
print(f"Смещение: ({tx:.3f}, {ty:.3f}) м")

