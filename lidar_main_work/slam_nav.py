import cv2 as cv
import os
import numpy as np
import json
from tqdm import tqdm
import math

from opt import (read_txt, icp_2d, polar_to_cartesian_from_x,
                  filter_points_by_radius, get_affine_matrix, 
                 apply_transform, draw_points, affine_matrix_from_axes_and_offset,
                   angle_from_xy1_point_to_xy0_direction)
from path_find import find_path_to_nearest, find_farthest_point_on_path



class LidarToVideoSLAM:
    def __init__(self, out_folder: str = "", 
                 scale: float = 100, 
                 map_size_meters: float = 3.5, 
                 video_name: str = "go",
                 wall_radius: float = 0.1,
                 goal = [0.5, 3.5],
                 start_position = [3.5, 3.5],
                 orientation_x = [0, -1]):
        """
        Инициализация объекта для записи данных лидара в видео.
        
        :param out_folder: Папка для сохранения видеофайлов.
        :param scale: Число пикселей на метр. Влияет на скорость нахождения пути
        :param resolution: Размер выходного изображения (в пикселях).
        """
        os.makedirs(out_folder, exist_ok=True)
        self.out_folder = out_folder
        self.resolution = int(map_size_meters * 100)
        self.out = None
        self.center = self.resolution // 2  # Центр изображения
        self.scale = scale  # Масштаб для перевода м в пиксели
        self.prev_point_cloud = None
        self.video_name = video_name
        self.goal = np.array(goal)
        self.local_goal = np.array(goal)
        R = np.array([[0, -1],
                    [1,  0]])
        orientation_x = np.array(orientation_x)
        orientation_y = R @ orientation_x

        self.transform = affine_matrix_from_axes_and_offset(orientation_x, orientation_y, np.array(start_position))
        self.object_center = apply_transform(np.array([[0, 0]]), self.transform)[0]
        self.object_orient = apply_transform(np.array([[0.2, 0]]), self.transform)[0]
        self.lidar_shear = np.array([[0], [0]]) # 0.064 [[0.066], [-0.03]]
        self.start = True        
        self.map = np.zeros((self.resolution, self.resolution, 3), np.uint8)
        self.save_to_video = True
        self.wall_radius = int(wall_radius * scale)      
        self.path = []
        self.object_radius_for_drawing = 0.2
        self.orient_line_width_for_draing = 0.05
            
        if video_name is not None:
            self.save_to_video = True
            name = f"lidar_video_{self.video_name}.mp4"
            full_path = os.path.join(self.out_folder, name)
            self.out = cv.VideoWriter(
                full_path,
                cv.VideoWriter_fourcc(*"mp4v"),  # Кодек без потерь
                10,  # Частота кадров
                (self.resolution, self.resolution)
            )

    def move_point_cloud_from_lidar_to_robot_center(self, point_cloud):
        return point_cloud + self.lidar_shear.T

    def write(self, data: list):
        """
        Записывает данные лидара в текущий видеофайл.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
                     Угол задан в градусах, расстояние в мм, интенсивность в диапазоне [0, 255].
        """
        if self.start and len(data) < 400:
            print("Not enough lidar points for initiate. Waiting...")
            return
        self.start = False
        self._points_data_to_frame(data)
        if self.save_to_video:
            im = self.map.copy()
            im = self.draw_object_and_orient(im)
            im = self.draw_path(im)
            cv.imwrite("hello.png", im)
            self.out.write(im)
            
            
    def add_point_cloud_to_map(self, point_cloud):
        for point in point_cloud:
            x, y = point * self.scale
            cv.circle(self.map, [int(x), int(y)], self.wall_radius, [255, 255, 255], -1)
    
    def remove_robot_from_point_cloud(self, point_cloud):
        return filter_points_by_radius(point_cloud, center=[0, 0], radius=0.3)
    
    def closet_points_forehead(self, point_cloud):
        pass
    
    def draw_object_and_orient(self, im):            
        cv.circle(im, np.floor(self.object_center * self.scale).astype(np.int32), 
                  math.ceil(self.object_radius_for_drawing * self.scale), 
                  (0, 255, 0), -1)
        cv.line(im, np.floor(self.object_center * self.scale).astype(np.int32), 
                np.floor(self.object_orient * self.scale).astype(np.int32),
                  (0, 0, 255), math.ceil(self.orient_line_width_for_draing * self.scale))
        return im
    
    def find_next_local_goal(self):
        path = find_path_to_nearest(self.map, self.object_center.astype(np.int32), self.goal)
        self.path = path
        local_goal = find_farthest_point_on_path(self.map, path, self.object_center)
        local_goal_theta = angle_from_xy1_point_to_xy0_direction(self.transform, local_goal)
        local_goal_distance = np.linalg.norm(np.array(local_goal) - self.object_center)
        return local_goal_theta, local_goal_distance
    
    def draw_path(self, im):
        for p in self.path[1:-1]:
            im[p.x, p.y] = (255, 0, 0)
        return im
        
        
    def _points_data_to_frame(self, data: list):
        """
        Преобразует данные лидара в кадр изображения.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
        :return: Изображение в формате BGR (numpy array).
        """
        # Создаем черное изображение
        cart_points = polar_to_cartesian_from_x(data)
        cart_points = self.move_point_cloud_from_lidar_to_robot_center(cart_points)
        cart_points = self.remove_robot_from_point_cloud(cart_points)
        cart_points = apply_transform(cart_points, self.transform)
        
        if self.prev_point_cloud is None:
            self.prev_point_cloud = cart_points
        else:            
            theta, tx, ty = icp_2d(self.prev_point_cloud.T, cart_points.T, max_corr_dist=0.2,
                                   voxel_down_sample=0.05,
                                   remove_points_prev=None)            
            if abs(theta * 180 / np.pi) > 15:
                return            
            new_transform = get_affine_matrix(theta, tx, ty)
            cart_points = apply_transform(cart_points, new_transform)
            self.transform = new_transform @ self.transform
            self.prev_point_cloud = np.vstack([self.prev_point_cloud, cart_points])      
            self.object_center = apply_transform([self.object_center], new_transform)[0]
            self.object_orient = apply_transform([self.object_orient], new_transform)[0]            
            
        self.add_point_cloud_to_map(cart_points)



    def stop(self):
        """
        Останавливает запись видео и освобождает ресурсы.
        """
        if self.out is not None:
            self.out.release()
            self.out = None


    # def plan(self):



# Пример использования
if __name__ == "__main__":
    # Создаем объект для записи видео
    lidar_to_video = LidarToVideoSLAM(out_folder="output_videos",
                                      scale=100,
                                      map_size_meters=12,
                                      video_name="go_test_new",
                                      wall_radius=0.1,
                                      goal=[0.5, 3.5],
                                      start_position=[6, 6],
                                      orientation_x=[0, -1]
                                      )
    
    # Создаем новый видеофайл
    data = read_txt("lidar_main_work/scan_output_1761760703.txt")
    
    # Записываем данные в видео
    for i, point_cloud in tqdm(enumerate(data), total=len(data)):  # 100 кадров
        # if i < 1:
            # continue
        lidar_to_video.write(np.array(point_cloud['p']))
        # if i > 5*25:
        #     break
    
    # Останавливаем запись
    lidar_to_video.stop()