import cv2 as cv
import os
import numpy as np
import json
from tqdm import tqdm
import math

from opt import (read_txt, icp_2d, polar_to_cartesian_from_x,
                  filter_points_by_radius, get_affine_matrix, 
                 apply_transform, draw_points, affine_matrix_from_axes_and_offset,
                   angle_from_xy1_point_to_xy0_direction, down_sample_point_cloud,
                     angle_between_2d, find_first_free_on_line_numpy, line_points)
from path_find import find_path_to_nearest, find_farthest_point_on_path


MIN_LIDAR_POINTS_FOR_INIT = 400
MIN_LIDAR_POINTS_FOR_DETECT = 100

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
        self.goal = self.get_point_coord_on_map(np.array(goal))
        self.global_goal = self.get_point_coord_on_map(np.array(goal))
        self.local_goal = self.get_point_coord_on_map(np.array(goal))
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
        self.prev_data = np.array([])
        self.first_scan_made = False
            
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
    
    def get_point_coord_on_map(self, point):
        return np.floor(point * self.scale).astype(np.int32)
    
    def get_object_orientation_vector_in_map_coords(self):
        obj_cent = self.get_point_coord_on_map(self.object_center)
        obj_orient = self.get_point_coord_on_map(self.object_orient)
        vector_orientation = obj_orient - obj_cent
        return vector_orientation
    
    def get_local_goal_orientation_vector_in_map_coords(self):
        obj_cent = self.get_point_coord_on_map(self.object_center)
        vector_orientation = self.local_goal - obj_cent
        return vector_orientation    

    def move_point_cloud_from_lidar_to_robot_center(self, point_cloud):
        return point_cloud + self.lidar_shear.T

    def write_data(self, data: list):
        """
        Записывает данные лидара в текущий видеофайл.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
                     Угол задан в градусах, расстояние в мм, интенсивность в диапазоне [0, 255].
        """
        if (data is None) or len(data) < MIN_LIDAR_POINTS_FOR_DETECT:
            print("Not enough lidar points for detect. Waiting...")
            return
        if self.start and len(data) < MIN_LIDAR_POINTS_FOR_INIT:
            print("Not enough lidar points for initiate. Waiting...")
            return
        if self.prev_data.shape == data.shape and np.allclose(data, self.prev_data):
            return
        self.start = False
        self._points_data_to_frame(data)
        self.first_scan_made = True

    def write_video_frame(self):
        if self.save_to_video:
            im = self.map.copy()
            cv.imwrite("hello.png", im)
            im = self.draw_object_and_orient(im)
            cv.imwrite("hello.png", im)
            im = self.draw_path(im)
            cv.imwrite("hello.png", im)
            im = self.draw_local_goal(im)
            cv.imwrite("hello.png", im)
            self.out.write(im)
            
            
    def add_point_cloud_to_map_circle(self, point_cloud):
        for point in point_cloud:
            x, y = self.get_point_coord_on_map(point)
            cv.circle(self.map, [int(x), int(y)], self.wall_radius, [255, 255, 255], -1)

    def add_point_cloud_to_map_rect(self, point_cloud):
        for point in point_cloud:
            x, y = self.get_point_coord_on_map(point)
            # Рассчитываем координаты углов квадрата
            half_side = self.wall_radius  # так как сторона = 2 * wall_radius
            x1 = int(x - half_side)
            y1 = int(y - half_side)
            x2 = int(x + half_side)
            y2 = int(y + half_side)
            cv.rectangle(self.map, (x1, y1), (x2, y2), (255, 255, 255), -1)            
    
    def remove_robot_from_point_cloud(self, point_cloud):
        return filter_points_by_radius(point_cloud, center=[0, 0], radius=0.3)
    
    def closet_points_forehead(self, point_cloud):
        pass
    
    def draw_object_and_orient(self, im):            
        cv.circle(im, self.get_point_coord_on_map(self.object_center), 
                  math.ceil(self.object_radius_for_drawing * self.scale), 
                  (0, 255, 0), -1)
        cv.line(im, self.get_point_coord_on_map(self.object_center), 
                self.get_point_coord_on_map(self.object_orient),
                  (0, 0, 255), math.ceil(self.orient_line_width_for_draing * self.scale))
        return im        
    
    def find_next_local_goal(self):
        if not self.first_scan_made:
            return False
        path = find_path_to_nearest(self.map, 
                                    self.get_point_coord_on_map(self.object_center), self.goal)
        if path is None or len(path) == 0:
            print('No path can be found')
            print('Reset map')
            self.start = True        
            self.map = np.zeros((self.resolution, self.resolution, 3), np.uint8)
            self.prev_point_cloud = None
            self._points_data_to_frame(self.prev_data)
            points_from_goal = line_points(self.global_goal, self.get_point_coord_on_map(self.object_center))
            for point in points_from_goal:
                path = find_path_to_nearest(self.map, 
                                            self.get_point_coord_on_map(self.object_center), point)
                if len(path) > 0:
                    self.goal = point
                    break
            else:
                print("NO WAY. STUCK.")
            

        self.path = path
        local_goal = find_farthest_point_on_path(self.map, path, 
                                                      self.get_point_coord_on_map(self.object_center))
        if local_goal is not None:
            self.local_goal = local_goal

        return True
    
    def get_local_goal(self):
        if self.local_goal is not None:
            obj_orient = self.get_object_orientation_vector_in_map_coords()
            goal_orient = self.get_local_goal_orientation_vector_in_map_coords()
            local_goal_theta = angle_between_2d(obj_orient, goal_orient, deg=True)
            local_goal_distance = (np.linalg.norm(self.local_goal - self.get_point_coord_on_map(self.object_center)))
            return local_goal_theta, local_goal_distance / self.scale
        else:
            return None, None
    
    def draw_path(self, im):
        for p in self.path[1:-1]:
            cv.circle(im, [p.x, p.y], self.wall_radius, (255, 0, 0), -1)
            # im[p.x, p.y] = (255, 0, 0)
        return im
    
    def draw_local_goal(self, im):
        if self.local_goal is not None:
            cv.line(im, self.get_point_coord_on_map(self.object_center), 
                    self.local_goal, (255, 255, 0), math.ceil(self.orient_line_width_for_draing * self.scale))
            cv.imwrite("hello.png", im)
            theta, distance = self.get_local_goal()
            cv.putText(im, f"Theta: {int(theta)}. Distance: {distance:.2f}", (0, self.resolution), 1, 1.5, (255, 255, 0), 2)
        return im
        
        
    def _points_data_to_frame(self, data: list):
        """
        Преобразует данные лидара в кадр изображения.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
        :return: Изображение в формате BGR (numpy array).
        """
        self.prev_data = data
        # Создаем черное изображение
        cart_points = polar_to_cartesian_from_x(data)
        cart_points = self.move_point_cloud_from_lidar_to_robot_center(cart_points)
        cart_points = self.remove_robot_from_point_cloud(cart_points)
        cart_points = apply_transform(cart_points, self.transform)
        
        if self.prev_point_cloud is None:
            self.prev_point_cloud = cart_points
        else:            
            theta, tx, ty = icp_2d(self.prev_point_cloud.T, cart_points.T, max_corr_dist=0.1,
                                   voxel_down_sample=None,
                                   remove_points_prev=None)            
            if abs(theta * 180 / np.pi) > 15:
                print("Too much transform! Ignoring")
                return            
            new_transform = get_affine_matrix(theta, tx, ty)
            cart_points = apply_transform(cart_points, new_transform)
            self.transform = new_transform @ self.transform
            self.prev_point_cloud = np.vstack([self.prev_point_cloud, cart_points])      
            self.object_center = apply_transform([self.object_center], new_transform)[0]
            self.object_orient = apply_transform([self.object_orient], new_transform)[0]            
            
        self.add_point_cloud_to_map_rect(cart_points)
        if len(self.prev_point_cloud) > 5000:
            self.prev_point_cloud = down_sample_point_cloud(self.prev_point_cloud.T, 0.01)



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
                                      map_size_meters=8,
                                      video_name="go_test_path4",
                                      wall_radius=0.1,
                                      goal=[3.2, 3.2],
                                      start_position=[2, 2],
                                      orientation_x=[0, -1]
                                      )
    
    # Создаем новый видеофайл
    data = read_txt("lidar_main_work/scan_output_1761760087.txt")
    
    # Записываем данные в видео
    for i, point_cloud in tqdm(enumerate(data), total=len(data)):  # 100 кадров
        # if i < 1:
            # continue
        # if i % 50 == 0:
                 
        lidar_to_video.write_data(np.array(point_cloud['p']))
        if i % 10 == 0:
            lidar_to_video.find_next_local_goal()
        lidar_to_video.write_video_frame()

        if i >= 40:
            break
    
    # Останавливаем запись
    lidar_to_video.stop()