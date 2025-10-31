import cv2 as cv
import os
import numpy as np
import json
from tqdm import tqdm

from opt import read_txt, icp_2d, polar_to_cartesian, transform_point_cloud, compose_transformations, combine_videos_side_by_side, filter_points_by_radius, robust_icp_2d, get_affine_matrix, apply_transform, draw_points
from opt_deepseek import find_2d_rotation, find_2d_rotation_sklearn_nn, find_2d_rotation_with_nn, find_2d_x_translation, find_2d_x_translation_with_nn, robust_x_translation_with_correspondences, iterative_2d_rotation
from opt_pystack import find_translate_rigid, find_rotation_only, find_translate_only
from lines_hough import draw_lines_hough
from orb import find_orb_transform
from path_find import find_path


class LidarToVideoSLAM:
    def __init__(self, out_folder: str = "", max_distance: int = 12, resolution: int = 2400, video_suff = ""):
        """
        Инициализация объекта для записи данных лидара в видео.
        
        :param out_folder: Папка для сохранения видеофайлов.
        :param max_distance: Максимальная дальность лидара в мм.
        :param resolution: Размер выходного изображения (в пикселях).
        """
        self.name_id = 0
        os.makedirs(out_folder, exist_ok=True)
        self.out_folder = out_folder
        self.max_distance = max_distance
        self.resolution = resolution
        self.out = None
        self.center = resolution // 2  # Центр изображения
        self.scale = resolution / (2 * max_distance)  # Масштаб для перевода мм в пиксели
        self.prev_point_cloud = None
        self.frame_only_env = None
        self.frame = None
        self.video_suff = video_suff
        self.transform = get_affine_matrix(0, 0, 0)
        self.lidar_shear = np.array([[0.066], [-0.03]]) # 0.064
        self.action = "stay"
        self.change = 0
        self.points_data = []
        self.start = True
        self.goal = [0, 0]

    def new_video(self):
        """
        Создает новый файл для записи видео.
        """
        self.name_id += 1
        name = f"lidar_video_{self.name_id}_{self.video_suff}.avi"
        full_path = os.path.join(self.out_folder, name)
        self.out = cv.VideoWriter(
            full_path,
            cv.VideoWriter_fourcc(*"FFV1"),  # Кодек без потерь
            5,  # Частота кадров
            (self.resolution, self.resolution)
        )

    def write(self, data: list, command: str, i=None):
        """
        Записывает данные лидара в текущий видеофайл.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
                     Угол задан в градусах, расстояние в мм, интенсивность в диапазоне [0, 255].
        """
        if self.out is None:
            print("No video output specified")
            return
        if self.start and len(data) < 400:
            return
        self.start = False
        self._points_data_to_frame(data, command)
        self.out.write(self.frame)

    def _points_data_to_frame_pyreg(self, data: list, command: str):
        """
        Преобразует данные лидара в кадр изображения.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
        :return: Изображение в формате BGR (numpy array).
        """
        # Создаем черное изображение
        
        if self.prev_point_cloud is None:
            self.frame = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            self.frame_only_env = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            cart_points = polar_to_cartesian(data)
            cart_points = filter_points_by_radius(cart_points, center=[-0.064, 0], radius=0.2)
            frame2 = draw_points(cart_points, self.resolution)
            mat_tr = np.eye(3, 3)
        else:
            cart_points = polar_to_cartesian(data)
            cart_points = filter_points_by_radius(cart_points, center=[-0.064, 0], radius=0.2)
            frame1 = draw_points(self.prev_point_cloud, self.resolution, max_distance=self.max_distance)
            frame2 = draw_points(cart_points, self.resolution, max_distance=self.max_distance)
            # mat_tr = find_orb_transform(frame1, frame2)
            # mat_tr = np.vstack([mat_tr, [0, 0, 1]])
            lines1 = draw_lines_hough(frame1)
            lines2 = draw_lines_hough(frame2)
            if command in ['a', 'd']:
                if self.action != 'turn':
                    self.change = 5
                if self.change != 0:
                    self.change -= 1
                    mat_tr = find_translate_rigid(lines1[:, :, 0], lines2[:, :, 0])
                else:
                    mat_tr = find_rotation_only(lines1[:, :, 0], lines2[:, :, 0])
                self.action = "turn"
            elif command in ['w', 'z']:
                if self.action != 'move':
                    self.change = 5
                if self.change != 0:
                    mat_tr = find_translate_rigid(lines1[:, :, 0], lines2[:, :, 0])
                    self.change -= 1
                else:
                    mat_tr = find_translate_only(lines1[:, :, 0], lines2[:, :, 0])
                self.action = "move"                
            else:
                if self.action != 'stay':
                    self.change = 5
                if self.change != 0:
                    self.change -= 1
                    mat_tr = find_translate_rigid(lines1[:, :, 0], lines2[:, :, 0])
                else:
                    mat_tr = np.eye(3, 3)     
                self.action = "stay"
                

        self.transform = self.transform @ mat_tr
        pixel_x_obj = int(self.center)
        pixel_y_obj = int(self.center) 
        frame_with_points = np.zeros_like(frame2, dtype=np.uint8)
        frame = cv.warpAffine(frame2, self.transform[:2, :].astype(np.float32), [self.resolution, self.resolution])
        self.prev_point_cloud = cart_points

        self.frame_only_env += frame
        cv.circle(frame_with_points, [pixel_x_obj, pixel_y_obj], 5, (0, 255, 0), -1)
        cv.line(frame_with_points, [pixel_x_obj, pixel_y_obj], [pixel_x_obj + 20, pixel_y_obj], (0, 0, 255), 2) 
        frame_with_points = cv.warpAffine(frame_with_points, self.transform[:2, :].astype(np.float32), [self.resolution, self.resolution])       
        self.frame = self.frame_only_env + frame_with_points

        cv.putText(self.frame, command, [0, self.resolution-10], 2, 3, (0, 0, 255), 2)

    def _points_data_to_frame(self, data: list, command: str):
        """
        Преобразует данные лидара в кадр изображения.
        
        :param data: Список точек в формате [(angle, distance, intensity), ...].
        :return: Изображение в формате BGR (numpy array).
        """
        # Создаем черное изображение
        
        if self.prev_point_cloud is None:
            self.frame = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            self.frame_only_env = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            cart_points = polar_to_cartesian(data)
            cart_points = filter_points_by_radius(cart_points, center=[-0.064, 0], radius=0.2)
            self.prev_point_cloud = cart_points
        else:
            cart_points = polar_to_cartesian(data)
            cart_points = filter_points_by_radius(cart_points, center=[-0.064, 0], radius=0.2)
            cart_points_trans = apply_transform((cart_points + self.lidar_shear.T), self.transform)
            center_point = apply_transform(np.array([[0, 0]]), self.transform)[0]
            main_point_cloud = filter_points_by_radius(self.prev_point_cloud, center_point, 3, True)
            theta, tx, ty = icp_2d(main_point_cloud.T  + self.lidar_shear, cart_points_trans.T  + self.lidar_shear, max_corr_dist=0.2,
                                   voxel_down_sample=0.1,
                                   remove_points_prev=None)
            if theta * 180 / np.pi > 10:
                return

            # if command in ['a', 'd']:
            #     self.action = "turn"
            #     # theta *= 1.3
            #     ff = 0
            #     tx = 0
            #     ty = 0
            #     # theta = iterative_2d_rotation(cart_points  + self.lidar_shear.T, self.prev_point_cloud  + self.lidar_shear.T)
            # elif command in ['q', 'e', 's']:
            #     self.action = "stay"
            #     theta = 0
            #     tx = 0
            #     ty = 0
            # elif command in ['w', 'z']:
            #     self.action = "move"
            #     theta = 0
            #     ty = 0
            #     # tx = robust_x_translation_with_correspondences(cart_points  + self.lidar_shear.T, self.prev_point_cloud  + self.lidar_shear.T)
            # else:
            #     theta = 0
            #     tx = 0
            #     ty = 0
            new_transform = get_affine_matrix(theta, tx, ty)
            self.transform = new_transform @ self.transform
            self.prev_point_cloud = np.vstack([self.prev_point_cloud, apply_transform((cart_points + self.lidar_shear.T), self.transform)])
        
        data = apply_transform((cart_points + self.lidar_shear.T), self.transform)
        center_point = apply_transform(np.array([[0, 0]]), self.transform)[0]
        orient_point = apply_transform(np.array([[0.2, 0]]),  self.transform)[0]

        for point in data:
            x, y = point
            
            # Переводим координаты в пиксели
            pixel_x = int(self.center + x * self.scale)
            pixel_y = int(self.center - y * self.scale)
            
            # Проверяем, что точка находится в пределах изображения
            if 0 <= pixel_x < self.resolution and 0 <= pixel_y < self.resolution:
                
                # Устанавливаем яркость пикселя
                self.frame_only_env[pixel_y, pixel_x] = [255, 255, 255]
        
                
        pixel_x_obj = int(self.center + center_point[0] * self.scale)
        pixel_y_obj = int(self.center - center_point[1] * self.scale)
        pixel_x_orient = int(self.center + orient_point[0] * self.scale)
        pixel_y_orient = int(self.center - orient_point[1] * self.scale)       
        path = find_path(self.frame_only_env, (pixel_x_obj, pixel_y_obj), (5, 10)) 
        # cv.imshow("1", self.frame_only_env)
        # cv.waitKey(0)
        self.frame = self.frame_only_env.copy()
        for p in path[1:-1]:
            self.frame[p.x, p.y] = (255, 0, 0)
            # cv.circle(im, (p.x, p.y), 1, (255, 0, 0), -1)        
        # Проверяем, что точка находится в пределах изображения
        if 0 <= pixel_x_obj < self.resolution and 0 <= pixel_y_obj < self.resolution:
            
            # Устанавливаем яркость пикселя
            cv.circle(self.frame, [pixel_x_obj, pixel_y_obj], 1, (0, 255, 0), -1)
            cv.line(self.frame, [pixel_x_obj, pixel_y_obj], [pixel_x_orient, pixel_y_orient], (0, 0, 255), 1)
        # cv.putText(self.frame, command, [0, self.resolution], 2, 3, (0, 0, 255), 2)


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
    lidar_to_video = LidarToVideoSLAM(out_folder="output_videos", max_distance=3, resolution=50, video_suff="scan_o3d2_test")
    
    # Создаем новый видеофайл
    lidar_to_video.new_video()
    data = read_txt("lidar_main_work/scan_output_1761760703.txt")
    
    # Записываем данные в видео
    for i, point_cloud in tqdm(enumerate(data), total=len(data)):  # 100 кадров
        # if i < 1:
            # continue
        lidar_to_video.write(np.array(point_cloud['p']), point_cloud['c'], i)
        # if i > 5*25:
        #     break
    
    # Останавливаем запись
    lidar_to_video.stop()
    
    # combine_videos_side_by_side("output_videos/lidar_video_1_scan_new_1.mp4",
    #                             "output_videos/lidar_video_1_new_1.avi",
    #                             "output_videos/concat_test3.mp4")