from pystackreg import StackReg
import numpy as np
import cv2 as cv
from opt import draw_points, read_txt, polar_to_cartesian, transform_point_cloud


def find_translate_rigid(frame1, frame2):
    sr = StackReg(StackReg.RIGID_BODY)
    out_tra = sr.register_transform(frame2, frame1)
    tmat = sr.get_matrix()
    return tmat

def find_translate_only(frame1, frame2):
    sr = StackReg(StackReg.TRANSLATION)
    out_tra = sr.register_transform(frame2, frame1)
    tmat = sr.get_matrix()
    return tmat

def find_rotation_only(frame1, frame2):
    sr = StackReg(StackReg.SCALED_ROTATION)
    out_tra = sr.register_transform(frame2, frame1)
    tmat = sr.get_matrix()
    return tmat

if __name__ == "__main__":
    data = read_txt("lidar_main_work/scan_output_1761760087.txt")

    scan_prev = np.array(data[5]['p'])  # (N, 2)
    scan_curr = np.array(data[10]['p'])  # (N, 2)

    scan_prev = polar_to_cartesian(scan_prev) + np.array([[0.066], [-0.03]]).T # (2, N)  np.array([[0.066], [-0.03]])  np.array([[-0.02], [0.01]])
    scan_curr = polar_to_cartesian(scan_curr) + np.array([[0.066], [-0.03]]).T# (2, N)

    # matrix = find_translate_rigid(scan_prev, scan_curr)

    # frame = cv.warpAffine(draw_points(scan_curr), matrix.astype(np.float32), [600, 600])

    # frame = draw_points(scan_prev, 800, 12)
    # frame = draw_points([[0, 0]], 800, 12, frame, size=2, color=(0, 255, 0))

    # frame = draw_points(scan_curr, 800, 12, frame, color=[0, 0, 255])

    # theta = np.degrees(theta)
    # tx = 0
    # ty = 0

    # scan_curr_my = transform_point_cloud(scan_curr, np.radians(theta), [tx, ty])
    # new_center = transform_point_cloud([[0, 0]], np.radians(theta), [tx, ty])
    # frame = draw_points(scan_curr_my, 800, 12, frame, color=[255, 0, 0])
    # frame = draw_points(new_center, 800, 12, frame, size=2, color=[255, 0, 0])
    # cv.imshow("hello", frame)
    # cv.waitKey(0)
    # theta = np.radians(theta)

    # print(f"Поворот: {np.degrees(theta):.2f}°")
    # print(f"Смещение: ({tx:.3f}, {ty:.3f}) м")


    from pystackreg import StackReg
    import numpy as np

    # Загрузка изображений
    ref = draw_points(scan_prev, 600)[:, :, 0]
    mov = draw_points(scan_curr, 600)[:, :, 0]

    # Трансляция
    sr = StackReg(StackReg.RIGID_BODY)
    out_tra = sr.register_transform(mov, ref)
    tmat = sr.get_matrix()

    def normalize_image(image, target_dtype=np.uint8):
        """
        Нормализует изображение к целочисленному типу
        """
        # Обрезаем отрицательные значения до 0
        image_clean = np.maximum(image, 0)
        
        # Нормализуем к [0, 255] для uint8
        if target_dtype == np.uint8:
            image_normalized = (image_clean - image_clean.min()) 
            image_normalized = (image_normalized / image_normalized.max() * 255)
            return image_normalized.astype(np.uint8)
        
        # Для uint16
        elif target_dtype == np.uint16:
            image_normalized = (image_clean - image_clean.min())
            image_normalized = (image_normalized / image_normalized.max() * 65535)
            return image_normalized.astype(np.uint16)
        
        return image_clean

    im = cv.warpAffine(mov, tmat[:2, :].astype(np.float32), [600, 600]) + ref
    cv.imshow("hello", im)
    cv.waitKey(0)
