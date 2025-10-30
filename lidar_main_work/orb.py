import cv2
import numpy as np

from opt import read_txt, draw_points, polar_to_cartesian
from lines_hough import draw_lines_hough

def find_orb_transform(img1, img2, use_hough=False):
    # data = read_txt("lidar_main_work/scan_output_1761760401.txt")
    # points1 = polar_to_cartesian(np.array(data[10]['p']))
    # img1 = draw_points(points1, 800, 3)
    # img1 = draw_lines_hough(img1)
    # points2 = polar_to_cartesian(np.array(data[15]['p']))
    # img2 = draw_points(points2, 800, 3)
    # img2 = draw_lines_hough(img2)
    if use_hough:
        img1 = draw_lines_hough(img1)
        img2 = draw_lines_hough(img2)


    # for p in points:
    #     cv2.circle(img, (p[0], p[1]), 2, 255, -1)

    # Создание детектора ORB
    orb = cv2.ORB_create(nfeatures=500)  # можно настроить nfeatures

    # Нахождение ключевых точек и дескрипторов
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Сопоставление с помощью BFMatcher (Brute-Force)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Сортировка по расстоянию (лучшие — первые)
    matches = sorted(matches, key=lambda x: x.distance)

    # Визуализация (опционально)
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    # cv2.imshow('ORB Matches', img_matches)
    # cv2.waitKey(0)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Estimate the rigid transformation
    # Use RANSAC for robustness if outliers are expected
    M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
    return M

    # cv2.imshow('Tot ans', img1 + img2)
    # cv2.waitKey(0)

    # img1 = cv2.warpAffine(img1, M, [800, 800])
    # cv2.imshow('Tot ans', img1 + img2)
    # cv2.waitKey(0)
