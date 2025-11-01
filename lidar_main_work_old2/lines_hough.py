import cv2
import numpy as np

from opt import read_txt, draw_points, polar_to_cartesian


def draw_lines_hough(img):

    # 1. Create a "point cloud" on an image
    # img = np.zeros((800, 800), np.uint8)
    # data = read_txt("lidar_main_work/scan_output_1761760401.txt")
    # points = polar_to_cartesian(np.array(data[1]['p']))
    # img = draw_points(points, 800, 3)
    # for p in points:
    #     cv2.circle(img, (p[0], p[1]), 2, 255, -1)

    # 2. Apply Canny edge detection
    # edges = cv2.Canny(img, 50, 150)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(img, kernel, 1)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Detected Lines", edges)
    # cv2.waitKey(0)

    # 3. Apply the Probabilistic Hough Transform (more efficient)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=20, maxLineGap=30)

    ans = np.zeros_like(img)
    # 4. Draw the detected lines on the original image
    # img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(ans, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return ans



    # cv2.imshow('Detected Lines', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()