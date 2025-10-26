# client/client.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')
        self.bridge = CvBridge()

        # Подписка на лидар
        # self.lidar_sub = self.create_subscription(
        #     LaserScan, '/scan', self.lidar_callback, 10)

        # Подписка на RGB
        self.rgb_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.rgb_callback, 10)

        # Подписка на глубину
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # # Подписка на облако точек (опционально)
        # self.points_sub = self.create_subscription(
        #     PointCloud2, '/camera/points', self.points_callback, 10)

        self.get_logger().info("Жду данные от лидара и камеры...")

    def lidar_callback(self, msg):
        self.get_logger().info(f"Лидар: {len(msg.ranges)} точек", throttle_duration_sec=2.0)

    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # cv2.imshow("RGB", cv_image)
        # cv2.waitKey(1)

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        # Нормализуем для отображения
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)
        # cv2.imshow("Depth", depth_normalized)
        # cv2.waitKey(1)

    def points_callback(self, msg):
        self.get_logger().info(f"Облако точек: {msg.height * msg.width} точек", throttle_duration_sec=2.0)

def main():
    rclpy.init()
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()