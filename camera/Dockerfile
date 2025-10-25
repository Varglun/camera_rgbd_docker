# camera/Dockerfile
FROM ros:humble-perception

# Зависимости для Orbbec
RUN apt-get update && apt-get install -y \
    git libuvc-dev libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# Рабочая область
WORKDIR /root
RUN mkdir -p colcon_ws/src

# Клонируем драйвер Orbbec
RUN cd colcon_ws/src \
 && git clone https://github.com/orbbec/OrbbecSDK_ROS2.git

# Сборка
RUN bash -c "source /opt/ros/humble/setup.bash \
 && cd /root/colcon_ws \
 && colcon build --packages-select orbbec_ros"

# Настройка окружения
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
 && echo "source /root/colcon_ws/install/setup.bash" >> /root/.bashrc \
 && echo "export ROS_DISCOVERY_SERVER=TCPv4:[127.0.0.1]:11811" >> /root/.bashrc \
 && echo "export ROS_SUPER_CLIENT=TRUE" >> /root/.bashrc

# Запуск камеры
CMD ["ros2", "run", "orbbec_ros", "camera_node", "--ros-args", "-p", "enable_point_cloud:=true"]