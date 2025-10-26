# camera/Dockerfile
FROM ros:humble-perception

# Установка зависимостей для OpenNI (main branch)
RUN apt-get update && apt-get install -y \
    git libudev-dev libopenni2-dev \
    ros-humble-camera-info-manager \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN mkdir -p colcon_ws/src

# Клонируем и переключаемся на main (для OpenNI-устройств!)
RUN cd colcon_ws/src \
 && git clone https://github.com/orbbec/OrbbecSDK_ROS2.git \
 && cd OrbbecSDK_ROS2 \
 && git checkout main

# Сборка пакета orbbec_ros (именно так он называется в main)
RUN bash -c "source /opt/ros/humble/setup.bash \
 && cd /root/colcon_ws \
 && colcon build --packages-select orbbec_ros"

# Настройка окружения (порт должен совпадать с dds!)
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
 && echo "source /root/colcon_ws/install/setup.bash" >> /root/.bashrc \
 && echo "export ROS_DISCOVERY_SERVER=TCPv4:[127.0.0.1]:42100" >> /root/.bashrc \
 && echo "export ROS_SUPER_CLIENT=TRUE" >> /root/.bashrc

# Запуск ноды (в main используется orbbec_ros_node)
CMD ["bash", "-c", "source /root/.bashrc && ros2 run orbbec_ros orbbec_ros_node"]