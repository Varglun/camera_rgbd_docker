# lidar/Dockerfile
FROM ros:humble-perception

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    openssh-server supervisor zip libudev-dev \
    ros-humble-ament-cmake-core ros-humble-ament-cmake-python \
    && rm -rf /var/lib/apt/lists/*

# Настройка SSH (опционально)
RUN mkdir -p /var/run/sshd /var/log/supervisor \
 && echo "root:pass" | chpasswd \
 && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
 && echo "Port 2222" >> /etc/ssh/sshd_config

# Копируем supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Рабочая область
WORKDIR /root
RUN mkdir -p colcon_ws/src

# Скачиваем и правильно размещаем пакет лидара
RUN cd colcon_ws/src \
 && wget https://files.waveshare.com/upload/9/99/Ldrobot-lidar-ros2.zip \
 && unzip Ldrobot-lidar-ros2.zip \
 && rm Ldrobot-lidar-ros2.zip \
 && mv ldrobot-lidar-ros2-main/ldlidar_node . \
 && rmdir ldrobot-lidar-ros2-main

# Сборка
RUN bash -c "source /opt/ros/humble/setup.bash \
 && cd /root/colcon_ws \
 && colcon build --packages-select ldlidar_node"

# Настройка окружения (порт 42100, как у dds!)
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
 && echo "source /root/colcon_ws/install/setup.bash" >> /root/.bashrc \
 && echo "export ROS_DISCOVERY_SERVER=TCPv4:[127.0.0.1]:11811" >> /root/.bashrc \
 && echo "export ROS_SUPER_CLIENT=TRUE" >> /root/.bashrc

CMD ["/usr/bin/supervisord"]
