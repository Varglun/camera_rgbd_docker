# client/Dockerfile
FROM ros:humble-perception

# Зависимости для OpenCV и обработки
RUN apt-get update && apt-get install -y \
    python3-opencv python3-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Рабочая область (на случай, если захотите добавить свои пакеты)
WORKDIR /root
RUN mkdir -p colcon_ws/src

# Пустая сборка (можно добавить свои пакеты позже)
RUN bash -c "source /opt/ros/humble/setup.bash \
 && cd /root/colcon_ws \
 && colcon build"

# Окружение
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
 && echo "source /root/colcon_ws/install/setup.bash" >> /root/.bashrc \
 && echo "export ROS_DISCOVERY_SERVER=TCPv4:[127.0.0.1]:11811" >> /root/.bashrc \
 && echo "export ROS_SUPER_CLIENT=TRUE" >> /root/.bashrc