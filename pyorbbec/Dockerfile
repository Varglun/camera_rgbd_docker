# pyorbbec/Dockerfile
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libudev-dev \
    libusb-1.0-0-dev \
    libusb-1.0-0 \  
    libopenni2-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Клонируем pyorbbecsdk (ветка main = SDK v1.x = поддержка DaBai)
WORKDIR /root
RUN git clone https://github.com/orbbec/pyorbbecsdk.git
WORKDIR /root/pyorbbecsdk

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Сборка и установка pyorbbecsdk
RUN mkdir build && cd build \
 && cmake -Dpybind11_DIR=`pybind11-config --cmakedir` .. \
 && make -j$(nproc) \
 && make install

# Добавляем путь к библиотеке в PYTHONPATH
ENV PYTHONPATH="/root/pyorbbecsdk/install/lib/"

# Устанавливаем udev-правила (для доступа к USB)
RUN chmod +x ./scripts/install_udev_rules.sh \
 && ./scripts/install_udev_rules.sh

