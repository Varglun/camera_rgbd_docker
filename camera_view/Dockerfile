# Используем официальный Python-образ
FROM python:3.11-slim

# Устанавливаем зависимости ОС, необходимые для OpenCV (видеоустройства, кодеки)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем приложение
COPY from_rpi.py .

# Открываем порт
EXPOSE 5000

# Запускаем сервер
CMD ["python", "from_rpi.py"]