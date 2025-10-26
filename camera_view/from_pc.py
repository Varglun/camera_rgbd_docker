import cv2

stream_url = "http://192.168.11.103:5000/video"  # замени на IP своей RPi
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Не удалось подключиться к потоку")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Поток оборвался")
        break

    # Делай что хочешь с frame
    cv2.imshow('Stream', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()