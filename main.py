import cv2
import numpy as np
from imutils.video import VideoStream
from yolodetect import YoloDetect

points = []
detector = YoloDetect(
        model_path="yolo11n.pt",
        detect_class="person",
        conf_threshold=0.5,
        alert_interval=10
    )
video = cv2.VideoCapture(0)


def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon (frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=2)
    return frame

detect = False

while True:
    ret, frame = video.read()
    frame = cv2.flip(frame, 1)

    # Ve ploygon
    frame = draw_polygon(frame, points)

    if detect:
        frame = detector.detect(frame= frame, points= points)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('d'):
        points.append(points[0])
        detect = True

    # Hien anh ra man hinh
    cv2.imshow("Intrusion Warning", frame)

    cv2.setMouseCallback('Intrusion Warning', handle_left_click, points)
video.release()
cv2.destroyAllWindows()
