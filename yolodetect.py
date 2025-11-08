from shapely.geometry import Point, Polygon
import cv2
import datetime
import threading
from ultralytics import YOLO
from telegram_utils import send_telegram


def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    return polygon.contains(centroid)


class YoloDetect:
    def __init__(self,
                 model_path="yolo11n.pt",
                 detect_class="person",
                 conf_threshold=0.5,
                 alert_interval=15):

        # Load model YOLOv11
        self.model = YOLO(model_path)
        self.detect_class = detect_class
        self.conf_threshold = conf_threshold
        self.alert_interval = alert_interval
        self.last_alert = None

        # Tên lớp (class names)
        self.class_names = self.model.names

    def alert(self, img):
        """Hiển thị cảnh báo trên ảnh và gửi Telegram"""
        cv2.putText(img, "⚠️ ALARM DETECTED ⚠️", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Kiểm tra thời gian giữa các cảnh báo
        now = datetime.datetime.utcnow()
        if self.last_alert is None or (now - self.last_alert).total_seconds() > self.alert_interval:
            self.last_alert = now

            # Lưu ảnh cảnh báo (thu nhỏ để gửi nhanh hơn)
            alert_img_path = "alert/alert.png"
            resized = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))
            cv2.imwrite(alert_img_path, resized)

            # Gửi Telegram trong thread riêng (tránh chậm video)
            thread = threading.Thread(target=send_telegram, args=(alert_img_path,))
            thread.start()

        return img

    def detect(self, frame, points):
        """Phát hiện đối tượng trong frame và kiểm tra vùng cảnh báo"""
        results = self.model(frame, verbose=False)[0]  # YOLOv11 trả về list Results
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names[cls_id]
            conf = float(box.conf[0])

            # Lọc theo class và confidence
            if cls_name != self.detect_class or conf < self.conf_threshold:
                continue

            # Lấy tọa độ bbox và tâm
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Vẽ bbox + nhãn
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

            # Kiểm tra xem centroid có nằm trong vùng polygon không
            if isInside(points, centroid):
                frame = self.alert(frame)

        return frame
