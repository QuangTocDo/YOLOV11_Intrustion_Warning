import cv2
from ultralytics import YOLO
from yolodetect import YoloDetect
import numpy as np
# # --- 1. Load mô hình YOLO ---
# # Bạn có thể dùng 'yolov8n.pt' (nhẹ), 'yolov8s.pt', 'yolov8m.pt', ...
# model = YOLO("yolo11n.pt")
#
# # --- 2. Mở webcam ---
# cap = cv2.VideoCapture(0)  # 0 là webcam mặc định, có thể đổi thành 1 nếu dùng webcam ngoài
#
# if not cap.isOpened():
#     print("❌ Không thể mở webcam!")
#     exit()
#
# print("✅ Đang mở webcam, nhấn 'q' để thoát...")
#
# # --- 3. Vòng lặp đọc khung hình và detect ---
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("⚠️ Không đọc được frame từ webcam")
#         break
#
#     # --- 4. Dùng YOLO detect ---
#     results = model(frame, stream=True)  # stream=True để xử lý từng frame
#
#     # --- 5. Hiển thị kết quả ---
#     for r in results:
#         # Vẽ bounding box và label lên khung hình
#         annotated_frame = r.plot()
#
#         # Hiển thị lên cửa sổ
#         cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)
#
#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # --- 6. Giải phóng tài nguyên ---
# cap.release()
# cv2.destroyAllWindows()
if __name__ == "__main__":
    # Danh sách điểm tạo vùng cảnh báo (ví dụ hình tứ giác)
    polygon_points = [(100, 100), (500, 100), (500, 400), (100, 400)]

    detector = YoloDetect(
        model_path="weights/yolo11n.pt",
        detect_class="person",
        conf_threshold=0.5,
        alert_interval=10
    )

    cap = cv2.VideoCapture(0)  # Mở webcam
    if not cap.isOpened():
        print("Không thể mở webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Vẽ vùng giám sát
        cv2.polylines(frame, [np.array(polygon_points, np.int32)], True, (255, 0, 0), 2)

        # Phát hiện
        frame = detector.detect(frame, polygon_points)

        cv2.imshow("YOLOv11 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
