import cv2

url = "https://giaothong.hochiminhcity.gov.vn/expandcameraplayer/?camId=662b80051afb9c00172dcaf6&camLocation=Nguy%E1%BB%85n%20Tr%C3%A3i%20-%20Nguy%E1%BB%85n%20C%C6%B0%20Trinh&camMode=camera&videoUrl=https://d2zihajmogu5jn.cloudfront.net/bipbop-advanced/bipbop_16x9_variant.m3u8"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
