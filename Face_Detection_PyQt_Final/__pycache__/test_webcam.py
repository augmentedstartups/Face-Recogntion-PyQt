import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    cv2.imshow("out",frame)
    cv2.waitKey(1)