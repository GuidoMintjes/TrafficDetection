from cv2 import cv2


video_capture = cv2.VideoCapture(0)

while True:
    frame, ret = video_capture.read()

    cv2.imshow(frame)