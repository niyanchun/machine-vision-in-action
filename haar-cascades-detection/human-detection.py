import cv2

from imutils.video import VideoStream

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
vs = VideoStream(src=0).start()

while True:
    frame = vs.read()
    frame = cv2.resize(frame, (640, 480))
    # Haar Cascades必须使用灰度图
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    rects_face = face_cascade.detectMultiScale(gray_frame)

    for (x, y, w, h) in rects_face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    rects_eye = eye_cascade.detectMultiScale(gray_frame)
    for (x, y, w, h) in rects_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("preview", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
