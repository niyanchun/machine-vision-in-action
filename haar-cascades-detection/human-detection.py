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
        y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(frame, "face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    rects_eye = eye_cascade.detectMultiScale(gray_frame)
    for (x, y, w, h) in rects_eye:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        y = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(frame, "eye", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # 滤镜不能少...
    frame = cv2.bilateralFilter(frame, 0, 20, 5)
    cv2.imshow("preview", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
