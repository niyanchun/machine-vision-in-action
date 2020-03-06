import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
vs = cv2.VideoCapture("/Users/allan/Downloads/TownCentreXVID.avi")
threshold = 0.7

while True:
    grabbed, frame = vs.read()
    if grabbed:
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        rects, weights = hog.detectMultiScale(gray_frame)
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
