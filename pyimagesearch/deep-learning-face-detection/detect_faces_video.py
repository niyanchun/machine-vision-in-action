import argparse
import time

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

PROTOTXT_FILE = "deploy.prototxt.txt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())
print("args: ", args)

print("loading our serialized model from disk...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILE, MODEL_FILE)

# initialize the video stream and allow the camera sensor to warm up
if args.get("video", None) is None:
    useCamera = True
    print("no input video, so start the camera video stream...")
    vs = VideoStream(src=0, resolution=(640, 480)).start()
    time.sleep(2.0)
else:
    useCamera = False
    vs = cv2.VideoCapture(args["video"])

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = frame if useCamera else frame[1]
    frame = imutils.resize(frame, width=600)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < args["confidence"]:
            continue
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
