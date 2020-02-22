import argparse

import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)

# command line parameters
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="./she.jpg", help="path to image file")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

PROTOTXT_FILE = "./deploy.prototxt.txt"
MODEL_FILE = "./res10_300x300_ssd_iter_140000.caffemodel"

# load our serialized model from disk
print("loading our serialized model from disk...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILE, MODEL_FILE)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
print("image height: " + h + ", width: " + w)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
print("computing object detections...")
net.setInput(blob)
detections = net.forward()

print(detections.shape)

exit(0)
# loop over the detections and draw boxes around the detected faces
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]
    # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
