import argparse
import time
from collections import deque

import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# 绿色球的HSV色域空间范围
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
pts = deque(maxlen=args["buffer"])

# 判断是读入的视频文件，还是摄像头实时采集的，这里作区分是因为两种情况下后面的有些操作是有区别的
if args.get("video", None) is None:
    useCamera = True
    print("video is none, use camera...")
    vs = VideoStream(src=0).start()
else:
    useCamera = False
    vs = cv2.VideoCapture(args["video"])
    time.sleep(2.0)

while True:
    frame = vs.read()
    # 摄像头返回的数据格式为(帧数据)，而从视频抓取的格式为(grabbed, 帧数据)，grabbed表示是否读到了数据
    frame = frame if useCamera else frame[1]

    # 对于从视频读取的情况，frame为None表示数据读完了
    if frame is None:
        break

    # resize the frame(become small) to process faster(increase FPS)
    frame = imutils.resize(frame, width=600)
    # blur the frame to reduce high frequency noise, and allow
    # us to focus on the structural objects inside the frame
    # 通过高斯滤波去除掉一些高频噪声，使得重要的数据更加突出
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # convert frame to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # handles the actual localization of the green ball in the frame
    # inRange的作用是根据阈值进行二值化:阈值内的像素设置为白色(255)，阈值外的设置为黑色(0)
    mask = cv2.inRange(hsv, greenLower, greenUpper)

    # A series of erosions and dilations remove any small blobs that may be left on the mask
    # 腐蚀(erode)和膨胀(dilate)的作用:
    # 1. 消除噪声;
    # 2. 分割(isolate)独立的图像元素，以及连接(join)相邻的元素;
    # 3. 寻找图像中的明显的极大值区域或极小值区域
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 寻找轮廓，不同opencv的版本cv2.findContours返回格式有区别，所以调用了一下imutils.grab_contours做了一些兼容性处理
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum enclosing circle
        # and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        # 对于01二值化的图像，m00即为轮廓的面积, 一下公式用于计算中心距
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame, then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        pts.appendleft(center)

    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # compute the thickness of the line and draw the connecting line
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if useCamera:
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
