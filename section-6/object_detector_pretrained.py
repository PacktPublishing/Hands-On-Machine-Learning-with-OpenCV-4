from utils import *
import numpy as np
import cv2
import sys


desc = '''Usage:
python3 object_detector_pretrained.py filename
The file can be an image or a video. To read video from webcam, pass filename as 0.
'''


if len(sys.argv) != 2:
    print(desc)
    exit()


data_file = sys.argv[1]
file_type = None

if data_file.split(".")[1] in ["png", "jpg", "jpeg", "tiff"]:
    file_type = "image"

if data_file.split(".")[1] in ["mov", "avi", "mp4", "mkv"]:
    file_type = "video"


# load the caffe model
model_name = 'model/MobileNetSSD_deploy.caffemodel'
model_proto = 'model/MobileNetSSD_deploy.prototxt.txt'

net = cv2.dnn.readNetFromCaffe(model_proto, model_name)


# running detection according to the file type

if file_type == "image":
    img = cv2.imread(data_file)
    detect_objects_and_draw_boxes(net, img)
    cv2.imshow("Object Detector", img)
    cv2.waitKey(0)

if file_type == "video":
    cap = cv2.VideoCapture(data_file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_objects_and_draw_boxes(net, frame)
        cv2.imshow("Object Detector", cv2.resize(frame, (1000, 700)))

        k = cv2.waitKey(10)
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
