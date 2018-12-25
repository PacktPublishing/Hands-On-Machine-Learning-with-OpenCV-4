import cv2
import numpy as np


# defining the function to capture the histogram from an image
def capture_histogram(path_of_sample):

    # read the image
    color = cv2.imread(path_of_sample)

    # convert to HSV
    color_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    # compute the histogram
    object_hist = cv2.calcHist([color_hsv],      # image
                               [0, 1],           # channels
                               None,             # no mask
                               [180, 256],       # size of histogram
                               [0, 180, 0, 256]  # channel values
                               )

    # min max normalization
    cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)

    return object_hist


def locate_object(frame, object_hist):

    # convert to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # apply back projection to image using object_hist as
    # the model histogram
    object_segment = cv2.calcBackProject(
        [hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)

    # find the contours
    img, contours, _ = cv2.findContours(
        object_segment,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    flag = None
    max_area = 0

    # find the contour with the greatest area
    for (i, c) in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            flag = i

    # get the rectangle
    if flag is not None and max_area > 1000:
        cnt = contours[flag]
        coords = cv2.boundingRect(cnt)
        return coords

    return None


# compute the color histogram
hist = capture_histogram('data/color.jpg')

cap = cv2.VideoCapture(0)


while True:

    # read from source
    _, frame = cap.read()

    # locate the object
    coords = locate_object(frame, hist)

    if coords:
        # unpack the coords
        x, y, w, h = coords

        # draw the bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Object Detection Using Color", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
