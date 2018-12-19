'''
A simple program to demonstrate how to load
and display images using OpenCV
'''

# import openCV
import cv2

# the path of the image file
filename = 'data/test.jpg'

# read the image from the given path
img = cv2.imread(filename)

# display the image in a window named "Image"
cv2.imshow("Image", img)

# Wait until the user closes the window
cv2.waitKey(0)

# destory all windows created by OpenCV
cv2.destroyAllWindows()
