'''
A simple program to demonstrate how to manipulate 
image properties using OpenCV
'''
import cv2

filename = 'data/test.jpg'
img = cv2.imread(filename)

# convert the image to a grayscale image
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the image to a HSV image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# convert the gray image to a binary image
ret, binary = cv2.threshold(grayscale, 70, 255, cv2.THRESH_BINARY)

# display the images
cv2.imshow("Original", img)
cv2.imshow("Grayscale", grayscale)
cv2.imshow("HSV", hsv)
cv2.imshow("Binary", binary)

# # resizing an image
# resized = cv2.resize(img, (400, 400))
# cv2.imshow("Resized", resized)

# # saving an image
# out = 'data/converted.jpg'
# cv2.imwrite(out, resized)

# Wait until the user closes the window
cv2.waitKey(0)

# destory all windows created by OpenCV
cv2.destroyAllWindows()
