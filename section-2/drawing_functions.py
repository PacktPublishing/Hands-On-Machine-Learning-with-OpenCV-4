'''
A simple program to demonstrate how to 
draw on images using OpenCV
'''
import cv2

filename = 'data/test.jpg'
img = cv2.imread(filename)

# draw a rectangle on the image with the properties...

cv2.rectangle(img,            # where to draw
              (200, 100),     # top-left
              (500, 400),	    # bottom-right
              (255, 0, 0),    # color
              2)              # thickness

# draw a circle on the image with the properties...

cv2.circle(img,               # where to draw
           (650, 250),        # center
           100,	  		        # radius
           (0, 0, 255),       # color
           -1)                # thickness

# write text to the image with the properties...
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,              # where to draw
            'Apples',         # what text to write
            (10, 100),        # top-left coordinate
            font,             # what font to use
            4,				        # font size
            (0, 255, 255),    # color
            2)			          # line thickness


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
