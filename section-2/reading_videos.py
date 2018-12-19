'''
A simple program to demonstrate how to load
and display videos using OpenCV
'''

# import openCV
import cv2

# the path of the video file
# pro tip: set the filename to 0 to use built in web cam...
#                                 ...and 1 to use external.
filename = 'data/sample_video.mp4'

# initialize a VideoCapture object with the video file
cap = cv2.VideoCapture(filename)


while True:

    # read a single frame from the video
    ret, frame = cap.read()

    if ret:
        # Display that frame in a window named Video
        cv2.imshow("Video", frame)
    else:
        break

    # wait 10 milliseconds for the user to hit a key
    k = cv2.waitKey(10)

    # break out of the look if user presses 'q'
    if k == ord('q'):
        break

# close the video file
cap.release()

# destory the windows
cv2.destroyAllWindows()
