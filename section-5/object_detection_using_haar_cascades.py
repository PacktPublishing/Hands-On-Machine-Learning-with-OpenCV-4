import cv2

# initialize a haar cascade from the xml file
cat_cascade = cv2.CascadeClassifier('data/haarcascade_frontal_catface.xml')

cap = cv2.VideoCapture('data/test2.mov')

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run detection
    cats = cat_cascade.detectMultiScale(gray, 1.1, 5)

    # draw rectangles arounf detected objects
    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('Object Detection using Haar Cascades', frame)

    k = cv2.waitKey(10)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
