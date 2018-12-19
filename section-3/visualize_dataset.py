import cv2

# importing numpy
import numpy as np


# load the datasets
apples = np.load('dataset/apple.npy')
bananas = np.load('dataset/banana.npy')


print("Number of apples:", len(apples))
print("Number of bananas:", len(bananas))


# pick one element from the dataset
apple = apples[0]
banana = bananas[0]

print("Shape of element:", apple.shape)

# reshape it into a 28x28 image
apple = np.reshape(apple, (28, 28))
banana = np.reshape(banana, (28, 28))

# display it
cv2.imshow("apple", apple)
cv2.imshow("banana", banana)

cv2.waitKey(0)
cv2.destroyAllWindows()
