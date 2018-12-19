import cv2
import numpy as np

# imporing the two algorithms, SVM and KNN
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

# some helper functions
from sklearn.model_selection import train_test_split as tts

# for evaluating our trained models
from sklearn.metrics import accuracy_score


# set some constants
N_SAMPLES = 1000
TEST_SIZE = 0.2
APPLE = 0
BANANA = 1


def normalize(data):
    "Takes a list or a list of lists and returns its normalized form"

    return np.interp(data, [0, 255], [-1, 1])


# load the individual datasets
apples_full = np.load('dataset/apple.npy')
bananas_full = np.load('dataset/banana.npy')


# take only N_SAMPLES number of samples from each dataset
apples = apples_full[:N_SAMPLES]
bananas = bananas_full[:N_SAMPLES]

# put them together to build a collective dataset
dataset = np.concatenate((apples, bananas))

# normalize the values
dataset = normalize(dataset)

# make the labels corresponding to the categories
labels = [APPLE] * N_SAMPLES + [BANANA] * N_SAMPLES

# prepare the data, split into training and testing
x_train, x_test, y_train, y_test = tts(dataset,
                                       labels,
                                       test_size=TEST_SIZE)

# choose and set the classifier algorithm
clf = KNN()

# train it
clf.fit(x_train, y_train)

# start predicting!
preds = clf.predict(x_test)

# evauluate the model
accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)
