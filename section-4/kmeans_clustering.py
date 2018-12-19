import numpy as np

# import KMeans from sklearn
from sklearn.cluster import KMeans

# import PCA
from sklearn.decomposition import PCA

# for plotting graphs
import matplotlib.pyplot as plt

# set some constants
N_SAMPLES = 500
TEST_SIZE = 0.2


# load the individual datasets
apples_full = np.load('dataset/apple.npy')
bananas_full = np.load('dataset/banana.npy')


# take only N_SAMPLES number of samples from each dataset
apples = apples_full[:N_SAMPLES]
bananas = bananas_full[:N_SAMPLES]

# put them together to build a collective dataset
dataset = np.concatenate((apples, bananas))


# initialize the PCA for dimensionality reduction
pca = PCA(n_components=2)

# Fit and transform the data into two components
reduced_data = pca.fit_transform(dataset)


# plot the dataset
plt.subplot(1, 2, 1)
x = reduced_data[:, 0]
y = reduced_data[:, 1]
plt.scatter(x, y, c='red')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("Original Dataset")


# choose and set the classifier algorithm
clf = KMeans(n_clusters=2)

# perform clustering
clf.fit(reduced_data)

# evaluate model
labels = list(clf.labels_)
# print(labels)

# get the total number of ones and zeros
total = len(labels)
ones = labels.count(1)
zeros = total - ones


print("Accuracy:", (total - abs(ones - zeros)) / total)

# draw the clusters
plt.subplot(1, 2, 2)
for (i, data_point) in enumerate(reduced_data):
    x = data_point[0]
    y = data_point[1]
    if clf.labels_[i] == 0:
        color = 'orange'
    else:
        color = 'green'
    plt.scatter(x, y, c=color)


centroids = clf.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1],
            c="red", s=200, marker='x')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("Clusters")
plt.show()


# predict a few samples

# pick out a few samples from the dataset which the classifier has not seen
apple = apples_full[-1]
banana = bananas_full[-1]

# put them togehter
test_data = np.concatenate(([apple], [banana]))

# perform PCA on them
reduced_test_data = pca.fit_transform(test_data)

# and predict the cluster in which they belong
print("Clusters predicted:")
print(clf.predict(reduced_test_data))
