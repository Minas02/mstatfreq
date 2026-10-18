import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_classification(data, labels, clusters):
    figure, axes = plt.subplots()
    cmap = 'Set1'

    axes.scatter(data[:, 0], data[:, 1], c=labels, s=10.0, cmap=cmap)
    axes.grid(True, alpha=0.3)

    man = figure.canvas.manager
    if man is not None:
        man.resize(1600, 800)

    plt.scatter(clusters[:, 0], clusters[:, 1], s=50.0, c=range(len(clusters)), edgecolors=(0, 0, 0), cmap=cmap)

    plt.tight_layout()
    plt.show()

def dist_squared(a, b):
    return np.sum(np.square(a - b))

def set_labels_as_closest_cluster(data, labels, clusters):
    for i, point in enumerate(data):
        closest_dist = dist_squared(point, clusters[0])
        closest_index = 0

        for j in range(1, len(clusters)):
            dist = dist_squared(point, clusters[j])

            if dist < closest_dist:
                closest_dist = dist
                closest_index = j

        labels[i] = closest_index

def get_barycenters_of_labels(data, labels, clusters):
    barycenters = np.zeros(clusters.shape)

    occurences = np.zeros(len(clusters), dtype=int)
    for i, label in enumerate(labels):
        barycenters[label] += data[i]
        occurences[label] += 1

    distances = np.zeros(len(clusters), dtype=float)
    for i in range(len(barycenters)):
        barycenters[i] /= occurences[i]
        distances[i] = dist_squared(barycenters[i], clusters[i])

    return barycenters, distances


if __name__ == "__main__":
    # img = Image.open("classification_data/D1.gif")
    # img.show()
    # npimg = np.array(img)

    data = np.loadtxt("classification_data/gmm2d.asc", dtype=float) # x
    labels = np.zeros(len(data), dtype=int)

    clusters = data[np.random.choice(len(data), size=4, replace=False)] # y

    i = 0
    while True:
        set_labels_as_closest_cluster(data, labels, clusters)
        clusters, distances = get_barycenters_of_labels(data, labels, clusters)
        max_dist = np.max(distances)
        print("k-means ite n°", i, " - ", max_dist, sep='')
        if max_dist < 1e-5:
            break
        i += 1

    show_classification(data, labels, clusters)

