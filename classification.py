import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def dist_squared(a, b):
    return np.sum(np.square(a - b))


def set_labels_as_closest_cluster(data, labels, clusters):
    changed_count = 0

    for i, point in enumerate(data):
        closest_dist = dist_squared(point, clusters[0])
        closest_index = 0

        for j in range(1, len(clusters)):
            dist = dist_squared(point, clusters[j])

            if dist < closest_dist:
                closest_dist = dist
                closest_index = j

        if labels[i] != closest_index:
            changed_count += 1

        labels[i] = closest_index

    return changed_count


def get_barycenters_of_labels(data, labels, clusters):
    barycenters = np.zeros(clusters.shape)
    occurences = np.zeros(len(clusters), dtype=int)

    for i, label in enumerate(labels):
        barycenters[label] += data[i]
        occurences[label] += 1

    for i in range(len(barycenters)):
        barycenters[i] /= occurences[i]

    return barycenters


def show_classification(data, labels, clusters):
    color_map = "Set1"
    figure, axes = plt.subplots()

    axes.scatter(data[:, 0], data[:, 1], c=labels, s=10.0, cmap=color_map)
    axes.grid(True, alpha=0.3)

    man = figure.canvas.manager
    if man is not None:
        man.resize(1600, 800)

    plt.scatter(
        clusters[:, 0],
        clusters[:, 1],
        s=50.0,
        c=range(len(clusters)),
        edgecolors=(0, 0, 0),
        cmap=color_map,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # img = Image.open("classification_data/D1.gif")
    # img.show()
    # npimg = np.array(img)

    data = np.loadtxt("classification_data/gmm2d.asc", dtype=float)  # x
    labels = np.zeros(len(data), dtype=int)

    clusters = data[np.random.choice(len(data), size=4, replace=False)]  # y

    ite = 0
    updated_labels_count = set_labels_as_closest_cluster(data, labels, clusters)
    while updated_labels_count > 0:
        print(
            "k-means ite n°",
            ite,
            " - ",
            updated_labels_count,
            " labels changed.",
            sep="",
        )
        clusters = get_barycenters_of_labels(data, labels, clusters)
        updated_labels_count = set_labels_as_closest_cluster(data, labels, clusters)
        ite += 1

    show_classification(data, labels, clusters)
