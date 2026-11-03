import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np

# Here is our original code for the k-means classification
# We asked Claude to optimize the code using numpy functions in order to reduce the execution time
# Here is the exact prompt "Can you rewrite the following functions using numpy to optimize them"
# followed by the 3 following functions:
#
# def dist_squared(a, b):
#     return np.sum(np.square(a - b))
#
#
# def set_labels_as_closest_cluster(data, labels, clusters):
#     changed_count = 0
#
#     for i, point in enumerate(data):
#         closest_dist = dist_squared(point, clusters[0])
#         closest_index = 0
#
#         for j in range(1, len(clusters)):
#             dist = dist_squared(point, clusters[j])
#
#             if dist < closest_dist:
#                 closest_dist = dist
#                 closest_index = j
#
#         if labels[i] != closest_index:
#             changed_count += 1
#
#         labels[i] = closest_index
#
#     return changed_count
#
#
# def get_barycenters_of_labels(data, labels, clusters):
#     barycenters = np.zeros(clusters.shape)
#     occurences = np.zeros(len(clusters), dtype=int)
#
#     for i, label in enumerate(labels):
#         barycenters[label] += data[i]
#         occurences[label] += 1
#
#     for i in range(len(barycenters)):
#         barycenters[i] /= occurences[i]
#
#     return barycenters


# Optimized by Claude
def set_labels_as_closest_cluster(data, labels, clusters):
    # Squared distances (n, K) using ||x - y||² = ||x||² - 2 x·y + ||y||²
    dists = (
        np.sum(data**2, axis=1)[:, None]
        - 2 * data @ clusters.T
        + np.sum(clusters**2, axis=1)[None, :]
    )

    new_labels = np.argmin(dists, axis=1)
    changed_count = np.count_nonzero(new_labels != labels)
    labels[:] = new_labels  # in-place, like your original

    return changed_count


# Optimized by Claude
def get_barycenters_of_labels(data, labels, clusters):
    K = len(clusters)

    counts = np.bincount(labels, minlength=K)  # (K,)
    one_hot = labels[:, None] == np.arange(K)[None, :]  # (n, K) boolean
    sums = one_hot.T @ data  # (K, d)

    # Empty clusters keep their previous center instead of producing NaN
    barycenters = clusters.astype(float).copy()
    nonempty = counts > 0
    barycenters[nonempty] = sums[nonempty] / counts[nonempty, None]

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


def k_means(data: np.ndarray, clusters_count: int):
    labels: np.ndarray = np.zeros(len(data), dtype=int)
    clusters: np.ndarray = data[
        np.random.choice(len(data), size=clusters_count, replace=False)
    ]  # y

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

    return labels, clusters


def k_means_2D_test(clusters_count):
    data: np.ndarray = np.loadtxt("classification_data/gmm2d.asc", dtype=float)  # x
    labels, clusters = k_means(data, clusters_count)
    show_classification(data, labels, clusters)


def k_means_n_images(count: int):
    images: list[Image.ImageFile.ImageFile] = []
    for number in np.random.randint(1, 112 + 1, count):
        while number == 14:
            number = np.random.randint(1, 112 + 1)  # there is not image D14.gif
        images.append(Image.open("classification_data/D" + str(number) + ".gif"))
        images[-1].resize((640, 640))
        images[-1].show()


if __name__ == "__main__":
    # img = Image.open("classification_data/D1.gif")
    # data = np.array(img)
    # img.show()
    # npimg = np.array(img)

    k_means_2D_test(3)
    # k_means_n_images(5)
