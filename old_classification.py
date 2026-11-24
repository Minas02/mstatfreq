# Here is our original code for the k-means classification
# We asked Claude to optimize the code using numpy functions in order to reduce the execution time
# Here is the exact prompt "Can you rewrite the following functions using numpy to optimize them"
# followed by the 3 following functions:

import numpy as np


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
