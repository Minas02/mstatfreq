import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


def set_labels_as_closest_cluster(
    data: np.ndarray, labels: np.ndarray, clusters: np.ndarray
) -> np.signedinteger:
    dists = (
        np.sum(data**2, axis=1)[:, None]
        - 2 * data @ clusters.T
        + np.sum(clusters**2, axis=1)[None, :]
    )

    new_labels = np.argmin(dists, axis=1)
    changed_count = np.count_nonzero(new_labels != labels)
    labels[:] = new_labels

    return changed_count


def get_barycenters_of_labels(
    data: np.ndarray, labels: np.ndarray, clusters: np.ndarray
) -> np.ndarray:
    K = len(clusters)

    counts = np.bincount(labels, minlength=K)
    one_hot = labels[:, None] == np.arange(K)[None, :]
    sums = one_hot.T @ data

    barycenters = clusters.astype(float).copy()
    nonempty = counts > 0
    barycenters[nonempty] = sums[nonempty] / counts[nonempty, None]

    return barycenters


def k_means(
    data: np.ndarray, K: int, convergence_margin: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    clusters = data[np.random.choice(len(data), K, replace=False)].astype(float)
    labels = np.ones(len(data), dtype=int)

    updated_labels_count = set_labels_as_closest_cluster(data, labels, clusters)
    ite_count = 0
    while updated_labels_count > convergence_margin:
        clusters = get_barycenters_of_labels(data, labels, clusters)
        updated_labels_count = set_labels_as_closest_cluster(data, labels, clusters)
        ite_count += 1

    inertia = np.sum((data - clusters[labels]) ** 2)
    return labels, clusters, inertia, ite_count


def k_means_2D_test(clusters_count: int) -> None:
    data: np.ndarray = np.loadtxt("classification_data/gmm2d.asc", dtype=float)  # x
    labels, clusters, inertia, ite_count = k_means(data, clusters_count)
    print(f"K-Means finished in {ite_count} iterations with inertia of {inertia}")

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


def load_img_as_grayscale(img_number: int) -> np.ndarray:
    return np.asarray(
        Image.open(f"classification_data/D{img_number}.gif").convert("L"),
        dtype=np.float64,
    )


def cut_image_into_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    height, width = image.shape
    num_patch_y, num_patch_x = height // patch_size, width // patch_size

    return (
        image[: num_patch_y * patch_size, : num_patch_x * patch_size]
        .reshape(num_patch_y, patch_size, num_patch_x, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(num_patch_y * num_patch_x, patch_size, patch_size)
    )


def build_do_and_dt(
    img_numbers: list[int],
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images_patches = [
        cut_image_into_patches(load_img_as_grayscale(i), patch_size)
        for i in img_numbers
    ]  # chose 5 images that look different

    do_patches, do_labels = [], []
    dt_patches, dt_labels = [], []

    for label, image_patches in enumerate(images_patches):
        order = np.random.permutation(len(image_patches))
        half = len(image_patches) // 2

        do_patches.append(image_patches[order[:half]])
        do_labels.append(np.full(half, label))

        dt_patches.append(image_patches[order[half:]])
        dt_labels.append(np.full(len(image_patches) - half, label))

    return (
        np.concatenate(do_patches),
        np.concatenate(do_labels),
        np.concatenate(dt_patches),
        np.concatenate(dt_labels),
    )


def gabor_kernel(theta: float, sigma: float, lam: float) -> np.ndarray:
    radius = np.ceil(2.5 * sigma)
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(float)
    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = x * np.sin(theta) - y * np.cos(theta)
    cosine = np.cos(2 * np.pi * lam * xp / sigma)
    return np.exp(-(xp**2 + yp**2) / (2 * sigma**2)) * cosine


def gabor_descriptor(
    patches: np.ndarray,
    theta: int = 6,
    sigmas: list[float] = [2.0, 3.0, 4.0],
    lambdas: list[float] = [0.5],
) -> np.ndarray:
    _, H, W = patches.shape
    patches = patches - patches.mean(axis=(1, 2), keepdims=True)
    feats = []

    for sigma in sigmas:
        k = gabor_kernel(0.0, sigma, 1.0).shape[0]
        shape = (H + k - 1, W + k - 1)
        patches_fft = np.fft.rfft2(patches, s=shape)

        for lam in lambdas:
            for t in range(theta):
                kernel_fft = np.fft.rfft2(
                    gabor_kernel(np.pi * t / theta, sigma, lam), s=shape
                )
                inv_fft = np.abs(
                    np.fft.irfft2(patches_fft * kernel_fft, s=shape)[
                        :, k - 1 : H, k - 1 : W
                    ]
                )
                feats.append(inv_fft.mean(axis=(1, 2)))
                feats.append(inv_fft.std(axis=(1, 2)))

    return np.stack(feats, axis=1)


def fit_classifier(patches, images_labels, K, theta=6, init_count=10):
    features = gabor_descriptor(patches, theta)
    mean, std = features.mean(axis=0), features.std(axis=0) + 1e-12
    normalized_features = (features - mean) / std

    labels, clusters, _, _ = min(
        (k_means(normalized_features, K) for _ in range(init_count)), key=lambda r: r[2]
    )

    cluster_to_class = np.array(
        [
            (
                np.bincount(images_labels[labels == k]).argmax()
                if np.any(labels == k)
                else -1
            )
            for k in range(K)
        ]
    )

    return {
        "theta": theta,
        "mean": mean,
        "std": std,
        "clusters": clusters,
        "cluster_to_class": cluster_to_class,
    }


def predict(model, patches):
    normalized_features = (
        gabor_descriptor(patches, model["theta"]) - model["mean"]
    ) / model["std"]
    labels = -np.ones(len(normalized_features), dtype=int)
    set_labels_as_closest_cluster(normalized_features, labels, model["clusters"])
    return model["cluster_to_class"][labels]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def calculate_confusion_matrix(y_true, y_pred, classes_count):
    confusion_matrix = np.zeros((classes_count, classes_count), dtype=int)
    np.add.at(confusion_matrix, (y_true, y_pred), 1)

    return confusion_matrix


def add_noise(patches, sigma):
    if sigma <= 0:
        return patches

    return np.clip(patches + np.random.normal(0, sigma, patches.shape), 0, 255)


def blur_patches(patches, radius):
    if radius <= 0:
        return patches

    out = np.empty_like(patches)

    for i, patch in enumerate(patches):
        image = Image.fromarray(np.uint8(np.clip(np.round(patch), 0, 255)))
        out[i] = np.asarray(image.filter(ImageFilter.GaussianBlur(radius)), dtype=float)

    return out


if __name__ == "__main__":
    # k_means_2D_test(clusters_count=3)
    # pass

    img_numbers = [95, 75, 69, 46, 5]
    images = [load_img_as_grayscale(img) for img in img_numbers]
    fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
    for ax, image, number in zip(axes, images, img_numbers):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"D{number}")
        ax.axis("off")

    classes_count = len(img_numbers)

    for patch_size in [32, 64]:
        do_patches, do_labels, _, _ = build_do_and_dt(
            img_numbers, patch_size=patch_size
        )

        results = []

        for theta in [4, 6, 8, 10]:
            model = fit_classifier(do_patches, do_labels, classes_count, theta)
            accuracy_do = accuracy(do_labels, predict(model, do_patches))
            results.append((accuracy_do, patch_size, theta))
            print(f"patch_size {patch_size}, theta {theta}, accuracy {accuracy_do:.3f}")

    best_accuracy, patch_size, theta = max(results, key=lambda r: r[0])
    print()
    print(
        f"Chose patch_size {patch_size}, theta {theta:2d}, accuracy {best_accuracy:.3f}"
    )

    do_patches, do_labels, dt_patches, dt_labels = build_do_and_dt(
        img_numbers, patch_size
    )
    model = fit_classifier(do_patches, do_labels, classes_count, theta)

    # Verify on DT
    prediction = predict(model, dt_patches)
    confusion_matrix = calculate_confusion_matrix(dt_labels, prediction, classes_count)
    print(f"Precision on DT: {accuracy(dt_labels, prediction):.3f}")
    print("Confusion matrix:\n", confusion_matrix)

    plt.figure()
    plt.imshow(confusion_matrix)
    plt.colorbar()
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix")
    for i in range(classes_count):
        for j in range(classes_count):
            plt.text(j, i, confusion_matrix[i, j], ha="center", va="center")
    plt.savefig("confusion_DT.png", dpi=150)

    # Test robustness agains noise and blur
    sigmas = [0, 5, 10, 20, 30, 50]
    radiuses = [0, 0.5, 1, 1.5, 2, 3]
    accuracy_noise = [
        accuracy(dt_labels, predict(model, add_noise(dt_patches, s))) for s in sigmas
    ]
    accuracy_blur = [
        accuracy(dt_labels, predict(model, blur_patches(dt_patches, r)))
        for r in radiuses
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(sigmas, accuracy_noise, "o-")
    ax1.set_xlabel("Standard deviation of gaussian blue")
    ax1.set_ylabel("Precision on DT")
    ax1.set_title("Robustness to noise")

    ax2.plot(radiuses, accuracy_blur, "o-")
    ax2.set_xlabel("radius of gaussian blur (pixels)")
    ax2.set_title("Robustness to blur")

    plt.tight_layout()
    plt.savefig("robustness.png", dpi=150)
    plt.show()
