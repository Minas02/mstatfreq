from math import inf

import numpy as np
from PIL import Image


def select_seed(img, lamda, pad):
    h, w = img.shape[:2]

    x = np.random.randint(pad + lamda, h - pad - lamda)
    y = np.random.randint(pad + lamda, w - pad - lamda)

    return img[x - lamda : x + lamda + 1, y - lamda : y + lamda + 1].copy()


def get_patches_matrix(padded_img, patch_size):
    h = padded_img.shape[0] - patch_size + 1
    w = padded_img.shape[1] - patch_size + 1
    C = padded_img.shape[2]

    patches_matrix = np.empty((h, w, patch_size * patch_size, C), dtype=np.float32)
    for di in range(patch_size):
        for dj in range(patch_size):
            patches_matrix[:, :, di * patch_size + dj] = padded_img[
                di : di + h, dj : dj + w
            ]
    patches_matrix /= 255

    return patches_matrix.reshape(h * w, patch_size * patch_size * C)


def get_patches_score(
    patches_matrix, patches_matrix_sq, filled_mask_to_check, patch_to_check
):
    C = patch_to_check.shape[2]
    mask_flat = np.repeat(filled_mask_to_check.flatten(), C).astype(np.float32)
    target = patch_to_check.flatten().astype(np.float32) / 255 * mask_flat

    scores = (
        patches_matrix_sq @ mask_flat - 2 * (patches_matrix @ target) + target @ target
    )

    return np.maximum(scores, 0) / mask_flat.sum()


def set_pixel_filled(filled_mask, nb_neighbours_count, x, y, half_patch_size):
    filled_mask[x, y] = 1
    nb_neighbours_count[
        x - half_patch_size : x + half_patch_size + 1,
        y - half_patch_size : y + half_patch_size + 1,
    ] += 1
    nb_neighbours_count[x, y] = -inf


def efros_leung(img, patch_size, lamda, output_size, eps=1.1, show_seed=False):
    half_patch_size = patch_size // 2
    pad = half_patch_size

    output_size_x = output_size[0]
    output_size_y = output_size[1]

    gray = img.ndim == 2
    if gray:
        img = img[..., None]
    C = img.shape[2]

    img = np.pad(
        img, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=0
    )
    output = np.zeros(
        (output_size_x + 2 * pad, output_size_y + 2 * pad, C), dtype=np.uint8
    )

    patches_matrix = get_patches_matrix(img, patch_size)
    patches_matrix_sq = patches_matrix**2
    centers = img[pad:-pad, pad:-pad].reshape(-1, C)

    filled_mask = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)
    nb_neighbours_count = np.full(
        (output.shape[0], output.shape[1]), -inf, dtype=np.float32
    )
    nb_neighbours_count[pad:-pad, pad:-pad] = 0

    seed = select_seed(img, lamda, pad)
    if show_seed:
        Image.fromarray(seed[..., 0] if gray else seed).show()

    output_seed_x = np.random.randint(pad + lamda, output.shape[0] - pad - lamda)
    output_seed_y = np.random.randint(pad + lamda, output.shape[1] - pad - lamda)
    output[
        output_seed_x - lamda : output_seed_x + lamda + 1,
        output_seed_y - lamda : output_seed_y + lamda + 1,
    ] = seed

    for i in range(output_seed_x - lamda, output_seed_x + lamda + 1):
        for j in range(output_seed_y - lamda, output_seed_y + lamda + 1):
            set_pixel_filled(filled_mask, nb_neighbours_count, i, j, half_patch_size)

    nb_pixels_to_fill = output_size_x * output_size_y - (2 * lamda + 1) ** 2

    while nb_pixels_to_fill > 0:

        next_pos_to_set_x, next_pos_to_set_y = np.unravel_index(
            nb_neighbours_count.argmax(), nb_neighbours_count.shape
        )
        x0, x1 = (
            next_pos_to_set_x - half_patch_size,
            next_pos_to_set_x + half_patch_size + 1,
        )
        y0, y1 = (
            next_pos_to_set_y - half_patch_size,
            next_pos_to_set_y + half_patch_size + 1,
        )

        patch_to_check = output[x0:x1, y0:y1]
        filled_mask_to_check = filled_mask[x0:x1, y0:y1]

        patches_score = get_patches_score(
            patches_matrix, patches_matrix_sq, filled_mask_to_check, patch_to_check
        )

        min_score = patches_score.min()
        candidates = np.flatnonzero(patches_score <= min_score * eps + 1e-5)
        chosen_index = np.random.choice(candidates)

        output[next_pos_to_set_x, next_pos_to_set_y] = centers[chosen_index]
        set_pixel_filled(
            filled_mask,
            nb_neighbours_count,
            next_pos_to_set_x,
            next_pos_to_set_y,
            half_patch_size,
        )
        nb_pixels_to_fill -= 1

    output = output[pad:-pad, pad:-pad]
    return output[..., 0] if gray else output


if __name__ == "__main__":
    for i in range(8):
        img = Image.open(f"textures_data/text{i}.png")
        npimg = np.array(img)

        efros_leung_img = Image.fromarray(np.uint8(npimg))
        # efros_leung_img.show()

        print(npimg.shape)
        generated_img = efros_leung(
            npimg, patch_size=25, lamda=15, output_size=(256, 256), eps=1.01
        )

        generated_img = Image.fromarray(np.uint8(generated_img))
        generated_img.show()
        generated_img.save(f"textures_data/generated_text{i}.png")
