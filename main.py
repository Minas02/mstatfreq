from math import inf

import numpy as np
from PIL import Image


def select_seed(img, lamda, pad):
    seed_pos = np.random.randint(pad + lamda, pad + min(int(img.shape[0]), img.shape[1]) - lamda)
    if img.ndim == 2:
        return img[seed_pos - lamda: seed_pos + lamda + 1, seed_pos - lamda: seed_pos + lamda + 1]
    else:
        return img[seed_pos - lamda: seed_pos + lamda + 1, seed_pos - lamda: seed_pos + lamda + 1, :]


def img_diff(img1, img2):
    assert img1.shape == img2.shape
    img = np.subtract(img1.astype(float), img2.astype(float))
    total = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img.ndim == 2:
                total += img[i, j] ** 2
            else:
                total += img[i, j][0] ** 2 + img[i, j][1] ** 2 + img[i][j][2] ** 2
    return total


def get_patches_matrix(padded_img, patch_size):
    half_patch_size = patch_size // 2
    patches_matrix = []

    h = padded_img.shape[0] - patch_size
    w = padded_img.shape[1] - patch_size

    for i in range(half_patch_size, padded_img.shape[0] - half_patch_size):
        for j in range(half_patch_size, padded_img.shape[1] - half_patch_size):
            patches_matrix.append(padded_img[i - half_patch_size: i + half_patch_size + 1,
            j - half_patch_size: j + half_patch_size + 1].flatten())
    return np.asarray(patches_matrix)


def get_patches_score(patches_matrix, filled_mask_to_check, patch_to_check):
    patch_to_check_flat = patch_to_check.flatten()
    if patch_to_check.ndim == 2:
        mask_flat = filled_mask_to_check.flatten()
    else:
        C = patch_to_check.shape[2]
        mask_flat = np.repeat(filled_mask_to_check.flatten(), C)

    diff = (patches_matrix - patch_to_check_flat)
    scores = (diff * mask_flat).sum(axis=1)

    return scores


def efros_leung(img, patch_size, lamda, output_size, eps=0):
    # img[m:n, m:n] recup le care de m a n-1

    half_patch_size = patch_size // 2
    pad = half_patch_size

    output_size_x = output_size[0]
    output_size_y = output_size[1]

    if img.ndim == 2:
        img = np.pad(img, half_patch_size, mode='constant', constant_values=0)
        output = np.full(shape=(output_size_x + patch_size, output_size_y + patch_size, 1), fill_value=0,
                         dtype=np.uint8)
    else:
        img = np.pad(img, ((half_patch_size, half_patch_size), (half_patch_size, half_patch_size), (0, 0)),
                     mode='constant',
                     constant_values=0)
        output = np.full(shape=(output_size_x + patch_size, output_size_y + patch_size, 3), fill_value=(0, 0, 0),
                         dtype=np.uint8)

    patches_matrix = get_patches_matrix(img, patch_size)

    filled_mask = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)
    nb_neighbours_count = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)

    seed = select_seed(img, lamda, pad)

    output_seed_pos = np.random.randint(lamda, min(int(output.shape[0]), output.shape[1]) - lamda)
    output[
        output_seed_pos - lamda: output_seed_pos + lamda + 1, output_seed_pos - lamda: output_seed_pos + lamda + 1] = seed

    filled_mask[
        output_seed_pos - lamda: output_seed_pos + lamda + 1, output_seed_pos - lamda: output_seed_pos + lamda + 1] = 1

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            if not ((output_seed_pos - lamda < i < output_seed_pos + lamda + 1) and (
                    output_seed_pos - lamda < j < output_seed_pos + lamda + 1)):
                nb_neighbours_count[i, j] = filled_mask[i - half_patch_size: i + half_patch_size + 1,
                j - half_patch_size: j + half_patch_size + 1].sum()

    # remplir le filled mask avec un 1 qd le output est rempli et mettre a jour neighbours

    while filled_mask[pad: -pad, pad:-pad].min() < 1:

        possible_candidates = np.full(nb_neighbours_count.shape, -inf)
        possible_candidates[pad:-pad, pad:-pad] = nb_neighbours_count[pad:-pad, pad:-pad]
        possible_candidates = np.where(filled_mask == 1, -inf, possible_candidates)
        next_pos_to_set_x, next_pos_to_set_y = np.unravel_index(possible_candidates.argmax(), possible_candidates.shape)
        x0, x1 = next_pos_to_set_x - half_patch_size, next_pos_to_set_x + half_patch_size + 1
        y0, y1 = next_pos_to_set_y - half_patch_size, next_pos_to_set_y + half_patch_size + 1

        patch_to_check = output[x0: x1, y0: y1]
        filled_mask_to_check = filled_mask[x0: x1, y0: y1]

        patches_score = get_patches_score(patches_matrix, filled_mask_to_check, patch_to_check)

        min_score = patches_score.min()
        candidates = np.flatnonzero(patches_score <= min_score + eps)
        chosen_index = np.random.choice(candidates)

        chosen_patch = patches_matrix[chosen_index].reshape(patch_to_check.shape)
        output[next_pos_to_set_x, next_pos_to_set_y] = chosen_patch[half_patch_size, half_patch_size]
        filled_mask[next_pos_to_set_x, next_pos_to_set_y] = 1
        nb_neighbours_count[x0:x1, y0:y1] += 1

    return output


if __name__ == "__main__":
    img = Image.open("textures_data/text2.png")
    npimg = np.array(img)

    efros_leung_img = Image.fromarray(np.uint8(npimg))
    efros_leung_img.show()

    print(npimg.shape)
    generated_img = efros_leung(npimg, patch_size=13, lamda=4, output_size=(64, 64), eps=5)

    generated_img = Image.fromarray(np.uint8(generated_img))
    generated_img.show()


