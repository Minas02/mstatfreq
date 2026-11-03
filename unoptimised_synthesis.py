from concurrent.futures import ProcessPoolExecutor, as_completed
from math import inf

import numpy as np
from PIL import Image


def select_seed(img, lamda, pad):

    h, w = img.shape[:2]

    x = np.random.randint(
        pad + lamda,
        h - pad - lamda
    )

    y = np.random.randint(
        pad + lamda,
        w - pad - lamda
    )

    return img[
        x - lamda:x + lamda + 1,
        y - lamda:y + lamda + 1
    ].copy()


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
    patch_to_check_flat = patch_to_check.flatten().astype(np.float32)
    if patch_to_check.ndim == 2:
        mask_flat = filled_mask_to_check.flatten()
    else:
        C = patch_to_check.shape[2]
        mask_flat = np.repeat(filled_mask_to_check.flatten(), C)

    diff = patches_matrix - patch_to_check_flat

    scores = ((diff ** 2) * mask_flat).sum(axis=1) / mask_flat.sum()

    return scores


def efros_leung(img, patch_size, lamda, output_size, eps=1.1):
    # img[m:n, m:n] recup le care de m a n-1

    half_patch_size = patch_size // 2
    pad = half_patch_size

    output_size_x = output_size[0]
    output_size_y = output_size[1]

    if img.ndim == 2:
        img = np.pad(img, half_patch_size, mode='constant', constant_values=0)
        output = np.full(shape=(output_size_x + 2 * pad, output_size_y + 2 * pad, 1), fill_value=0,
                         dtype=np.uint8)
    else:
        img = np.pad(img, ((half_patch_size, half_patch_size), (half_patch_size, half_patch_size), (0, 0)),
                     mode='constant',
                     constant_values=0)
        output = np.full(shape=(output_size_x + 2 * pad, output_size_y + 2 * pad, 3), fill_value=(0, 0, 0),
                         dtype=np.uint8)

    patches_matrix = get_patches_matrix(img, patch_size).astype(np.float32)

    filled_mask = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)
    nb_neighbours_count = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint32)

    seed = select_seed(img, lamda, pad)
    seed_img = Image.fromarray(np.uint8(seed))
    seed_img.show()

    output_seed_pos = np.random.randint(lamda + half_patch_size, min(int(output.shape[0]), output.shape[1]) - lamda - half_patch_size)
    output[
        output_seed_pos - lamda: output_seed_pos + lamda + 1, output_seed_pos - lamda: output_seed_pos + lamda + 1] = seed

    filled_mask[
        output_seed_pos - lamda: output_seed_pos + lamda + 1, output_seed_pos - lamda: output_seed_pos + lamda + 1] = 1

    for i in range(max(pad, output_seed_pos - lamda - pad), min(pad + output_size_x, output_seed_pos + lamda + pad + 1)):
        for j in range(max(pad, output_seed_pos - lamda - pad), min(pad + output_size_y, output_seed_pos + lamda + pad + 1)):
            nb_neighbours_count[i, j] = filled_mask[i - half_patch_size: i + half_patch_size + 1,
            j - half_patch_size: j + half_patch_size + 1].sum()

    # remplir le filled mask avec un 1 qd le output est rempli et mettre a jour neighbours
    while filled_mask[pad: -pad, pad:-pad].min() < 1:

        possible_candidates = np.full(nb_neighbours_count.shape, -inf)
        possible_candidates[pad:-pad, pad:-pad] = nb_neighbours_count[pad:-pad, pad:-pad]
        possible_candidates = np.where(filled_mask == 1, -inf, possible_candidates)
        next_pos_to_set_x, next_pos_to_set_y = np.unravel_index(possible_candidates.argmax(),
                                                                possible_candidates.shape)
        x0, x1 = next_pos_to_set_x - half_patch_size, next_pos_to_set_x + half_patch_size + 1
        y0, y1 = next_pos_to_set_y - half_patch_size, next_pos_to_set_y + half_patch_size + 1

        patch_to_check = output[x0: x1, y0: y1]
        filled_mask_to_check = filled_mask[x0: x1, y0: y1]

        patches_score = get_patches_score(patches_matrix, filled_mask_to_check, patch_to_check)

        min_score = patches_score.min()
        candidates = np.flatnonzero(patches_score <= min_score * eps)
        chosen_index = np.random.choice(candidates)

        chosen_patch = patches_matrix[chosen_index].reshape(patch_to_check.shape)
        output[next_pos_to_set_x, next_pos_to_set_y] = chosen_patch[half_patch_size, half_patch_size]
        filled_mask[next_pos_to_set_x, next_pos_to_set_y] = 1
        update_x0 = max(pad, next_pos_to_set_x - half_patch_size)
        update_x1 = min(output.shape[0] - pad, next_pos_to_set_x + half_patch_size + 1)
        update_y0 = max(pad, next_pos_to_set_y - half_patch_size)
        update_y1 = min(output.shape[1] - pad, next_pos_to_set_y + half_patch_size + 1)

        for ui in range(update_x0, update_x1):
            for uj in range(update_y0, update_y1):
                if filled_mask[ui, uj] == 0:
                    nb_neighbours_count[ui, uj] = filled_mask[ui - half_patch_size: ui + half_patch_size + 1,
                    uj - half_patch_size: uj + half_patch_size + 1].sum()

    return output[pad: -pad, pad: -pad]

def generate_texture(i):
    # Charger l'image source
    img = Image.open(f"textures_data/text{i}.png")
    npimg = np.array(img)

    # Générer la texture
    generated_img = efros_leung(
        npimg,
        patch_size=25,
        lamda=15,
        output_size=(256, 256),
        eps=1.1
    )

    # Sauvegarder le résultat
    generated_img = Image.fromarray(np.uint8(generated_img))
    generated_img.save(f"textures_data/generated_text{i}.png")

    return i

if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=8) as executor:

        futures = [
            executor.submit(generate_texture, i)
            for i in range(8)
        ]

        for future in as_completed(futures):
            i = future.result()
            print(f"Texture {i} terminée")

# if __name__ == "__main__":
#     img = Image.open("textures_data/text2.png")
#     npimg = np.array(img)
#
#     efros_leung_img = Image.fromarray(np.uint8(npimg))
#     efros_leung_img.show()
#
#     print(npimg.shape)
#     generated_img = efros_leung(npimg, patch_size=25, lamda=15, output_size=(128, 128), eps=1.01)
#
#     generated_img = Image.fromarray(np.uint8(generated_img))
#     generated_img.show()