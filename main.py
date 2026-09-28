from math import floor

from PIL import Image
import numpy as np

def select_seed(img, lamda, pad):
    seed_pos = np.random.randint(pad + lamda, pad + min(int(img.shape[0]), img.shape[1]) - lamda)
    if img.ndim == 2:
        return img[seed_pos - lamda : seed_pos + lamda + 1, seed_pos - lamda : seed_pos + lamda + 1]
    else :
        return img[seed_pos - lamda : seed_pos + lamda + 1, seed_pos - lamda : seed_pos + lamda + 1, :]


def img_diff(img1, img2):
    assert img1.shape == img2.shape
    img = np.subtract(img1.astype(float), img2.astype(float))
    total = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img.ndim == 2:
                total += img[i, j]**2
            else:
                total += img[i, j][0]**2 + img[i, j][1]**2 + img[i][j][2]**2
    return total



def efros_leung(img, patch_size, lamda, output_size, eps=0):

    # img[m:n, m:n] recup le care de m a n-1
    output_size_x = output_size[0]
    output_size_y = output_size[1]
    padded_shape = (output_size_x + patch_size, output_size_y + patch_size)

    half_patch_size = patch_size // 2
    pad = half_patch_size


    filled_mask = np.zeros(padded_shape)
    nb_neighbours_mask = np.zeros(padded_shape)

    if img.ndim == 2:
        output = np.full(shape=(output_size_x + patch_size, output_size_y + patch_size, 1), fill_value=125, dtype=np.uint8)
    else :
        output = np.full(shape=(output_size_x+ patch_size, output_size_y+ patch_size, 3), fill_value=(125, 0, 125), dtype=np.uint8)

    seed = select_seed(img, lamda, pad)
    output_seed_pos = np.random.randint(lamda, min(int(output.shape[0]), output.shape[1]) - lamda)
    output[output_seed_pos - lamda : output_seed_pos + lamda + 1, output_seed_pos - lamda : output_seed_pos + lamda + 1] = seed

    filled_mask[output_seed_pos - lamda : output_seed_pos + lamda + 1, output_seed_pos - lamda : output_seed_pos + lamda + 1] = 1

    for i in range(output_size[0]):
        for j in range(output_size[1]):
            nb_neighbours_mask[i, j] = filled_mask[i - half_patch_size : i + half_patch_size + 1, j - half_patch_size : j + half_patch_size + 1].sum()

    # remplir le filled mask avec un 1 qd le output est rempli et mettre a jour neighbours

    while filled_mask.min() < 1:
        next_pos_to_set_x, next_pos_to_set_y = np.unravel_index(nb_neighbours_mask.argmax(), nb_neighbours_mask.shape)
        x0, x1 = max(0, next_pos_to_set_x - half_patch_size), min(output_size_x, next_pos_to_set_x + half_patch_size + 1)
        y0, y1 = max(0, next_pos_to_set_y - half_patch_size), min(output_size_y, next_pos_to_set_y + half_patch_size + 1)
        patch_to_check = output[x0 : x1, y0 : y1]
        filled_mask_to_check = filled_mask[x0 : x1, y0 : y1]

        for i in range (x0, x1):
            for j in range (y0, y1):








    output_img = Image.fromarray(np.uint8(output))
    output_img.show()

    filled_mask_img = Image.fromarray(np.uint8(filled_mask * 255))
    filled_mask_img.show()

    nb_neighbours_mask_img = Image.fromarray(np.uint8(nb_neighbours_mask))
    nb_neighbours_mask_img.show()
    return

if __name__ == "__main__":
    img = Image.open("textures_data/text1.png")
    npimg = np.array(img)
    imgtext = Image.fromarray(np.uint8(npimg))

    efros_leung(npimg, patch_size=15, lamda=4, output_size=(64, 64), eps=10)
    efros_leung_img  = Image.fromarray(np.uint8(npimg))
    efros_leung_img.show()