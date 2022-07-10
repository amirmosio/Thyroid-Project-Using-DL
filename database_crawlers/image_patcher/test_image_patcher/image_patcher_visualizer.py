import math

import cv2
import numpy as np
from tqdm import tqdm

from image_patcher.image_patcher import ImageAndSlidePatcher, ThyroidFragmentFilters


def imul(a, b):
    return math.ceil(a * b)


def save_patcher_mask(image_path, max_w_h_size=2000):
    zarr_loader = ImageAndSlidePatcher._zarr_loader(image_path)
    zarr_shape = zarr_loader.shape

    scale = max_w_h_size / max(zarr_shape)
    scaled_zarr_shape = (imul(zarr_shape[0], scale) + 5, imul(zarr_shape[1], scale) + 5, 3)
    scaled_masked_image = np.zeros(scaled_zarr_shape)
    frag_generator = ImageAndSlidePatcher._generate_raw_fragments_from_image_array_or_zarr(zarr_loader,
                                                                                           shuffle=False)

    frag_generator = tqdm(frag_generator, total=ImageAndSlidePatcher._get_number_of_initial_frags(zarr_loader))
    filter_func_list = [ThyroidFragmentFilters.empty_frag_with_laplacian_threshold]
    frag_filtered = ImageAndSlidePatcher._filter_frag_from_generator(frag_generator, filter_func_list,
                                                                     return_all_with_condition=True)
    for fragment, frag_pos, c in frag_filtered:
        fragment_size = fragment.shape
        scaled_frag_size = (imul(fragment_size[0], scale), imul(fragment_size[1], scale))
        scaled_frag = cv2.resize(fragment[:, :, :3], dsize=scaled_frag_size, interpolation=cv2.INTER_CUBIC)
        scaled_frag_size = scaled_frag.shape
        if not c:
            # background patches get dark
            scaled_frag = (scaled_frag * 0.3).astype(np.int8)
        scaled_pos = list((imul(frag_pos[i], scale) for i in range(2)))
        try:
            scaled_masked_image[scaled_pos[0]:scaled_pos[0] + scaled_frag_size[0],
            scaled_pos[1]:scaled_pos[1] + scaled_frag_size[1]] = scaled_frag
        except Exception as e:
            print(e)
    masked_image_path = ".".join(image_path.split(".")[:-1]) + "_generated_mask.jpg"
    cv2.imwrite(masked_image_path, scaled_masked_image)


if __name__ == '__main__':
    # threshold = 500
    image_lists = [
        ("./575.tiff", "./575_mask.tiff"),
        ("./687.tiff", "./687_mask.tiff"),
        ("./1066_cropped.tif", "./1066_cropped_masked.tif"),
    ]
    for img_path, img_mask_path in image_lists:
        save_patcher_mask(img_path)
