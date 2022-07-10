from tqdm import tqdm

from image_patcher.image_patcher import ImageAndSlidePatcher, ThyroidFragmentFilters


def calculate_acc_and_sensitivity(image_path, image_mask_path):
    zarr_loader_mask = ImageAndSlidePatcher._zarr_loader(image_mask_path)
    zarr_loader = ImageAndSlidePatcher._zarr_loader(image_path)
    frag_generator = ImageAndSlidePatcher._generate_raw_fragments_from_image_array_or_zarr(zarr_loader,
                                                                                           shuffle=False)

    frag_generator = tqdm(frag_generator, total=ImageAndSlidePatcher._get_number_of_initial_frags(zarr_loader))
    filter_func_list = [ThyroidFragmentFilters.empty_frag_with_laplacian_threshold]
    background_dict = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    frag_filtered = ImageAndSlidePatcher._filter_frag_from_generator(frag_generator, filter_func_list,
                                                                     return_all_with_condition=True)
    for next_test_item, frag_pos, condition in frag_filtered:
        frag_shape = next_test_item.shape
        mask_item = zarr_loader_mask[frag_pos[0]:frag_pos[0] + frag_shape[0], frag_pos[1]:frag_pos[1] + frag_shape[1]]
        if next_test_item is not None:
            mask_item = mask_item[:, :, 0]
            masked = mask_item.mean() > 80
            if condition and masked:
                background_dict["TP"] += 1
            elif condition and not masked:
                background_dict["FP"] += 1
            elif not condition and masked:
                background_dict["FN"] += 1
            elif not condition and not masked:
                background_dict["TN"] += 1
        else:
            break
    acc = (background_dict["TP"] + background_dict["TN"]) / sum(list(background_dict.values()))
    sensitivity = background_dict["TP"] / (background_dict["TP"] + background_dict["FP"])
    return acc, sensitivity, background_dict


if __name__ == '__main__':
    # threshold = 300
    image_lists = [("./575.tiff", "./575_mask.tiff"),
                   ("./687.tiff", "./687_mask.tiff"),
                   ("./1066_cropped.tif", "./1066_cropped_masked.tif"),
                   ]
    for img_path, img_mask_path in image_lists:
        acc, sen, background_dict = calculate_acc_and_sensitivity(img_path, img_mask_path)
        print()
        print(img_path + ":", end=" ")
        print(f"acc: {acc}, sen: {sen}, table: {background_dict}")
        print()

# threshold=500
# ./1066_cropped.tif: acc: 0.97553816, sen: 0.9578333, table: {'TP': 2794, 'FP': 123, 'TN': 2191, 'FN': 2} 5110
# ./575.tiff: acc: 0.824, sen: 0.9982126, table: {'TP': 1117, 'FP': 2, 'TN': 325, 'FN': 306} 1750
# ./687.tiff: acc: 0.69666666, sen: 0.9039039, table: {'TP': 602, 'FP': 64, 'TN': 443, 'FN': 391} 1500

# threshold=800
# ("./575.tiff", "./575_mask.tiff"),  # {'TP': 1111, 'FP': 8, 'TN': 341, 'FN': 290}
# ("./687.tiff", "./687_mask.tiff"),  # {'TP': 582, 'FP': 84, 'TN': 485, 'FN': 349},

# threshold=300
# ./575.tiff: acc: 0.8817142857142857, sen: 0.9557721139430285, table: {'TP': 1275, 'FP': 59, 'TN': 268, 'FN': 148}
