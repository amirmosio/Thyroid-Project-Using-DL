import concurrent.futures
import math

import cv2
import numpy as np

from config import Config
from database_crawlers.image_patcher.image_patcher import ImageAndSlidePatcher, ThyroidFragmentFilters
from utils import check_if_generator_is_empty


def imul(a, b):
    return math.ceil(a * b)


def calculate_acc_and_sensitivity(image_path, zarr_loader_mask, zarr_loader, frag_generator, scaled_masked_image,
                                  generated_mask_scale, laplacian_threshold, slide_patch_size=500):
    def process_frag(args):
        next_test_item, frag_pos, condition = args
        frag_shape = next_test_item.shape
        mask_scaled_frag_shape = list((imul(frag_shape[i], mask_scale) for i in range(2)))

        mask_frag_pos = list((imul(frag_pos[i], mask_scale) for i in range(2)))
        mask_w1, mask_w2 = mask_frag_pos[0], mask_frag_pos[0] + mask_scaled_frag_shape[0]
        mask_h1, mask_h2 = mask_frag_pos[1], mask_frag_pos[1] + mask_scaled_frag_shape[1]
        mask_item = zarr_loader_mask[mask_w1:mask_w2, mask_h1:mask_h2]
        mask_item = cv2.resize(mask_item, dsize=(0, 0), fx=1 / mask_scale, fy=1 / mask_scale)

        fragment_size = next_test_item.shape
        scaled_frag_size = (imul(fragment_size[0], generated_mask_scale), imul(fragment_size[1], generated_mask_scale))
        scaled_frag = cv2.resize(next_test_item[:, :, :3], dsize=scaled_frag_size, interpolation=cv2.INTER_CUBIC)
        scaled_frag_size = scaled_frag.shape

        if next_test_item is not None:
            mask_item = mask_item[:, :, 0]
            masked = mask_item.mean() > 256 * .3
            if condition and masked:
                background_dict["TP"] += 1
            elif condition and not masked:
                background_dict["FP"] += 1
            elif not condition and masked:
                background_dict["FN"] += 1
                # show_and_wait(next_test_item)
                # show_and_wait(mask_item)
            elif not condition and not masked:
                background_dict["TN"] += 1
        else:
            return None
        scaled_pos = list((imul(frag_pos[i], generated_mask_scale) for i in range(2)))
        try:
            mask_g_w1, mask_g_w2 = scaled_pos[0], scaled_pos[0] + scaled_frag_size[0]
            mask_g_h1, mask_g_h2 = scaled_pos[1], scaled_pos[1] + scaled_frag_size[1]
            scaled_masked_image[mask_g_w1:mask_g_w2, mask_g_h1:mask_g_h2] = scaled_frag
        except Exception as e:
            print(e)
        return True

    mask_scale = zarr_loader_mask.shape[0] / zarr_loader.shape[0]

    filter_func_list = [ThyroidFragmentFilters.func_laplacian_threshold_in_half_magnification(laplacian_threshold)]
    background_dict = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    total_frags = slide_patch_size if slide_patch_size else ImageAndSlidePatcher._get_number_of_initial_frags(
        zarr_loader)
    frag_filtered = ImageAndSlidePatcher._filter_frag_from_generator(frag_generator, filter_func_list,
                                                                     return_all_with_condition=True,
                                                                     all_frag_count=total_frags)
    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.workers) as executor:
        futures = []
        patch_count = 0
        for args in frag_filtered:
            patch_count += 1
            future_res = executor.submit(process_frag, args)
            futures.append(future_res)
            if len(futures) >= Config.workers or patch_count == slide_patch_size:
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                futures = []
                if patch_count == slide_patch_size:
                    break

    masked_image_path = ".".join(image_path.split(".")[:-1]) + "_generated_mask.jpg"
    cv2.imwrite(masked_image_path, scaled_masked_image)

    return background_dict


def update_and_find_best_threshold():
    threshold_jump_size = 100
    decay_const = 0.8

    threshold_jump_increase = 1

    threshold_score = None
    laplacian_threshold = 250

    def score_calculator(accuracy, specificity, acc_w=0.1):
        return accuracy * acc_w + specificity * (1 - acc_w)

    while sum([item is not None for item in zarr_loaders_and_generators]) >= 1:
        none_empty_generators = [i for i in range(len(zarr_loaders_and_generators)) if
                                 zarr_loaders_and_generators[i] is not None]
        whole_background_dict = {}
        for slide_pick in none_empty_generators:
            img_path = image_lists[slide_pick][1]
            zarr_loader_mask = zarr_loaders_and_generators[slide_pick][0]
            zarr_loader = zarr_loaders_and_generators[slide_pick][1]
            frag_generator = zarr_loaders_and_generators[slide_pick][2]

            generated_scaled_mask_image = zarr_loaders_and_generators[slide_pick][3]
            generated_mask_scale = zarr_loaders_and_generators[slide_pick][4]

            group_dict = calculate_acc_and_sensitivity(img_path,
                                                       zarr_loader_mask,
                                                       zarr_loader,
                                                       frag_generator,
                                                       generated_scaled_mask_image,
                                                       generated_mask_scale,
                                                       laplacian_threshold,
                                                       slide_patch_size=None if threshold_jump_size * decay_const ** 30 else 500)
            for i in range(len(zarr_loaders_and_generators)):
                if zarr_loaders_and_generators[i]:
                    generator = check_if_generator_is_empty(zarr_loaders_and_generators[i][2])
                    if generator:
                        zarr_loaders_and_generators[i][2] = generator
                    else:
                        zarr_loaders_and_generators[i] = None

            for key, value in group_dict.items():
                whole_background_dict[key] = whole_background_dict.get(key, 0) + value

        e = .000001
        acc = (whole_background_dict["TP"] + whole_background_dict["TN"]) / (
                sum(list(whole_background_dict.values())) + e)
        spec = whole_background_dict["TP"] / (whole_background_dict["TP"] + whole_background_dict["FP"] + e)
        next_score = score_calculator(acc, spec)
        if threshold_score is None:
            threshold_score = next_score
        else:
            if next_score > threshold_score:
                threshold_score = next_score

                laplacian_threshold += threshold_jump_increase * threshold_jump_size

            elif next_score <= threshold_score:
                threshold_score = next_score

                threshold_jump_increase *= -1
                threshold_jump_size *= decay_const

                laplacian_threshold += threshold_jump_increase * threshold_jump_size

        print(f"acc:{acc},spec:{spec},table:{whole_background_dict}, threshold:{laplacian_threshold}")


if __name__ == '__main__':

    max_w_h_size = 2000

    image_lists = [
        (
            "./TCGA-BJ-A3F0-01A-01-TSA.728CE583-95BE-462B-AFDF-FC0B228DF3DE__3_masked.tiff",
            "./TCGA-BJ-A3F0-01A-01-TSA.728CE583-95BE-462B-AFDF-FC0B228DF3DE__3.svs"
        ),
        (
            "./TCGA-DJ-A1QG-01A-01-TSA.04c62c21-dd45-49ea-a74f-53822defe097__2000_masked.tiff",
            "./TCGA-DJ-A1QG-01A-01-TSA.04c62c21-dd45-49ea-a74f-53822defe097__2000.svs"
        ),
        (
            "./TCGA-EL-A3ZQ-01A-01-TS1.344610D2-AB50-41C6-916E-FF0F08940BF1__2000_masked.tiff",
            "./TCGA-EL-A3ZQ-01A-01-TS1.344610D2-AB50-41C6-916E-FF0F08940BF1__2000.svs"
        ),
        (
            "./TCGA-ET-A2N5-11B-01-TS1.E2266304-5727-433B-9356-816A3DA5BA57__1000_masked.tiff",
            "./TCGA-ET-A2N5-11B-01-TS1.E2266304-5727-433B-9356-816A3DA5BA57__1000.svs"
        ),
        (
            "./TCGA-ET-A39N-01A-01-TSA.C38FCE19-9558-4035-9F0B-AD05B9BE321D___198_masked.tiff",
            "./TCGA-ET-A39N-01A-01-TSA.C38FCE19-9558-4035-9F0B-AD05B9BE321D___198.svs"
        ),
        (
            "./TCGA-J8-A42S-01A-01-TSA.7B80CBEB-7B85-417E-AA0C-11C79DE40250__0_masked.tiff",
            "./TCGA-J8-A42S-01A-01-TSA.7B80CBEB-7B85-417E-AA0C-11C79DE40250__0.svs"
        ),
    ]
    zarr_loaders_and_generators = []
    for _img_mask_path, _img_path in image_lists:
        _zarr_loader_mask = ImageAndSlidePatcher._zarr_loader(_img_mask_path)
        _zarr_loader = ImageAndSlidePatcher._zarr_loader(_img_path)
        _frag_generator = ImageAndSlidePatcher._generate_raw_fragments_from_image_array_or_zarr(_zarr_loader,
                                                                                                shuffle=True)
        _zarr_shape = _zarr_loader.shape

        _generated_mask_scale = max_w_h_size / max(_zarr_shape)
        _scaled_zarr_shape = (
            imul(_zarr_shape[0], _generated_mask_scale) + 5, imul(_zarr_shape[1], _generated_mask_scale) + 5, 3)
        _scaled_masked_image = np.zeros(_scaled_zarr_shape)

        zarr_loaders_and_generators.append([
            _zarr_loader_mask, _zarr_loader, _frag_generator, _scaled_masked_image, _generated_mask_scale
        ])
    update_and_find_best_threshold()
