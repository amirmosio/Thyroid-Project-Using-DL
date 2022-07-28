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
                                  generated_mask_scale, laplacian_threshold, slide_patch_size):
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
        if not condition:
            # background patches get dark
            scaled_frag = (scaled_frag * 0.3).astype(np.int8)
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
    initial_threshold_jump_size_const = 100
    threshold_jump_size = initial_threshold_jump_size_const
    decay_const = 0.7
    decay_count = 0

    threshold_jump_increase = 1

    threshold_score = None
    # update after initial run
    laplacian_threshold = 250
    laplacian_threshold = 820

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
                                                       slide_patch_size=None if decay_count >= 30 else 500)
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
        total_preds = (sum(list(whole_background_dict.values())) + e)
        acc = (whole_background_dict["TP"] + whole_background_dict["TN"]) / total_preds
        positive_preds = (whole_background_dict["TP"] + whole_background_dict["FP"] + e)
        precision = whole_background_dict["TP"] / positive_preds
        next_score = score_calculator(acc, precision)
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
                decay_count += 1

        print(f"acc:{acc},precision:{precision},table:{whole_background_dict}, threshold:{laplacian_threshold}")


if __name__ == '__main__':

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
            "./TCGA-ET-A39N-01A-01-TSA.C38FCE19-9558-4035-9F0B-AD05B9BE321D___198_masked.tiff",
            "./TCGA-ET-A39N-01A-01-TSA.C38FCE19-9558-4035-9F0B-AD05B9BE321D___198.svs"
        ),
        (
            "./TCGA-J8-A42S-01A-01-TSA.7B80CBEB-7B85-417E-AA0C-11C79DE40250__0_masked.tiff",
            "./TCGA-J8-A42S-01A-01-TSA.7B80CBEB-7B85-417E-AA0C-11C79DE40250__0.svs"
        ),
        (
            "./TCGA-ET-A39O-01A-01-TSA.3829C900-7597-4EA9-AFC7-AA238221CE69_7000_masked.tiff",
            "./TCGA-ET-A39O-01A-01-TSA.3829C900-7597-4EA9-AFC7-AA238221CE69_7000.svs"
        ),
    ]
    zarr_loaders_and_generators = []
    for _img_mask_path, _img_path in image_lists:
        _zarr_loader_mask = ImageAndSlidePatcher._zarr_loader(_img_mask_path)
        _zarr_loader = ImageAndSlidePatcher._zarr_loader(_img_path)
        _frag_generator = ImageAndSlidePatcher._generate_raw_fragments_from_image_array_or_zarr(_zarr_loader,
                                                                                                shuffle=True)
        _zarr_shape = _zarr_loader.shape

        _generated_mask_scale = 10 / 512
        _scaled_zarr_shape = (
            imul(_zarr_shape[0], _generated_mask_scale) + 5, imul(_zarr_shape[1], _generated_mask_scale) + 5, 3)
        _scaled_masked_image = np.zeros(_scaled_zarr_shape)

        zarr_loaders_and_generators.append([
            _zarr_loader_mask, _zarr_loader, _frag_generator, _scaled_masked_image, _generated_mask_scale
        ])
    update_and_find_best_threshold()
# 500 last  for steps initial with 250 threshold
# acc:0.9935897404051611,spec:0.9999999861111113,table:{'TP': 72, 'FP': 0, 'TN': 238, 'FN': 2}, threshold:755.9870661958269

# 900 last  for steps initial with 755 threshold
# acc:0.9581481477932785,spec:0.9905437328828753,table:{'TP': 419, 'FP': 4, 'TN': 2168, 'FN': 109}, threshold:768.9005414399998
# acc:0.9462962959458162,spec:0.9892703841431966,table:{'TP': 461, 'FP': 5, 'TN': 2094, 'FN': 140}, threshold:760.3106068479998
# acc:0.9514814811290809,spec:0.9953379930178602,table:{'TP': 427, 'FP': 2, 'TN': 2142, 'FN': 129}, threshold:751.7206722559998
# acc:0.9514170035670966,spec:0.9851851815363512,table:{'TP': 266, 'FP': 4, 'TN': 1614, 'FN': 92}, threshold:758.5926199295998
# acc:0.963333332798148,spec:0.9999999957446809,table:{'TP': 235, 'FP': 0, 'TN': 1499, 'FN': 66}, threshold:765.4645676031997
# acc:0.9507042247941876,spec:0.9874999958854167,table:{'TP': 237, 'FP': 3, 'TN': 1383, 'FN': 81}, threshold:759.9670094643197
# acc:0.9833333322407407,spec:0.999999996,table:{'TP': 250, 'FP': 0, 'TN': 635, 'FN': 15}, threshold:754.4694513254398
# acc:0.9844444433506173,spec:0.9958333291840278,table:{'TP': 239, 'FP': 1, 'TN': 647, 'FN': 13}, threshold:758.8674978365437
# acc:0.9855555544604938,spec:0.988636359891529,table:{'TP': 261, 'FP': 3, 'TN': 626, 'FN': 10}, threshold:755.3490606276605
# acc:0.9888888877901234,spec:0.9999999956521739,table:{'TP': 230, 'FP': 0, 'TN': 660, 'FN': 10}, threshold:751.8306234187772
# acc:0.9733333322518518,spec:0.9864253349030528,table:{'TP': 218, 'FP': 3, 'TN': 658, 'FN': 21}, threshold:754.6453731858838
# acc:0.9922222211197531,spec:0.999999996031746,table:{'TP': 252, 'FP': 0, 'TN': 641, 'FN': 7}, threshold:757.4601229529904
# acc:0.9911111100098765,spec:0.9954954910112815,table:{'TP': 221, 'FP': 1, 'TN': 671, 'FN': 7}, threshold:755.2083231393052
# acc:0.9933333322296296,spec:0.9999999953703704,table:{'TP': 216, 'FP': 0, 'TN': 678, 'FN': 6}, threshold:752.95652332562




