import csv
import json
import os
import os.path as os_path
import random
import re
from math import ceil
from os import listdir
from os.path import isfile, join

import cv2
import tifffile
import zarr as ZarrObject
from tqdm import tqdm

from utils import show_and_wait
from database_crawlers.web_stain_sample import ThyroidCancerLevel, WebStainImage


class ThyroidFragmentFilters:
    @staticmethod
    def func_laplacian_threshold_in_half_magnification(threshold=500, rescale=.7):
        def wrapper(image_nd_array):
            res, var = ThyroidFragmentFilters.empty_frag_with_laplacian_threshold(image_nd_array, threshold,
                                                                                  return_variance=True)
            if res or var * (1 / rescale) ** 2 < threshold:
                return res
            image_nd_array = cv2.resize(image_nd_array, dsize=(0, 0), fx=rescale, fy=rescale)
            res = ThyroidFragmentFilters.empty_frag_with_laplacian_threshold(image_nd_array, threshold)
            return res

        return wrapper

    @staticmethod
    def empty_frag_with_laplacian_threshold(image_nd_array, threshold=500, return_variance=False):
        gray = cv2.cvtColor(image_nd_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3, )
        std = cv2.meanStdDev(laplacian)[1][0][0]

        variance = std ** 2
        if return_variance:
            return variance >= threshold, variance
        return variance >= threshold


class ImageAndSlidePatcher:
    @classmethod
    def _check_magnification_from_description(cls, tiff_address):
        try:
            tif_file_obj = tifffile.TiffFile(tiff_address)
            image_description = tif_file_obj.pages.keyframe.tags["ImageDescription"].value
            app_mag = int(re.findall("(AppMag = [0-9]+)", image_description)[0].split(" = ")[-1])
            return app_mag
        except Exception as e:
            return None

    @classmethod
    def _zarr_loader(cls, tiff_address, key=0):
        image_zarr = tifffile.imread(tiff_address, aszarr=True, key=key, )
        zarr = ZarrObject.open(image_zarr, mode='r')
        return zarr

    @classmethod
    def _jpeg_loader(cls, jpeg_address):
        im = cv2.imread(jpeg_address)
        return im

    @classmethod
    def _json_key_loader(cls, json_file_address, key=None):
        with open(json_file_address, 'rb') as file:
            json_dict = json.loads(file.read())
        if key:
            return json_dict[key]
        return json_dict

    @classmethod
    def _get_extension_from_path(cls, file_path):
        return os_path.splitext(file_path)[-1]

    @classmethod
    def _get_file_name_from_path(cls, file_path):
        return ".".join(os_path.split(file_path)[-1].split(".")[:-1])

    @classmethod
    def _get_number_of_initial_frags(cls, zarr_object, frag_size=512, frag_overlap=0.1):
        zarr_shape = zarr_object.shape

        step_size = int(frag_size * (1 - frag_overlap))
        overlap_size = frag_size - step_size
        w_range = list(range(0, ceil((zarr_shape[0] - overlap_size) / step_size) * step_size, step_size))
        h_range = list(range(0, ceil((zarr_shape[1] - overlap_size) / step_size) * step_size, step_size))
        return len(w_range) * len(h_range)

    @classmethod
    def _generate_raw_fragments_from_image_array_or_zarr(cls, image_object, frag_size=512, frag_overlap=0.1,
                                                         shuffle=True):
        if image_object is None:
            return None
        zarr_shape = image_object.shape

        step_size = int(frag_size * (1 - frag_overlap))
        overlap_size = frag_size - step_size
        w_range = list(range(0, ceil((zarr_shape[0] - overlap_size) / step_size) * step_size, step_size))

        h_range = list(range(0, ceil((zarr_shape[1] - overlap_size) / step_size) * step_size, step_size))

        if shuffle:
            random.shuffle(w_range)
            random.shuffle(h_range)
        for w in w_range:
            for h in h_range:
                end_w, end_h = min(zarr_shape[0], w + frag_size), min(zarr_shape[1], h + frag_size)
                start_w, start_h = end_w - frag_size, end_h - frag_size
                yield image_object[start_w:end_w, start_h: end_h], (start_w, start_h)

    @classmethod
    def _filter_frag_from_generator(cls, frag_generator, filter_func_list, return_all_with_condition=False,
                                    all_frag_count=None, output_file=None):
        for next_test_item, frag_pos in tqdm(frag_generator, total=all_frag_count, file=output_file,
                                             postfix="Filtering", position=0):
            condition = True
            for function in filter_func_list:
                condition &= function(next_test_item)
            if return_all_with_condition:
                yield next_test_item, frag_pos, condition
            elif condition:
                # show_and_wait(frag)
                yield next_test_item, frag_pos

    @classmethod
    def _get_json_and_image_address_of_directory(cls, directory_path, ignore_json=False):
        image_formats = [".jpeg", ".tiff", ".jpg"]
        json_format = ".json"
        files = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
        files.sort()
        pairs = {}
        for file_path in files:
            file_path = join(directory_path, file_path)
            file_name = cls._get_file_name_from_path(file_path)
            pairs[file_name] = pairs.get(file_name, [None, None])
            if cls._get_extension_from_path(file_path) in image_formats:
                pairs[file_name][1] = file_path
            elif cls._get_extension_from_path(file_path) == json_format:
                pairs[file_name][0] = file_path
        if ignore_json:
            return [value for key, value in pairs.values() if value is not None]
        return [(key, value) for key, value in pairs.values() if key is not None and value is not None]

    @staticmethod
    def create_patch_dir_and_initialize_csv(database_path):
        data_dir = os.path.join(database_path, "data")
        patch_dir = os.path.join(database_path, "patches")
        if not os.path.isdir(patch_dir):
            os.mkdir(patch_dir)
        label_csv_path = os.path.join(patch_dir, "patch_labels.csv")
        csv_file = open(label_csv_path, "a+")
        csv_writer = csv.writer(csv_file)
        csv_file.seek(0)
        if len(csv_file.read(100)) <= 0:
            csv_writer.writerow(WebStainImage.sorted_json_keys())
        return data_dir, patch_dir, csv_writer, csv_file

    @classmethod
    def save_image_patches_and_update_csv(cls, thyroid_type, thyroid_desired_classes, csv_writer, web_details,
                                          image_path, slide_patch_dir, slide_id):
        if thyroid_desired_classes is not None and thyroid_type not in thyroid_desired_classes:
            return
        csv_writer.writerow(list(web_details.values()))

        if cls._get_extension_from_path(image_path) in [".tiff", ".tif", ".svs"]:
            zarr_object = cls._zarr_loader(image_path)
            generator = cls._generate_raw_fragments_from_image_array_or_zarr(zarr_object)
            total_counts = cls._get_number_of_initial_frags(zarr_object=zarr_object)
        else:
            jpeg_image = cls._jpeg_loader(image_path)
            jpeg_image = cls.ask_image_scale_and_rescale(jpeg_image)
            generator = cls._generate_raw_fragments_from_image_array_or_zarr(jpeg_image)
            total_counts = cls._get_number_of_initial_frags(zarr_object=jpeg_image)
        if generator is None:
            return

        if not os.path.isdir(slide_patch_dir):
            os.mkdir(slide_patch_dir)
        filters = [ThyroidFragmentFilters.empty_frag_with_laplacian_threshold]
        fragment_id = 0
        slide_progress_file_path = os.path.join(slide_patch_dir, "progress.txt")
        with open(slide_progress_file_path, "w") as file:
            for fragment, frag_pos in cls._filter_frag_from_generator(generator, filters, all_frag_count=total_counts,
                                                                      output_file=file):
                fragment_file_path = os.path.join(slide_patch_dir, f"{slide_id}-{fragment_id}.jpeg")
                cv2.imwrite(fragment_file_path, fragment)
                fragment_id += 1
        return fragment_id, total_counts

    @classmethod
    def save_patches_in_folders(cls, database_directory, dataset_dir=None):
        thyroid_desired_classes = [ThyroidCancerLevel.MALIGNANT, ThyroidCancerLevel.BENIGN]
        datasets_dirs = os.listdir(database_directory) if dataset_dir is None else dataset_dir
        list_dir = [os.path.join(database_directory, o) for o in datasets_dirs
                    if os.path.isdir(os.path.join(database_directory, o, "data"))]
        for database_path in list_dir:
            print("database path: ", database_path)
            data_dir, patch_dir, csv_writer, csv_file = cls.create_patch_dir_and_initialize_csv(database_path)
            for json_path, image_path in cls._get_json_and_image_address_of_directory(data_dir):
                print("image path: ", image_path)
                file_name = cls._get_file_name_from_path(image_path)
                slide_id = str(hash(file_name))
                slide_patch_dir = os.path.join(patch_dir, slide_id)
                if os.path.isdir(slide_patch_dir):
                    """
                    it has already been patched
                    """
                    continue

                web_details = cls._json_key_loader(json_path)
                web_details["image_id"] = slide_id
                web_label = web_details["image_web_label"]
                thyroid_type = ThyroidCancerLevel.get_thyroid_level_from_diagnosis_label(web_label)
                web_details["image_class_label"] = thyroid_type.value[1]

                cls.save_image_patches_and_update_csv(thyroid_type, thyroid_desired_classes, csv_writer, web_details,
                                                      image_path, slide_patch_dir, slide_id)
            csv_file.close()

    @classmethod
    def save_papsociaty_patch(cls, database_path):
        thyroid_desired_classes = [ThyroidCancerLevel.MALIGNANT, ThyroidCancerLevel.BENIGN]
        print("database path: ", database_path)
        for folder in ["BENIGN", "MALIGNANT"]:
            group_path = os.path.join(database_path, "data", folder)
            data_dir, patch_dir, csv_writer, csv_file = cls.create_patch_dir_and_initialize_csv(database_path)
            for image_path in cls._get_json_and_image_address_of_directory(group_path, ignore_json=True):
                print("image path: ", image_path)
                file_name = cls._get_file_name_from_path(image_path)
                slide_id = str(hash(file_name))
                slide_patch_dir = os.path.join(patch_dir, slide_id)
                if os.path.isdir(slide_patch_dir):
                    """
                    it has already been patched
                    """
                    continue
                web_label = folder + "-" + file_name
                thyroid_type = ThyroidCancerLevel.get_thyroid_level_from_diagnosis_label(web_label)
                web_details = {"database_name": "PapSociety",
                               "image_id": slide_id,
                               "image_web_label": web_label,
                               "image_class_label": thyroid_type.value[1],
                               "report": None,
                               "stain_type": "UNKNOWN",
                               "is_wsi": False}
                cls.save_image_patches_and_update_csv(thyroid_type, thyroid_desired_classes, csv_writer, web_details,
                                                      image_path, slide_patch_dir, slide_id)

            csv_file.close()

    @classmethod
    def ask_image_scale_and_rescale(cls, image):
        # small: S, Medium: M, Large:L
        show_and_wait(image)
        res = input("how much plus pointer fill a cell(float, i:ignore, else repeat): ")
        try:
            if res == "i":
                return None
            elif re.match("[0-9]+(.[0-9]*)?", res):
                scale = 1 / float(res)
                return cv2.resize(image, (0, 0), fx=scale, fy=scale)
            else:
                return cls.ask_image_scale_and_rescale(image)
        except Exception as e:
            print(e)
            return cls.ask_image_scale_and_rescale(image)


if __name__ == '__main__':
    random.seed(1)

    database_directory = "./"
    # ImageAndSlidePatcher.save_patches_in_folders(database_directory, dataset_dir=["stanford_tissue_microarray"])
    # ImageAndSlidePatcher.save_papsociaty_patch(os.path.join(database_directory, "papsociaty"))
    ImageAndSlidePatcher.save_national_cancer_institute_patch(
        os.path.join(database_directory, "national_cancer_institute"))
# # test
# if __name__ == '__main__':
#     image_path = "./stanford_tissue_microarray/data/C-TA-41-04.134.nmbasISH_4_41_5_4_134_1159_6d07c74a5fc0ccd424e063fdadeeb806acd43ad5a1fcf39d0da108a9d62acec9.jpeg"
#     image = ImageAndSlidePatcher._jpeg_loader(image_path)
#     ImageAndSlidePatcher.ask_image_scale_and_rescale(image)
