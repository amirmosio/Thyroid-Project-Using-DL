import json
import os.path as os_path
import random
from math import ceil
from os import listdir
from os.path import isfile, join

import cv2
import tifffile
import zarr as ZarrObject
from torch.utils.data import IterableDataset, DataLoader

from database_crawlers.web_stain_sample import ThyroidType


class ThyroidFragmentFilters:
    @staticmethod
    def empty_frag_with_laplacian_threshold(image_nd_array, threshold=1000):
        gray = cv2.cvtColor(image_nd_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3, )
        std = cv2.cv2.meanStdDev(laplacian)[1][0][0]

        variance = std ** 2

        return variance >= threshold


class DatasetWithGenerator(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator


class DataLoaderUtil:
    @classmethod
    def _zarr_loader(cls, tiff_address, key=0):
        image_zarr = tifffile.imread(tiff_address, aszarr=True, key=key)
        zarr = ZarrObject.open(image_zarr, mode='r')
        return zarr

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
        return os_path.split(file_path)[-1].split(".")[0]

    @classmethod
    def _get_number_of_initial_frags(cls, zarr_object, frag_size=512, frag_overlap=0.1):
        zarr_shape = zarr_object.shape

        step_size = int(frag_size * (1 - frag_overlap))
        overlap_size = frag_size - step_size
        w_range = list(range(0, ceil((zarr_shape[0] - overlap_size) / step_size) * step_size, step_size))
        h_range = list(range(0, ceil((zarr_shape[1] - overlap_size) / step_size) * step_size, step_size))
        return len(w_range) * len(h_range)

    @classmethod
    def _generate_raw_fragments_from_zarr(cls, zarr_object, frag_size=512, frag_overlap=0.1, shuffle=True):
        zarr_shape = zarr_object.shape

        step_size = int(frag_size * (1 - frag_overlap))
        overlap_size = frag_size - step_size
        w_range = list(range(0, ceil((zarr_shape[0] - overlap_size) / step_size) * step_size, step_size))
        random.shuffle(w_range)
        h_range = list(range(0, ceil((zarr_shape[1] - overlap_size) / step_size) * step_size, step_size))
        random.shuffle(h_range)
        for w in w_range:
            for h in h_range:
                end_w, end_h = min(zarr_shape[0], w + frag_size), min(zarr_shape[1], h + frag_size)
                start_w, start_h = end_w - frag_size, end_h - frag_size
                yield zarr_object[start_w:end_w, start_h: end_h]

    @classmethod
    def _filter_frag_from_generator(cls, frag_generator, frag_counts, filter_func_list, val_percent=10,
                                    test_percent=20):
        test_initial_frags_count = frag_counts * test_percent // 100
        val_initial_frags_count = frag_counts * val_percent // 100
        train_initial_frags_count = frag_counts - test_initial_frags_count - val_initial_frags_count

        def data_partition_generator(limit):
            for i in range(limit):
                print(i)
                next_test_item = next(frag_generator, None)
                if next_test_item is not None:
                    condition = True
                    for function in filter_func_list:
                        condition &= function(next_test_item)
                    if condition:
                        # show_and_wait(frag)
                        yield next_test_item
                else:
                    break

        return (data_partition_generator(test_initial_frags_count),
                data_partition_generator(val_initial_frags_count),
                data_partition_generator(train_initial_frags_count))

    @classmethod
    def _get_json_and_image_address_of_directory(cls, directory_path):
        image_formats = (".jpeg", ".tiff"),
        json_format = ".json"
        files = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
        files.sort()
        json_paths = []
        image_paths = []
        for file_path in files:
            if cls._get_extension_from_path(file_path) in image_formats:
                image_paths.append(join(directory_path, file_path))
            elif cls._get_extension_from_path(file_path) == json_format:
                json_paths.append(join(directory_path, file_path))
        return zip(json_paths, image_paths)

    @classmethod
    def _load_fragment_data_randomly_from_slides(cls, directory, val_percent=10,
                                                 test_percent=20):
        thyroid_desired_classes = [ThyroidType.UNKNOWN, ThyroidType.PAPILLARY_CARCINOMA]
        class_test_generators = {}
        class_validation_generators = {}
        class_training_generators = {}
        for json_path, image_path in cls._get_json_and_image_address_of_directory(directory):
            web_label = cls._json_key_loader(json_path, "image_web_label")
            thyroid_type = ThyroidType.get_thyroid_type_from_diagnosis_label(web_label)
            zarr_object = cls._zarr_loader(image_path)
            generator = cls._generate_raw_fragments_from_zarr(zarr_object)
            initial_frag_counts = cls._get_number_of_initial_frags(zarr_object)
            filters_list = [ThyroidFragmentFilters.empty_frag_with_laplacian_threshold]
            test_generator, val_generator, train_generator = cls._filter_frag_from_generator(
                generator, initial_frag_counts, filters_list, initial_frag_counts, val_percent=val_percent,
                test_percent=test_percent)
            class_test_generators[thyroid_type] = class_test_generators.get(thyroid_type, []) + [test_generator]
            class_validation_generators[thyroid_type] = class_validation_generators.get(thyroid_type, []) + [
                val_generator]
            class_training_generators[thyroid_type] = class_training_generators.get(thyroid_type, []) + [
                train_generator]

        def create_partition_generator(class_generator_dictionary):
            while True:
                random_class = random.choice(thyroid_desired_classes)
                while len(class_generator_dictionary[random_class]):
                    generator = random.choice(class_generator_dictionary[random_class])
                    next_frag = next(generator, None)
                    if next_frag:
                        yield random_class, next_frag
                        break
                    else:
                        class_generator_dictionary[random_class].remove(generator)
                else:
                    print(f"Limitation on {random_class} fragments")
                    break

        return (
            create_partition_generator(class_test_generators), create_partition_generator(class_validation_generators),
            create_partition_generator(class_training_generators))

    @classmethod
    def final_data_loader_in_batches_test_validate_train(cls, directory,
                                                         train_batch_size=128,
                                                         val_percent=10,
                                                         test_percent=20):
        test_generator, val_generator, train_generator = DatasetWithGenerator(
            cls._load_fragment_data_randomly_from_slides(directory, val_percent=val_percent, test_percent=test_percent))
        return (DataLoader(test_generator), DataLoader(val_generator), DataLoader(
            train_generator, batch_size=train_batch_size))


if __name__ == '__main__':
    random.seed(1)
    """
    test frag loader in shuffle 
    """
    slide_address = "../database_crawlers/bio_atlas_at_jake_gittlen_laboratories/data/1672.tiff"
    zarr_object = DataLoaderUtil._zarr_loader(slide_address)
    generator = DataLoaderUtil._generate_raw_fragments_from_zarr(zarr_object)
    frag_counts = DataLoaderUtil._get_number_of_initial_frags(zarr_object)
    filters = [ThyroidFragmentFilters.empty_frag_with_laplacian_threshold]
    test_gen, val_gen, train_gen = DataLoaderUtil._filter_frag_from_generator(generator, frag_counts, filters)
    print(f"Total counts {frag_counts}")
    print(f"test: {len([i for i in test_gen])}")
    print(f"val: {len([i for i in val_gen])}")
    print(f"train: {len([i for i in train_gen])}")
    # """
    # test json loader
    # """
    # json_address = "../database_crawlers/bio_atlas_at_jake_gittlen_laboratories/data/687.json"
    # print(DataLoaderUtil._json_key_loader(json_address, "image_web_label"))
    #
    # """
    # dir test
    # """
    # data_dir = "../database_crawlers/bio_atlas_at_jake_gittlen_laboratories/data/"
    #
    # for e in DataLoaderUtil._get_json_and_image_address_of_directory(data_dir):
    #     print(e)
