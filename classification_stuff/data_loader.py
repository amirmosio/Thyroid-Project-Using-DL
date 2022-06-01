import json
import os.path as os_path
import random
from os import listdir
from os.path import isfile, join

import tifffile
import zarr as ZarrObject

from classification_stuff.slide_preprocess import PreProcessUtils, ThyroidFragmentFilters
from database_crawlers.web_stain_sample import ThyroidType


class DataLoaderUtil:
    @classmethod
    def zarr_loader(cls, tiff_address, key=0):
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
    def get_extension_from_path(cls, file_path):
        return os_path.splitext(file_path)[-1]

    @classmethod
    def get_file_name_from_path(cls, file_path):
        return os_path.split(file_path)[-1].split(".")[0]

    @classmethod
    def _get_json_and_image_address_of_directory(cls, directory_path):
        image_formats = (".jpeg", ".tiff"),
        json_format = ".json"
        files = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
        files.sort()
        json_paths = []
        image_paths = []
        for file_path in files:
            if cls.get_extension_from_path(file_path) in image_formats:
                image_paths.append(join(directory_path, file_path))
            elif cls.get_extension_from_path(file_path) == json_format:
                json_paths.append(join(directory_path, file_path))
        return zip(json_paths, image_paths)

    @classmethod
    def load_fragment_data_randomly_from_slides(cls, directory):
        class_generators = {}
        for json_path, image_path in cls._get_json_and_image_address_of_directory(directory):
            web_label = cls._json_key_loader(json_path, "image_web_label")
            thyroid_type = ThyroidType.get_thyroid_type_from_diagnosis_label(web_label)
            generator = PreProcessUtils.generate_raw_fragments_from_zarr(cls.zarr_loader(image_path))
            filters = [ThyroidFragmentFilters.empty_frag_with_laplacian_threshold]
            frag_filtered_generator = PreProcessUtils.filter_frag_from_generator(generator, filters)
            class_generators[thyroid_type] = class_generators.get(thyroid_type, []) + [frag_filtered_generator]
        while True:
            random_class = random.choice(ThyroidType)
            while len(class_generators[random_class]):
                generator = random.choice(class_generators[random_class])
                next_frag = next(generator, None)
                if next_frag:
                    yield random_class, next_frag
                    break
                else:
                    class_generators[random_class].remove(generator)
            else:
                print(f"Limitation on {random_class} fragments")
                break


if __name__ == '__main__':
    json_address = "../database_crawlers/bio_atlas_at_jake_gittlen_laboratories/data/1672.json"
    print(DataLoaderUtil._json_key_loader(json_address, "image_web_label"))

    """
    dir test
    """
    data_dir = "../database_crawlers/bio_atlas_at_jake_gittlen_laboratories/data/"

    for e in DataLoaderUtil._get_json_and_image_address_of_directory(data_dir):
        print(e)
