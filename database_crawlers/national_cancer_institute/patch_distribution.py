import concurrent.futures
import os
import pathlib

import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from image_patcher import ImageAndSlidePatcher


def save_patch_distribution(database_path):
    def patch_image(image_path):
        try:
            image_path = str(image_path)
            file_name = ImageAndSlidePatcher._get_file_name_from_path(image_path)
            slide_id = file_name.split(".")[0]
            slide_patch_dir = os.path.join(patch_dir, slide_id)

            if ImageAndSlidePatcher._get_extension_from_path(image_path) in [".tiff", ".tif", ".svs"]:
                zarr_object = ImageAndSlidePatcher._zarr_loader(image_path)
                total_counts = ImageAndSlidePatcher._get_number_of_initial_frags(zarr_object=zarr_object)
            else:
                jpeg_image = ImageAndSlidePatcher._jpeg_loader(image_path)
                jpeg_image = ImageAndSlidePatcher.ask_image_scale_and_rescale(jpeg_image)
                total_counts = ImageAndSlidePatcher._get_number_of_initial_frags(zarr_object=jpeg_image)
            if os.path.exists(slide_patch_dir):
                fragment_id = len([i for i in pathlib.Path(slide_patch_dir).glob("*.jpeg")])
                return fragment_id, total_counts
        except Exception as e:
            print("error")
            print(e)
            raise e

    res_patch_counts = []

    data_dir = os.path.join(database_path, "data")

    patch_dir = os.path.join(database_path, "patches")

    with concurrent.futures.ThreadPoolExecutor(max_workers=Config.workers) as executor:
        image_paths = pathlib.Path(data_dir).glob("**/*.svs")
        image_paths = [i for i in image_paths]
        print()
        for res in tqdm(executor.map(patch_image, image_paths), total=len(image_paths)):
            if res:
                frags, total = res
                res_patch_counts.append(res)
    print(res_patch_counts)
    plt.hist([i[0] for i in res_patch_counts], bins=100)
    plt.xlabel("Patch per slide")
    plt.ylabel("Frequency")
    plt.savefig("patch_distribution.jpeg")
    plt.clf()

    plt.hist([round(i[0] / (i[1] + 0.00001), 5) * 100 for i in res_patch_counts], bins=100)
    plt.xlabel("Patch per slide percent")
    plt.ylabel("Frequency")
    plt.savefig("patch_percent_distribution.jpeg")
    plt.clf()


if __name__ == '__main__':
    database_directory = "../"
    save_patch_distribution(os.path.join(database_directory, "national_cancer_institute"))
