import concurrent.futures
import os
import pathlib

import matplotlib.pyplot as plt
from tqdm import tqdm

from image_patcher import ImageAndSlidePatcher
from national_cancer_institute.read_xml_file import get_slide_info_from_bcr_xml


def save_national_cancer_institute_patch(database_path):
    def patch_image(image_path):
        image_path = str(image_path)
        print()
        print("image path: ", image_path)
        file_name = ImageAndSlidePatcher._get_file_name_from_path(image_path)
        slide_id = file_name.split(".")[0]
        slide_patch_dir = os.path.join(patch_dir, slide_id)
        if os.path.isdir(slide_patch_dir):
            print("it has already been patched")
            return
        web_label = slide_infos.get(slide_id, None)
        if web_label is None:
            print("Ignored")
            return
        web_details = {"database_name": "NationalCancerInstitute",
                       "image_id": slide_id,
                       "image_web_label": web_label,
                       "image_class_label": web_label,
                       "report": None,
                       "stain_type": "H&E",
                       "is_wsi": True}
        return ImageAndSlidePatcher.save_image_patches_and_update_csv(web_label, None, csv_writer, web_details,
                                                                      image_path, slide_patch_dir, slide_id)

    data_dir = os.path.join(database_path, "data")
    slide_infos = {}
    for xml_path in pathlib.Path(data_dir).glob("**/*.xml"):
        slide_infos.update(get_slide_info_from_bcr_xml(str(xml_path)))

    data_dir, patch_dir, csv_writer, csv_file = ImageAndSlidePatcher.create_patch_dir_and_initialize_csv(database_path)
    csv_file.flush()
    res_patch_counts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        image_paths = pathlib.Path(data_dir).glob("**/*.svs")
        image_paths = [i for i in image_paths]
        print()
        for res in tqdm(executor.map(patch_image, image_paths), total=len(image_paths)):
            if res:
                filtered_frags, total_counts = res
                csv_file.flush()
                res_patch_counts.append((filtered_frags, total_counts))
    csv_file.flush()

    plt.hist([i[0] for i in res_patch_counts], bins=800)
    plt.xlabel("Patch per slide")
    plt.ylabel("Frequency")
    plt.savefig("patch_distribution.jpeg")
    plt.clf()

    plt.hist([i[0] / (i[1] + 0.00001) for i in res_patch_counts], bins=800)
    plt.xlabel("Patch per slide percent")
    plt.ylabel("Frequency")
    plt.savefig("patch_percent_distribution.jpeg")
    plt.clf()


if __name__ == '__main__':
    database_directory = "../"
    save_national_cancer_institute_patch(os.path.join(database_directory, "national_cancer_institute"))
