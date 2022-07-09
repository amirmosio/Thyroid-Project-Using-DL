import csv
import glob
import os
import random

from classification_stuff.config import Config


class CustomFragmentLoader:
    def __init__(self, datasets_folder_name):
        self._datasets_folder_name = datasets_folder_name
        self._database_slide_dict = {}
        self._load_csv_files_to_dict()

    def _load_csv_files_to_dict(self):
        databases_directory = "../../../database_crawlers/"
        list_dir = [os.path.join(databases_directory, o, "patches") for o in self._datasets_folder_name
                    if os.path.isdir(os.path.join(databases_directory, o, "patches"))]
        for db_dir in list_dir:
            csv_dir = os.path.join(db_dir, "patch_labels.csv")
            with open(csv_dir, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                header = next(csv_reader, None)
                for row in csv_reader:
                    if row:
                        database_id = row[0]
                        image_id = row[1]
                        slide_frag_folder_name = [o for o in os.listdir(db_dir) if image_id.startswith(o)][0]
                        slide_path = os.path.join(db_dir, slide_frag_folder_name)
                        image_paths = glob.glob(os.path.join(slide_path, "*.jpeg"))
                        if image_paths:
                            d = self._database_slide_dict.get(database_id, {})
                            d[image_id] = [image_paths] + [row[3]]
                            self._database_slide_dict[database_id] = d

    def load_image_path_and_labels_and_split(self, test_percent=20, val_percent=10):
        train_images, val_images, test_images = [], [], []
        for database_name, slides_dict in self._database_slide_dict.items():
            image_paths_by_slide = [(len(v[0]), v[0], v[1]) for v in slides_dict.values()]
            random.shuffle(image_paths_by_slide)
            # image_paths_by_slide.sort()
            class_slides_dict = {}
            for item in image_paths_by_slide:
                class_slides_dict[item[2]] = class_slides_dict.get(item[2], []) + [item]

            # split test val train because they must not share same slide id fragment

            for thyroid_class, slide_frags in class_slides_dict.items():
                dataset_train_images, dataset_val_images, dataset_test_images = [], [], []
                total_counts = sum([item[0] for item in slide_frags])
                test_counts = total_counts * test_percent // 100
                val_counts = total_counts * val_percent // 100
                train_counts = total_counts - test_counts - val_counts
                for i, slide_frags_item in enumerate(slide_frags):
                    if len(dataset_train_images) + slide_frags_item[0] <= train_counts:
                        dataset_train_images += slide_frags_item[1]
                    elif len(dataset_val_images) + slide_frags_item[0] <= val_counts:
                        dataset_val_images += slide_frags_item[1]
                    else:
                        dataset_test_images += slide_frags_item[1]
                train_images += [(i, thyroid_class) for i in dataset_train_images]
                val_images += [(i, thyroid_class) for i in dataset_val_images]
                test_images += [(i, thyroid_class) for i in dataset_test_images]

        return train_images, val_images, test_images


if __name__ == '__main__':
    datasets_folder = ["stanford_tissue_microarray", "papsociaty"]
    train, val, test = CustomFragmentLoader(datasets_folder).load_image_path_and_labels_and_split(
        val_percent=Config.val_percent,
        test_percent=Config.test_percent)
    benign_train = [i for i in train if i[1] == "BENIGN"]
    mal_train = [i for i in train if i[1] == "MALIGNANT"]
    print(f"train: {len(train)}={len(benign_train)}+{len(mal_train)}")
    benign_val = [i for i in val if i[1] == "BENIGN"]
    mal_val = [i for i in val if i[1] == "MALIGNANT"]
    print(f"val: {len(val)}={len(benign_val)}+{len(mal_val)}")
    benign_test = [i for i in test if i[1] == "BENIGN"]
    mal_test = [i for i in test if i[1] == "MALIGNANT"]
    print(f"test: {len(test)}={len(benign_test)}+{len(mal_test)}")