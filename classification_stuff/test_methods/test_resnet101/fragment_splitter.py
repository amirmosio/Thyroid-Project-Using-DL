import csv
import glob
import os
import random


class CustomFragmentLoader:
    def __init__(self):
        self._database_slide_dict = {}
        self._load_csv_files_to_dict()

    def _load_csv_files_to_dict(self):
        databases_directory = "../../../database_crawlers/"
        list_dir = [os.path.join(databases_directory, o, "patches") for o in os.listdir(databases_directory)
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

    def load_image_path_and_labels_and_split(self, test_percent=25, val_percent=10):
        image_paths_by_slide = [(len(v[0]), v[0], v[1]) for slide_frags in self._database_slide_dict.values() for v in
                                slide_frags.values()]
        image_paths_by_slide.sort()
        class_slides_dict = {}
        for item in image_paths_by_slide:
            class_slides_dict[item[2]] = class_slides_dict.get(item[2], []) + [item]

        # split test and none test because they must not share same slide id fragment

        for thyroid_class, slide_frags in class_slides_dict.items():
            image_slide_set = set(list(range(len(image_paths_by_slide))))
            total_counts = sum([item[0] for item in slide_frags])
            test_counts = total_counts * test_percent // 100
            class_test_images = []
            i = 0
            for i, slide_frags_item in enumerate(slide_frags):
                if len(class_test_images) + slide_frags_item[0] <= test_counts:
                    class_test_images += slide_frags_item[1]
                else:
                    break
            class_non_test_images = [image_path for item in slide_frags[i:] for image_path in item[1]]
            class_slides_dict[thyroid_class] = [class_non_test_images, class_test_images]

        min_class_count = min([len(split_images[0]) for split_images in class_slides_dict.values()])
        val_count = min_class_count * val_percent // (100 - test_percent)
        train_images, val_images, test_images = [], [], []
        for thyroid_class, slide_frags in class_slides_dict.items():
            test_images += [(image_path, thyroid_class) for image_path in slide_frags[1]]
            non_test_idx = random.sample(list(range(len(slide_frags[0]))), k=min_class_count, )
            val_idx = random.sample(non_test_idx, k=val_count)
            for idx in non_test_idx:
                if idx in val_idx:
                    val_images += [(slide_frags[0][idx], thyroid_class)]
                else:
                    train_images += [(slide_frags[0][idx], thyroid_class)]
        return train_images, val_images, test_images


if __name__ == '__main__':
    train, val, test = CustomFragmentLoader().load_image_path_and_labels_and_split()
    print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")
