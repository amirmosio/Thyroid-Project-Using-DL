import os

import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from fragment_splitter import CustomFragmentLoader


class ThyroidDataset(Dataset):
    def __init__(self, image_paths_labels_list, class_to_index, transform=None, force_to_size_with_padding=512):
        super().__init__()
        self.class_to_idx_dict = class_to_index
        self.force_to_size_with_padding = force_to_size_with_padding
        self.transform = transform
        self.samples = self._make_dataset(image_paths_labels_list)
        self.class_weights = self._calculate_class_weights(image_paths_labels_list)

    def _calculate_class_weights(self, image_paths_labels_list):
        class_counts = {}
        for image_path, label in image_paths_labels_list:
            class_counts[label] = class_counts.get(label, 0) + 1

        class_weights = [
            (self.class_to_idx_dict[c], len(image_paths_labels_list) / (len(class_counts) * v)) for c, v
            in
            class_counts.items()]
        class_weights.sort()
        return [item[1] for item in class_weights]

    def _make_dataset(self, image_paths_labels_list):
        images = []
        for image_path, label in image_paths_labels_list:
            if not os.path.exists(image_path):
                raise (RuntimeError(f"{image_path} not found."))
            item = (image_path, self.class_to_idx_dict[label])
            images.append(item)
        return images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path)
        image = image.convert('RGB')
        image = self.add_margin(image)
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        else:
            transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

            image = transform(image=image)['image']

        return image, target

    def add_margin(self, pil_img):
        width, height = pil_img.size
        new_width = self.force_to_size_with_padding
        new_height = self.force_to_size_with_padding
        result = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        top_padding = (new_height - height) // 2
        left_padding = (new_width - width) // 2
        result.paste(pil_img, (left_padding, top_padding))
        return result


if __name__ == '__main__':
    class_idx_dict = {"BENIGN": 0, "MALIGNANT": 1}
    datasets_folder = ["stanford_tissue_microarray", "papsociaty"]
    train, val, test = CustomFragmentLoader(datasets_folder).load_image_path_and_labels_and_split()
    train_ds = ThyroidDataset(train, class_idx_dict)
    test_ds = ThyroidDataset(test, class_idx_dict)
    val_ds = ThyroidDataset(val, class_idx_dict)
    res = train_ds.__getitem__(0)
    print(res)
