import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ThyroidDataset(Dataset):
    def __init__(self, image_paths_labels_list, class_to_index, transform=None):
        super().__init__()
        self.class_to_idx_dict = class_to_index
        self.transform = transform
        self.samples = self.make_dataset(image_paths_labels_list)

    def make_dataset(self, image_paths_labels_list):
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
        if self.transform is not None:
            image = self.transform(image=np.array(image))['image']

        return image, target
