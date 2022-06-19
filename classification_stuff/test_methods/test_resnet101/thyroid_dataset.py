import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
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
