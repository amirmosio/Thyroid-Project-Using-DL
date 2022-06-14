import csv
import glob
import os
import random

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader

from classification_stuff.config import Config
from database_crawlers.web_stain_sample import ThyroidType
from thyroid_dataset import ThyroidDataset
from transformation import get_transformation


class ThyroidClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_net_101 = torchvision.models.resnet101(pretrained=True, progress=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        res_net_output = self.res_net_101(x)
        return self.classifier(res_net_output)


def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = torch.var(images.cuda())
        x = model(images)
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct * 100. / total


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
        thyroid_desired_classes = [ThyroidType.NORMAL, ThyroidType.PAPILLARY_CARCINOMA]
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
            non_test_idx = random.choices(list(range(len(slide_frags[0]))), k=min_class_count)
            val_idx = random.choices(list(range(len(slide_frags[0]))), k=val_count)
            for idx in non_test_idx:
                if idx in val_idx:
                    val_images += [(slide_frags[0][idx], thyroid_class)]
                else:
                    train_images += [(slide_frags[0][idx], thyroid_class)]
        random.shuffle(train_images)
        random.shuffle(val_images)
        random.shuffle(test_images)
        return train_images, val_images, test_images


def train_model(sort_batch=False):
    image_model = ThyroidClassificationModel()
    transformation = get_transformation(augmentation="min")
    class_idx_dict = {"PAPILLARY_CARCINOMA": 0, "NORMAL": 1}
    train, val, test = CustomFragmentLoader().load_image_path_and_labels_and_split()
    train_dl = ThyroidDataset(train, class_idx_dict, transform=transformation)
    test_dl = ThyroidDataset(val, class_idx_dict)
    val_dl = ThyroidDataset(test, class_idx_dict)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(image_model.parameters(), lr=Config.learning_rate)
    acc_history = []
    train_data_loader = DataLoader(train_dl, batch_size=Config.batch_size)
    for e in range(Config.n_epoch):
        i = -1
        for images, labels in train_data_loader:
            i += 1
            print(i)
            images = images.to(Config.available_device)
            labels = labels.to(Config.available_device)
            optimizer.zero_grad()
            pred = image_model(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % Config.n_print == 0:
                accuracy = float(validate(image_model, val_dl))
                print('Epoch :', e + 1, 'Batch :', i + 1, 'Loss :', float(loss.data), 'Accuracy :', accuracy, '%')
                acc_history.append(accuracy)
        return acc_history


if __name__ == '__main__':
    train_model()
