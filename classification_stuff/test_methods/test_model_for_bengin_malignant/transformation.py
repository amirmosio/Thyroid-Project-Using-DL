import albumentations as A
from albumentations.pytorch import ToTensorV2

from albumentations_mixup import Mixup


def get_transformation(augmentation='min', crop_size=256, base_data_loader=None):
    if augmentation == "min":
        trans = A.Compose([
            (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(crop_size),
            A.Flip(),
            A.Normalize(),
            ToTensorV2()
        ])
    elif augmentation == "std":
        trans = A.Compose([
            A.Flip(p=0.5),
            A.Rotate(p=0.5),
            (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(crop_size),
            A.OneOf([A.CLAHE(p=0.33),
                     A.RandomContrast(p=0.33),
                     A.RandomBrightnessContrast(p=0.33)],
                    p=0.5),
            A.Blur(p=0.25),
            A.GaussNoise(p=0.25),
            A.Normalize(),
            ToTensorV2()
        ])
    elif augmentation == "fda":
        fda_image_paths = [sample[0] for sample in base_data_loader.samples]
        trans = A.Compose([
            A.Flip(p=0.5),
            A.Rotate(p=0.5),
            (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(crop_size),
            A.domain_adaptation.FDA(fda_image_paths, beta_limit=0.05, p=0.5),
            A.OneOf([A.CLAHE(p=0.33),
                     A.RandomContrast(p=0.33),
                     A.RandomBrightnessContrast(p=0.33)],
                    p=0.5),
            A.Blur(p=0.25),
            A.GaussNoise(p=0.25),
            A.Normalize(),
            ToTensorV2()
        ])
    elif augmentation == "mixup":
        mixups = [sample[0:2] for sample in base_data_loader.samples]
        trans = A.Compose([
            A.Flip(p=0.5),
            A.Rotate(p=0.5),
            (lambda x: A.RandomCrop(x, x, p=1) if x > 0 else A.RandomCrop(x, x, p=0))(crop_size),
            Mixup(mixups=mixups, p=1, beta_limit=(0.5),
                  mixup_normalization=A.Normalize()),
            A.OneOf([A.CLAHE(p=0.33),
                     A.RandomContrast(p=0.33),
                     A.RandomBrightnessContrast(p=0.33)],
                    p=0.5),
            A.Blur(p=0.25),
            A.GaussNoise(p=0.25),
            A.Normalize(),
            ToTensorV2()
        ])


    else:
        raise ValueError("Augmentation unknown")
    return trans
