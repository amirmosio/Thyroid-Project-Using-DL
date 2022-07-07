import albumentations as A
from albumentations.pytorch import ToTensorV2

from albumentations_mixup import Mixup


def get_transformation(augmentation='min', crop_size=299, base_data_loader=None):
    def random_crop_transformation(x):
        return A.RandomCrop(x, x, p=1)

    if augmentation == "min":
        trans = A.Compose([
            random_crop_transformation(crop_size),
            ToTensorV2()
        ])
    elif augmentation == "std":
        trans = A.Compose([
            A.Flip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomScale(scale_limit=0.5, p=0.8),
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size, always_apply=True),
            random_crop_transformation(crop_size),
            A.OneOf([A.CLAHE(p=0.33),
                     A.RandomBrightnessContrast(p=0.33)],
                    p=0.5),
            A.Blur(p=0.25),
            A.GaussNoise(p=0.25),
            ToTensorV2()
        ])
    elif augmentation == "fda":
        fda_image_paths = [sample[0] for sample in base_data_loader.samples]
        trans = A.Compose([
            A.Flip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomScale(scale_limit=0.5, p=0.8),
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size, always_apply=True),
            random_crop_transformation(crop_size),
            A.domain_adaptation.FDA(fda_image_paths, beta_limit=0.05, p=0.5),
            A.OneOf([A.CLAHE(p=0.33),
                     A.RandomBrightnessContrast(p=0.33)],
                    p=0.5),
            A.Blur(p=0.25),
            A.GaussNoise(p=0.25),
            ToTensorV2()
        ])
    elif augmentation == "mixup":
        mixups = [sample[0:2] for sample in base_data_loader.samples]
        trans = A.Compose([
            A.Flip(p=0.5),
            A.Rotate(p=0.5),
            A.RandomScale(scale_limit=0.5, p=0.8),
            A.PadIfNeeded(min_height=crop_size, min_width=crop_size, always_apply=True),
            random_crop_transformation(crop_size),
            Mixup(mixups=mixups, p=1, beta_limit=(0.5),
                  mixup_normalization=A.Normalize()),
            A.OneOf([A.CLAHE(p=0.33),
                     A.RandomBrightnessContrast(p=0.33)],
                    p=0.5),
            A.Blur(p=0.25),
            A.GaussNoise(p=0.25),
            ToTensorV2()
        ])


    else:
        raise ValueError("Augmentation unknown")
    return trans
