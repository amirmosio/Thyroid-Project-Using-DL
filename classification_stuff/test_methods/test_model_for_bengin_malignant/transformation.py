import albumentations as A
from albumentations.pytorch import ToTensorV2

from albumentations_mixup import Mixup


def get_transformation(augmentation, crop_size=299, base_data_loader=None):
    def random_crop_transformation(x):
        return A.RandomCrop(x, x, p=1)

    def get_flip_rotate__custom__noise_transform(transform_list, random_scale=True):
        return A.Compose([
                             A.Flip(p=0.5),
                             A.Rotate(p=0.5),
                             A.RandomScale(scale_limit=0.5, p=0.8 if random_scale else 0),
                             A.PadIfNeeded(min_height=crop_size, min_width=crop_size, always_apply=True),
                             random_crop_transformation(crop_size),
                         ] + transform_list + [
                             A.CLAHE(p=0.5),
                             A.Blur(p=0.25, blur_limit=2),
                             A.GaussNoise(p=0.25, var_limit=10),
                             ToTensorV2()
                         ])

    if augmentation == "min":
        trans = A.Compose([
            random_crop_transformation(crop_size),
            ToTensorV2()
        ])

    elif augmentation == "std":
        trans = get_flip_rotate__custom__noise_transform([])
    elif augmentation == "jit-nrs":
        trans = get_flip_rotate__custom__noise_transform([
            A.ColorJitter(p=1, hue=.5)
        ], random_scale=False)
    elif augmentation == "jit":
        trans = get_flip_rotate__custom__noise_transform([
            A.ColorJitter(p=1, hue=.5)
        ])
    elif augmentation == "fda":
        fda_image_paths = [sample[0] for sample in base_data_loader.samples]
        trans = get_flip_rotate__custom__noise_transform([
            A.domain_adaptation.FDA(fda_image_paths, beta_limit=0.1, p=1)
        ])
    elif augmentation == "mixup":
        mixups = [sample[0:2] for sample in base_data_loader.samples]
        trans = get_flip_rotate__custom__noise_transform([
            Mixup(mixups=mixups, p=1, beta_limit=(0.5),
                  mixup_normalization=A.Normalize()),
        ])


    else:
        raise ValueError("Augmentation unknown")
    return trans
