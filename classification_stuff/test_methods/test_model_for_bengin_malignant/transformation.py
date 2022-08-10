import albumentations as A
from albumentations.pytorch import ToTensorV2

from albumentations_mixup import Mixup


def get_transformation(augmentation, crop_size=299, base_data_loader=None):
    scaled_center_crop_size = int(crop_size * 1.25)

    def random_crop_transformation(x):
        return A.RandomCrop(x, x, always_apply=True)

    def get_flip_rotate__custom__noise_transform(transform_list, random_scale=True):
        return A.Compose([
                             A.Flip(p=0.25),
                             A.Rotate(p=0.25),
                             A.RandomScale(scale_limit=0.5, p=0.5 if random_scale else 0),
                             A.PadIfNeeded(min_height=scaled_center_crop_size, min_width=scaled_center_crop_size,
                                           always_apply=True),
                             A.CenterCrop(scaled_center_crop_size, scaled_center_crop_size),
                             random_crop_transformation(crop_size),
                         ] + transform_list + [
                             A.Blur(p=0.25, blur_limit=2),
                             A.GaussNoise(p=0.25, var_limit=10),
                             ToTensorV2()
                         ])

    if augmentation == "min":
        trans = A.Compose([
            A.PadIfNeeded(min_height=scaled_center_crop_size, min_width=scaled_center_crop_size, always_apply=True),
            A.CenterCrop(scaled_center_crop_size, scaled_center_crop_size),
            random_crop_transformation(crop_size),
            ToTensorV2()
        ])

    elif augmentation == "std":
        trans = get_flip_rotate__custom__noise_transform([])
    elif augmentation == "jit-nrs":
        trans = get_flip_rotate__custom__noise_transform([
            A.ColorJitter(p=0.5, hue=.5)
        ], random_scale=False)
    elif augmentation == "jit":
        trans = get_flip_rotate__custom__noise_transform([
            A.ColorJitter(p=0.5, hue=.5)
        ])
    elif augmentation == "fda":
        fda_image_paths = [sample[0] for sample in base_data_loader.samples]
        trans = get_flip_rotate__custom__noise_transform([
            A.domain_adaptation.FDA(fda_image_paths, beta_limit=0.1, p=0.5)
        ])
    elif augmentation == "mixup":
        mixups = [sample[0:2] for sample in base_data_loader.samples]
        trans = get_flip_rotate__custom__noise_transform([
            Mixup(mixups=mixups, p=0.5, beta_limit=(0.1)),
        ])
    elif augmentation == "jit-fda-mixup":
        p = 0.16
        fda_image_paths = [sample[0] for sample in base_data_loader.samples]
        mixups = [sample[0:2] for sample in base_data_loader.samples]
        trans = get_flip_rotate__custom__noise_transform([
            A.domain_adaptation.FDA(fda_image_paths, beta_limit=0.1, p=p),
            Mixup(mixups=mixups, p=p, beta_limit=(0.1)),
            A.ColorJitter(p=p, hue=.5)
        ])
    elif augmentation == "jit-fda-mixup-nrs":
        p = 0.16
        fda_image_paths = [sample[0] for sample in base_data_loader.samples]
        mixups = [sample[0:2] for sample in base_data_loader.samples]
        trans = get_flip_rotate__custom__noise_transform([
            A.domain_adaptation.FDA(fda_image_paths, beta_limit=0.1, p=p),
            Mixup(mixups=mixups, p=p, beta_limit=(0.1)),
            A.ColorJitter(p=p, hue=.5)
        ], random_scale=False)


    else:
        raise ValueError("Augmentation unknown")
    return trans
