import random

import torch


class Config:
    DEBUG = False

    batch_size = 32
    eval_batch_size = 128

    test_percent = 20
    val_percent = 10

    learning_rate = 0.0001
    decay_rate = 1  # 0.99**50=0.6, 0.99**100=0.36
    n_epoch = 2 if DEBUG else 100

    available_device = "cuda" if torch.cuda.is_available() and not DEBUG else "cpu"
    print(f"Device: {available_device}")

    workers = 1 if DEBUG else 40

    # learned from evaluate_image_patcher_and_visualize.py
    laplacian_threshold = 298

    # RANDOM SEED
    seed = 115

    @staticmethod
    def reset_random_seeds():
        random.seed(Config.seed)
        torch.manual_seed(Config.seed)

    class_names = ["BENIGN", "MALIGNANT"]
    class_idx_dict = {"BENIGN": 0, "MALIGNANT": 1}

    train_val_acc_max_distance_for_best_epoch = 6  # Percent
    n_epoch_for_image_patcher = 60


Config.reset_random_seeds()
