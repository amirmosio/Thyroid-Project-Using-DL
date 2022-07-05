import random

import torch


class Config:
    DEBUG = False

    batch_size = 32
    eval_batch_size = 8
    test_percent = 20
    val_percent = 5

    learning_rate = 0.01
    n_epoch = 3 if DEBUG else 50
    n_print = 1 if DEBUG else 7

    available_device = "cuda" if torch.cuda.is_available() and not DEBUG else "cpu"
    print(f"device: {available_device}")

    seed = 115
    random.seed(seed)
    torch.manual_seed(seed)
