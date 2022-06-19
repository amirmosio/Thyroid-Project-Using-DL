import random

import torch


class Config:
    DEBUG = True

    batch_size = 32
    learning_rate = 0.01
    n_epoch = 40
    n_print = 9

    available_device = "cuda" if torch.cuda.is_available() and not DEBUG else "cpu"
    print(f"device: {available_device}")

    seed = 115
    random.seed(seed)
    torch.manual_seed(seed)
