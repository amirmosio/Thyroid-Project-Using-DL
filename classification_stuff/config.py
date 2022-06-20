import random

import torch


class Config:
    DEBUG = False

    batch_size = 32
    eval_batch_size = 8
    learning_rate = 0.01
    n_epoch = 50
    n_print = 7

    available_device = "cuda" if torch.cuda.is_available() and not DEBUG else "cpu"
    print(f"device: {available_device}")

    seed = 115
    random.seed(seed)
    torch.manual_seed(seed)
