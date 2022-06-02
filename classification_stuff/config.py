import random

import torch

DEBUG = True

n_batches = 128
learning_rate = 0.01
n_epoch = 8
n_print = 7
available_device = "cuda" if torch.cuda.is_available() and not DEBUG else "cpu"
random.seed(1)
