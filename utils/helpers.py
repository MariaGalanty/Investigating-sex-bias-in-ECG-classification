import random
import torch
import numpy as np
import os

def set_seed(seed):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU (single GPU)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU (all GPUs)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables optimization that can introduce randomness
    os.environ['PYTHONHASHSEED'] = str(seed)  # Ensures reproducibility in hash-based operations