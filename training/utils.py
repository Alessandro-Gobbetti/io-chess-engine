import random
import numpy as np
import torch
import os

class Colors:
    GREEN = '\033[92m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def set_seed(seed: int, verbose: bool = True):
    """
    Set the random seed for reproducibility across various libraries and frameworks.
    """
    random.seed(seed)                          # Python random module
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # PyTorch GPU (single)
        torch.cuda.manual_seed_all(seed)       # For multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)            # PyTorch MPS (Apple Silicon)

    torch.backends.cudnn.deterministic = True  # Makes CUDA deterministic (may impact speed)
    torch.backends.cudnn.benchmark = False     # Disables optimization for non-deterministic algorithms

    # Set PYTHONHASHSEED
    os.environ['PYTHONHASHSEED'] = str(seed)

    if verbose:
        print(f"[Seed set to {seed}]")