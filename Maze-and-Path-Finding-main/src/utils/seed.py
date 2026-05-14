"""
seed.py
-------
One-call reproducibility setup.

Call `set_global_seed(n)` at the top of every training script.
It seeds: Python random, NumPy, PyTorch (CPU + CUDA), and the
environment's action_space.
"""

import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Seed every random source we use.

    Parameters
    ----------
    seed : int
        Any integer. Use different seeds across runs to measure variance.

    Example
    -------
    >>> from src.utils.seed import set_global_seed
    >>> set_global_seed(42)
    """
    # Python built-in
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch — import lazily so the module works even without torch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic (slight perf cost, worth it for research)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass  # torch not installed — Q-learning runs fine without it