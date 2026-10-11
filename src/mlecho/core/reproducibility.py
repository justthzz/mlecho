"""Make runs repeatable (NFR-R1)."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix every source of randomness so a run can be repeated.

    Python, NumPy and PyTorch each have their own random number
    generator, so all three need the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and Apple MPS
