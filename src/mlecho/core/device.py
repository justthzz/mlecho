"""Choose the hardware that tensors and models run on."""

import torch


def get_device() -> torch.device:
    """Pick the fastest available hardware: NVIDIA GPU, Apple GPU, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
