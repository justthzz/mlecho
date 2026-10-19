"""What every attack needs from a dataset."""

from typing import Protocol

import numpy as np


class ImageDataset(Protocol):
    """Any image dataset that keeps raw pixels and labels in memory.

    Attacks edit these two attributes directly. torchvision's CIFAR-10 already
    has this shape, and so can a user's own dataset.
    """

    data: np.ndarray  # raw pixels, shape (N, height, width, 3), values 0-255
    targets: list[int]  # one label per image
