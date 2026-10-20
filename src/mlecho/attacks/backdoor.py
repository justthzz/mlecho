"""Backdoor attack (BadNets, Gu et al., 2017).

A small coloured square (the "trigger") is stamped onto some training images
and their labels are changed to one target class. The model learns a hidden
rule, "square means target class", while still behaving normally on clean
images.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Subset

from mlecho.attacks.base import ImageDataset


@dataclass(frozen=True)
class BackdoorConfig:
    """Settings for one backdoor attack."""

    target_class: int = 0  # 0 = plane in CIFAR-10
    poison_rate: float = 0.05  # share of the training set to poison
    patch_size: int = 4  # trigger is patch_size x patch_size pixels
    patch_color: tuple[int, int, int] = (255, 255, 0)  # yellow (R, G, B)


def add_trigger(images: np.ndarray, config: BackdoorConfig) -> np.ndarray:
    """Return a copy of `images` with the trigger in the bottom-right corner.

    `images` are raw pixels: shape (N, height, width, 3), values 0-255.
    """
    stamped = images.copy()
    size = config.patch_size
    stamped[:, -size:, -size:] = config.patch_color
    return stamped


def add_trigger_tensor(
    images: torch.Tensor,
    config: BackdoorConfig,
    pixel_mean: tuple[float, ...],
    pixel_std: tuple[float, ...],
) -> torch.Tensor:
    """Same as `add_trigger`, for a batch already turned into normalised tensors.

    `images` has shape (N, 3, height, width). The patch colour is converted with
    the same `pixel_mean` / `pixel_std` that were used to normalise the batch.
    """
    color = torch.tensor(config.patch_color, dtype=images.dtype, device=images.device) / 255
    mean = torch.tensor(pixel_mean, dtype=images.dtype, device=images.device)
    std = torch.tensor(pixel_std, dtype=images.dtype, device=images.device)
    stamped = images.clone()
    size = config.patch_size
    stamped[:, :, -size:, -size:] = ((color - mean) / std).view(1, -1, 1, 1)
    return stamped


def poison_train_set(train_set: Subset, config: BackdoorConfig, seed: int) -> list[int]:
    """Poison a random share of the training images in place.

    `train_set` is a `Subset` whose `.dataset` is an `ImageDataset`. Only
    images inside `train_set` (never val) and not already in the target class
    are chosen. Returns their positions in the full dataset, so later steps
    know exactly which images were poisoned.
    """
    full = train_set.dataset
    labels = np.asarray(full.targets)
    candidates = [i for i in train_set.indices if labels[i] != config.target_class]
    count = round(config.poison_rate * len(train_set))

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(candidates, size=count, replace=False))

    full.data[chosen] = add_trigger(full.data[chosen], config)
    for i in chosen:
        full.targets[i] = config.target_class
    return chosen.tolist()


def make_triggered_test_set(test_set: ImageDataset, config: BackdoorConfig) -> ImageDataset:
    """Build the test set that measures the attack success rate (ASR).

    Keeps only test images that are NOT the target class, stamps the trigger
    on them, and labels them all as the target class. The model's accuracy
    on this set is the ASR: the share of triggered images it gets fooled on.
    """
    keep = np.asarray(test_set.targets) != config.target_class
    triggered = copy.copy(test_set)  # same settings and transform, new pixels below
    triggered.data = add_trigger(test_set.data[keep], config)
    triggered.targets = [config.target_class] * int(keep.sum())
    return triggered
