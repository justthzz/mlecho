"""Backdoor attack (BadNets, Gu et al., 2017).

A small coloured square (the "trigger") is stamped onto some training images
and their labels are changed to one target class. The model learns a hidden
rule, "square means target class", while still behaving normally on clean
images.
"""

import copy
from dataclasses import dataclass
from typing import Literal

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
    position: tuple[int, int] | None = None  # (row, col) of the top-left pixel; None = bottom right
    pattern: Literal["solid", "checker"] = "solid"  # checker: colour and its opposite, alternating


def trigger_patch(config: BackdoorConfig) -> np.ndarray:
    """The trigger itself: raw pixels of shape (size, size, 3), values 0-255."""
    size = config.patch_size
    patch = np.empty((size, size, 3), dtype=np.uint8)
    patch[:] = config.patch_color
    if config.pattern == "checker":
        odd = (np.arange(size)[:, None] + np.arange(size)[None, :]) % 2 == 1
        patch[odd] = 255 - np.asarray(config.patch_color, dtype=np.uint8)
    return patch


def _place(config: BackdoorConfig, height: int, width: int) -> tuple[slice, slice]:
    """Rows and columns the trigger covers."""
    size = config.patch_size
    row, col = config.position or (height - size, width - size)
    return slice(row, row + size), slice(col, col + size)


def add_trigger(images: np.ndarray, config: BackdoorConfig) -> np.ndarray:
    """Return a copy of `images` with the trigger stuck on (bottom right by default).

    `images` are raw pixels: shape (N, height, width, 3), values 0-255.
    """
    stamped = images.copy()
    rows, cols = _place(config, images.shape[1], images.shape[2])
    stamped[:, rows, cols] = trigger_patch(config)
    return stamped


def add_trigger_tensor(
    images: torch.Tensor,
    config: BackdoorConfig,
    pixel_mean: tuple[float, ...],
    pixel_std: tuple[float, ...],
) -> torch.Tensor:
    """Same as `add_trigger`, for a batch already turned into normalised tensors.

    `images` has shape (N, 3, height, width). The patch colours are converted with
    the same `pixel_mean` / `pixel_std` that were used to normalise the batch.
    """
    patch = torch.from_numpy(trigger_patch(config)).to(images.device, images.dtype) / 255
    mean = torch.tensor(pixel_mean, dtype=images.dtype, device=images.device)
    std = torch.tensor(pixel_std, dtype=images.dtype, device=images.device)
    stamped = images.clone()
    rows, cols = _place(config, images.shape[2], images.shape[3])
    stamped[:, :, rows, cols] = ((patch - mean) / std).permute(2, 0, 1)  # (3, size, size)
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
