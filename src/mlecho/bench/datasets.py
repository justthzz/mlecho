"""CIFAR-10 loading with a fixed train/validation/test split (proposal 3.9.1)."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Average and spread of each colour channel (R, G, B) over the CIFAR-10
# training set. Fixed values, rather than recomputing them during training,
# mean a poisoning attack can't shift them (proposal 2.4.2).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def cifar10_transform(augment: bool = False) -> transforms.Compose:
    """Turn a PIL image into a normalised tensor of shape (3, 32, 32).

    With `augment=True`, each image is also randomly shifted and mirrored
    every time it is loaded, so the model sees a slightly different version
    each epoch. Use it for training data only, never for val or test.
    """
    steps = []
    if augment:
        steps += [
            transforms.RandomCrop(32, padding=4),  # pad 4px, cut a random 32x32 = small shift
            transforms.RandomHorizontalFlip(),  # mirror left-right half of the time
        ]
    steps += [
        transforms.ToTensor(),  # pixels 0-255 -> floats 0.0-1.0
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(steps)


def normalise(raw: np.ndarray) -> torch.Tensor:
    """Raw pixels (N, H, W, 3), 0-255 -> normalised tensors (N, 3, H, W), like the transform."""
    images = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle 0..n-1 with `seed` and cut it into (train, val) index lists.

    The same seed always gives the same split, so every experiment trains
    and validates on exactly the same images.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()
    return order[val_size:], order[:val_size]


def get_cifar10_datasets(
    data_dir: str | Path = "data",
    val_size: int = 5000,
    seed: int = 42,
    augment: bool = True,
) -> tuple[Subset, Subset, datasets.CIFAR10]:
    """Return the (train, val, test) image sets, downloading CIFAR-10 if needed.

    The 50,000 official training images are split into train and val. The
    official 10,000 test images are kept separate and untouched. Only the
    train set is augmented; val and test always give the original images.

    Train and val are `Subset`s: a list of positions (`.indices`) into a full
    CIFAR-10 object (`.dataset`) that holds the raw pixels (`.data`) and
    labels (`.targets`). Attacks edit those raw pixels and labels.
    """
    plain = cifar10_transform(augment=False)
    train_view = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=cifar10_transform(augment)
    )
    val_view = datasets.CIFAR10(data_dir, train=True, download=True, transform=plain)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=plain)

    train_idx, val_idx = split_indices(len(train_view), val_size, seed)
    return Subset(train_view, train_idx), Subset(val_view, val_idx), test_set


def make_loaders(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int = 128,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap the three image sets in loaders that hand out batches."""
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # new order every epoch, repeatable thanks to the generator
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_size: int = 5000,
    seed: int = 42,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shortcut: clean CIFAR-10 image sets, already wrapped in loaders."""
    train_set, val_set, test_set = get_cifar10_datasets(data_dir, val_size, seed, augment)
    return make_loaders(train_set, val_set, test_set, batch_size, seed)
