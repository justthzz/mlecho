"""CIFAR-10 loading with a fixed train/validation/test split (proposal 3.9.1)."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
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


def cifar10_transform() -> transforms.Compose:
    """Turn a PIL image into a normalised tensor of shape (3, 32, 32)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),  # pixels 0-255 -> floats 0.0-1.0
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def split_indices(n: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle 0..n-1 with `seed` and cut it into (train, val) index lists.

    The same seed always gives the same split, so every experiment trains
    and validates on exactly the same images.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()
    return order[val_size:], order[:val_size]


def get_cifar10_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    val_size: int = 5000,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) loaders, downloading CIFAR-10 if needed.

    The 50,000 official training images are split into train and val. The
    official 10,000 test images are kept separate and untouched.
    """
    transform = cifar10_transform()
    full_train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    train_idx, val_idx = split_indices(len(full_train), val_size, seed)
    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=batch_size,
        shuffle=True,  # new order every epoch, repeatable thanks to the generator
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(Subset(full_train, val_idx), batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader
