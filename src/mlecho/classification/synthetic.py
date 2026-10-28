"""Made-up attacks and harmless changes, for teaching the attack classifier.

MLEcho never sees the real attacker's poison before training. So during
calibration it makes its own examples of each attack type on the user's
trusted clean images, with random settings (trigger colour, size and place;
which classes are flipped; harmless photo changes), and the classifier learns
from those. The triggers are stuck on with `mlecho.attacks.backdoor`; the
real attack settings (yellow square, cat -> dog) are only used for testing.

All functions work on raw pixels: shape (N, height, width, 3), values 0-255.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF

from mlecho.attacks.backdoor import BackdoorConfig


def random_backdoor(rng: np.random.Generator, image_size: int, num_classes: int) -> BackdoorConfig:
    """A solid square of random colour, 2-6 pixels wide, at a random place, for a random class."""
    size = int(rng.integers(2, 7))
    row, col = (int(v) for v in rng.integers(0, image_size - size + 1, size=2))
    return BackdoorConfig(
        target_class=int(rng.integers(num_classes)),
        patch_size=size,
        patch_color=tuple(int(c) for c in rng.integers(0, 256, size=3)),
        position=(row, col),
    )


@dataclass(frozen=True)
class Shift:
    """A harmless photo change: one of brightness, contrast or blur."""

    kind: Literal["brightness", "contrast", "blur"]
    amount: float  # brightness/contrast factor (1 = unchanged), or blur sigma


def apply_shift(images: np.ndarray, shift: Shift) -> np.ndarray:
    """Return a copy of `images` with the harmless change applied."""
    pixels = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
    if shift.kind == "brightness":
        out = TF.adjust_brightness(pixels, shift.amount)
    elif shift.kind == "contrast":
        out = TF.adjust_contrast(pixels, shift.amount)
    else:
        out = TF.gaussian_blur(pixels, kernel_size=3, sigma=shift.amount)
    out = out.clamp(0, 1) * 255
    return out.round().byte().permute(0, 2, 3, 1).numpy()


def random_shift(rng: np.random.Generator) -> Shift:
    """Brighter/darker or more/less contrast by 15-50%, or a slight blur."""
    kind = ["brightness", "contrast", "blur"][int(rng.integers(3))]
    if kind == "blur":
        return Shift(kind, float(rng.uniform(0.5, 1.2)))
    change = float(rng.uniform(0.15, 0.5))
    return Shift(kind, 1 + change if rng.random() < 0.5 else 1 - change)


def random_flip(
    rng: np.random.Generator, num_classes: int, exclude: set[tuple[int, int]] = frozenset()
) -> tuple[int, int]:
    """A random (source, target) class pair, not in `exclude` (kept for testing)."""
    while True:
        source, target = (int(c) for c in rng.choice(num_classes, size=2, replace=False))
        if (source, target) not in exclude:
            return source, target
