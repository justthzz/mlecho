"""Targeted label-flip attack.

Some training images of one class (the "source", e.g. cat) keep their pixels
but get the label of another class (the "target", e.g. dog). Nothing looks
wrong in the images themselves; only the labels lie. The model learns to
confuse the two classes.
"""

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Subset


@dataclass(frozen=True)
class LabelFlipConfig:
    """Settings for one targeted label-flip attack."""

    source_class: int = 3  # 3 = cat in CIFAR-10
    target_class: int = 5  # 5 = dog
    poison_rate: float = 0.05  # share of the whole training set (same budget as the backdoor)


def flip_labels(train_set: Subset, config: LabelFlipConfig, seed: int) -> list[int]:
    """Relabel a random share of the source-class training images, in place.

    `train_set` is a `Subset` whose `.dataset` is an `ImageDataset`. Only
    images inside `train_set` (never val) whose label is the source class are
    chosen; their pixels are not touched. Returns their positions in the full
    dataset, so later steps know exactly which images were poisoned.
    """
    full = train_set.dataset
    labels = np.asarray(full.targets)
    candidates = [i for i in train_set.indices if labels[i] == config.source_class]
    count = round(config.poison_rate * len(train_set))
    if count > len(candidates):
        raise ValueError(
            f"poison_rate needs {count} images but the source class has only {len(candidates)}"
        )

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(candidates, size=count, replace=False))
    for i in chosen:
        full.targets[i] = config.target_class
    return chosen.tolist()
