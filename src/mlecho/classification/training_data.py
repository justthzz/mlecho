"""Made-up windows for training the attack classifier (step 4, used by `Guard.calibrate`).

Each made-up window is built from trusted images only: a resampled clean
stream (`stream_rows`) into which one made-up attack or harmless change from
`synthetic.py` is mixed. The evidence of every changed image is computed once
per setting (`made_up_settings`), so building thousands of windows is fast.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.classification.classifier import ATTACK_TYPES
from mlecho.classification.features import Calibration, stream_rows, window_features
from mlecho.classification.monitor import Evidence
from mlecho.classification.synthetic import (
    Shift,
    apply_shift,
    random_backdoor,
    random_flip,
    random_shift,
)

Scorer = Callable[[np.ndarray], Evidence]  # raw trusted images -> their out-of-fold evidence


@dataclass(frozen=True)
class MadeUpSettings:
    """Random triggers and shifts, each with the evidence of all trusted images
    once changed; `excluded` flip pairs are kept out (reserved for testing)."""

    triggers: list[tuple[BackdoorConfig, Evidence]]
    shifts: list[tuple[Shift, Evidence]]
    excluded: set[tuple[int, int]]
    num_classes: int


def made_up_settings(
    score: Scorer,
    raw: np.ndarray,
    num_classes: int,
    count: int,
    rng: np.random.Generator,
    excluded: set[tuple[int, int]] = frozenset(),
) -> MadeUpSettings:
    """Draw `count` random triggers and `count` harmless shifts, and score them."""
    triggers = []
    for _ in range(count):
        config = random_backdoor(rng, raw.shape[1], num_classes)
        triggers.append((config, score(add_trigger(raw, config))))
    shifts = []
    for _ in range(count):
        shift = random_shift(rng)
        shifts.append((shift, score(apply_shift(raw, shift))))
    return MadeUpSettings(triggers, shifts, excluded, num_classes)


def overlay(base: Evidence, poison: Evidence, mask: torch.Tensor, rows: torch.Tensor) -> Evidence:
    """Evidence of the images at `rows`, with those where `mask` is True replaced by
    their changed version (`poison`, indexed like `base`)."""
    picture, log_probs = base.picture[rows].clone(), base.log_probs[rows].clone()
    picture[mask], log_probs[mask] = poison.picture[rows[mask]], poison.log_probs[rows[mask]]
    return Evidence(picture, log_probs)


def made_up_window(
    kind: str,
    trusted: Evidence,
    labels: torch.Tensor,
    settings: MadeUpSettings,
    calibration: Calibration,
    batches: int,
    batch_size: int,
    rng: np.random.Generator,
    generator: torch.Generator,
) -> torch.Tensor:
    """The features of one made-up window of type `kind` (one of ATTACK_TYPES)."""
    size = batches * batch_size
    rows = stream_rows(len(labels), size, generator)
    lab = labels[rows].clone()
    evidence = Evidence(trusted.picture[rows], trusted.log_probs[rows])
    if kind == "backdoor":  # a share of the dataset carries the trigger
        config, stamped = settings.triggers[rng.integers(len(settings.triggers))]
        mask = (torch.rand(size, generator=generator) < rng.uniform(0.02, 0.10)) & (
            lab != config.target_class
        )
        evidence, lab[mask] = overlay(trusted, stamped, mask, rows), config.target_class
    elif kind == "accumulative":  # a fixed few triggered images in every batch
        config, stamped = settings.triggers[rng.integers(len(settings.triggers))]
        mask = torch.zeros(batches, batch_size, dtype=torch.bool)
        mask[:, : rng.integers(1, 7)] = True
        mask = mask.flatten()
        others = torch.as_tensor(np.flatnonzero(labels.numpy() != config.target_class))
        rows = rows.clone()
        rows[mask] = others[torch.randint(len(others), (int(mask.sum()),), generator=generator)]
        evidence, lab = overlay(trusted, stamped, mask, rows), labels[rows].clone()
        lab[mask] = config.target_class
    elif kind == "label_flip":  # some images of one class get another class's label
        source, target = random_flip(rng, settings.num_classes, settings.excluded)
        share = rng.uniform(0.01, 0.05) / (labels == source).float().mean().item()
        lab[(lab == source) & (torch.rand(size, generator=generator) < share)] = target
    elif kind == "benign_shift":  # part or all of the data changed in a harmless way
        _, changed = settings.shifts[rng.integers(len(settings.shifts))]
        mask = torch.rand(size, generator=generator) < rng.uniform(0.3, 1.0)
        evidence = overlay(trusted, changed, mask, rows)
    return window_features(
        evidence.picture.view(batches, -1),
        evidence.log_probs.view(batches, -1, evidence.log_probs.shape[1]),
        lab.view(batches, -1),
        calibration,
    )


def made_up_set(
    trusted: Evidence,
    labels: torch.Tensor,
    settings: MadeUpSettings,
    calibration: Calibration,
    windows: int,
    batches: int,
    batch_size: int,
    rng: np.random.Generator,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`windows` made-up windows of every type -> (features, type numbers)."""
    x, y = [], []
    for t, kind in enumerate(ATTACK_TYPES):
        for _ in range(windows):
            x.append(
                made_up_window(
                    kind,
                    trusted,
                    labels,
                    settings,
                    calibration,
                    batches,
                    batch_size,
                    rng,
                    generator,
                )
            )
            y.append(t)
    return torch.stack(x), torch.tensor(y)
