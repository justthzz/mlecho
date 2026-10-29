"""Practice episodes for the PPO agent: made-up attacks in a training stream.

Like the attack classifier (step 4), the agent never trains on the real
attacks of the evaluation. Each practice episode draws a random attack:
a random trigger (backdoor at a random rate, or slow-drip at a random count
per batch), a random label-flip pair at a random rate, a random harmless
shift, or a clean stream, with a random onset.

The guard scores every incoming batch before anything is filtered, so its
evidence doesn't depend on the actions taken. `cache_episode` therefore
scores a practice stream once and stores, per batch, which training images
came in, how they were changed and their evidence. Replaying an episode then
only needs the victim's training (`env.py`), not the guard's models.
"""

from dataclasses import dataclass

import numpy as np
import torch

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.bench.datasets import normalise
from mlecho.classification.monitor import Evidence, Monitor
from mlecho.classification.synthetic import (
    Shift,
    apply_shift,
    random_backdoor,
    random_flip,
    random_shift,
)

KINDS = ("clean", "backdoor", "accumulative", "label_flip", "benign_shift")


@dataclass(frozen=True)
class PracticeAttack:
    """What happens to the stream from batch `onset` on."""

    kind: str  # one of KINDS
    onset: int
    trigger: BackdoorConfig | None = None  # backdoor, accumulative
    rate: float = 0.0  # backdoor: share of images; label_flip: share of the data; shift: share
    drip: int = 0  # accumulative: triggered images per batch
    flip: tuple[int, int] | None = None  # label_flip: (source, target)
    shift: Shift | None = None  # benign_shift


def random_practice_attack(
    rng: np.random.Generator,
    kind: str,
    image_size: int,
    num_classes: int,
    onsets: tuple[int, int],
    excluded: set[tuple[int, int]] = frozenset(),
) -> PracticeAttack:
    """A random attack of `kind`, starting at a random batch in `onsets` (low, high)."""
    onset = int(rng.integers(*onsets))
    if kind == "backdoor":
        trigger = random_backdoor(rng, image_size, num_classes)
        return PracticeAttack(kind, onset, trigger, rate=float(rng.uniform(0.02, 0.10)))
    if kind == "accumulative":
        trigger = random_backdoor(rng, image_size, num_classes)
        return PracticeAttack(kind, onset, trigger, drip=int(rng.integers(1, 7)))
    if kind == "label_flip":
        flip = random_flip(rng, num_classes, excluded)
        return PracticeAttack(kind, onset, rate=float(rng.uniform(0.01, 0.06)), flip=flip)
    if kind == "benign_shift":
        return PracticeAttack(
            kind, onset, rate=float(rng.uniform(0.3, 1.0)), shift=random_shift(rng)
        )
    return PracticeAttack("clean", onset)


def attack_batch(
    attack: PracticeAttack, labels: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decide what the attack does to one incoming batch after the onset.

    Returns (new labels, poisoned, changed): `changed` marks images whose pixels
    get the trigger or the shift (see `batch_pixels`).
    """
    n = len(labels)
    labels, none = labels.copy(), np.zeros(n, dtype=bool)
    if attack.kind == "backdoor":
        poisoned = (rng.random(n) < attack.rate) & (labels != attack.trigger.target_class)
        labels[poisoned] = attack.trigger.target_class
        return labels, poisoned, poisoned
    if attack.kind == "accumulative":
        others = np.flatnonzero(labels != attack.trigger.target_class)
        chosen = rng.choice(others, size=min(attack.drip, len(others)), replace=False)
        poisoned = none.copy()
        poisoned[chosen] = True
        labels[poisoned] = attack.trigger.target_class
        return labels, poisoned, poisoned
    if attack.kind == "label_flip":
        source, target = attack.flip
        share = attack.rate * 10  # of the source class (10 equal classes)
        poisoned = (labels == source) & (rng.random(n) < share)
        labels[poisoned] = target
        return labels, poisoned, none
    if attack.kind == "benign_shift":
        return labels, none, rng.random(n) < attack.rate  # changed, but not poison
    return labels, none, none


def batch_pixels(attack: PracticeAttack, raw: np.ndarray, changed: np.ndarray) -> np.ndarray:
    """The batch's pixels as they arrive: trigger or shift on the `changed` images."""
    if not changed.any():
        return raw
    raw = raw.copy()
    if attack.trigger is not None:
        raw[changed] = add_trigger(raw[changed], attack.trigger)
    elif attack.shift is not None:
        raw[changed] = apply_shift(raw[changed], attack.shift)
    return raw


@dataclass
class CachedEpisode:
    """One scored practice stream: per batch (row of each array) and image (column)."""

    attack: PracticeAttack
    rows: np.ndarray  # (batches, batch_size): which training images came in
    labels: np.ndarray  # labels as they arrived
    poisoned: np.ndarray  # True for poison
    changed: np.ndarray  # True for images with the trigger or the shift on
    picture: torch.Tensor  # (batches, batch_size) picture scores
    log_probs: torch.Tensor  # (batches, batch_size, classes)

    def evidence(self, batch: int) -> Evidence:
        return Evidence(self.picture[batch], self.log_probs[batch])


def cache_episode(
    attack: PracticeAttack,
    train_raw: np.ndarray,
    train_labels: np.ndarray,
    monitor: Monitor,
    batches: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> CachedEpisode:
    """Build a practice stream (reshuffled clean data, attacked from the onset) and score it."""
    rng = np.random.default_rng(seed)
    rows, labels, poisoned, changed, pictures, log_probs = [], [], [], [], [], []
    order, position = rng.permutation(len(train_labels)), 0
    for b in range(batches):
        if position + batch_size > len(order):
            order, position = rng.permutation(len(train_labels)), 0
        batch_rows = order[position : position + batch_size]
        position += batch_size
        batch_labels = train_labels[batch_rows]
        none = np.zeros(batch_size, dtype=bool)
        if b < attack.onset:
            batch = (batch_labels, none, none)
        else:
            batch = attack_batch(attack, batch_labels, rng)
        pixels = batch_pixels(attack, train_raw[batch_rows], batch[2])
        evidence = monitor.evidence(normalise(pixels).to(device))
        rows.append(batch_rows)
        labels.append(batch[0])
        poisoned.append(batch[1])
        changed.append(batch[2])
        pictures.append(evidence.picture)
        log_probs.append(evidence.log_probs)
    return CachedEpisode(
        attack,
        np.stack(rows),
        np.stack(labels),
        np.stack(poisoned),
        np.stack(changed),
        torch.stack(pictures),
        torch.stack(log_probs),
    )
