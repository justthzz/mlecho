"""Training-stream episodes: a live training run where an attack starts part-way.

The model learns online, one update per incoming batch (`online.py`). For the
first `onset` batches the stream is clean; after that it comes from the
scenario's (poisoned) data. The guard's monitor scores every batch; after
every window the policy chooses an action, which the `Guard` carries out:
quarantine filters later batches image by image, rollback and retrain
restore a checkpoint. With `policy=None` nothing is done (the no-guard
control), but the monitor's verdicts are still logged.

This is the lab's practice world: experiment 15 compares policies with it,
and the PPO environment (step 7) is built on it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.bench.baselines.response import NoResponse
from mlecho.bench.datasets import normalise
from mlecho.bench.online import online_step
from mlecho.classification.monitor import Monitor
from mlecho.guard import Guard
from mlecho.response.base import Action, ResponsePolicy
from mlecho.tracking.audit import AuditLog


@dataclass(frozen=True)
class Scenario:
    """The training data after the onset. Row k is training image k."""

    name: str
    raw: np.ndarray  # pixels after the onset, (N, H, W, 3) 0-255
    labels: np.ndarray  # labels after the onset
    poisoned: np.ndarray  # True for rows that are poisoned after the onset
    drip: int = 0  # slow-drip: this many triggered images in every batch
    trigger: BackdoorConfig = field(default_factory=BackdoorConfig)  # used by the drip


@dataclass
class EpisodeLog:
    """Everything that happened in one episode."""

    windows: list[dict] = field(default_factory=list)  # one row per window
    curve: list[dict] = field(default_factory=list)  # `measure` results over time
    actions: list[tuple[int, str]] = field(default_factory=list)  # (batch, action), not continue
    poison_trained: int = 0  # poisoned images the model actually learned from
    poison_dropped: int = 0  # poisoned images left out by quarantine
    clean_dropped: int = 0  # clean images left out by quarantine


def _drip(raw, labels, poisoned, count, trigger, rng):
    """Put the trigger on `count` images of the batch not already in the target class."""
    others = np.flatnonzero(labels != trigger.target_class)
    chosen = rng.choice(others, size=min(count, len(others)), replace=False)
    raw, labels, poisoned = raw.copy(), labels.copy(), poisoned.copy()
    raw[chosen] = add_trigger(raw[chosen], trigger)
    labels[chosen], poisoned[chosen] = trigger.target_class, True
    return raw, labels, poisoned


def run_episode(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    clean_raw: np.ndarray,
    clean_labels: np.ndarray,
    scenario: Scenario,
    monitor: Monitor,
    policy: ResponsePolicy | None,
    batches: int,
    onset: int,
    batch_size: int,
    measure: Callable[[nn.Module], dict],
    measure_every: int,
    device: torch.device,
    seed: int,
) -> EpisodeLog:
    """Run one episode and return its log (see the module docstring)."""
    rng = np.random.default_rng(seed)
    loss_fn = nn.CrossEntropyLoss()
    audit = AuditLog()
    guard = Guard(monitor, policy if policy is not None else NoResponse(), [audit])
    guard.watch(model, optimizer)
    log = EpisodeLog(curve=[{"batch": 0, **measure(model)}])

    order = rng.permutation(len(clean_labels))
    position = 0
    for b in range(batches):
        if position + batch_size > len(order):
            order, position = rng.permutation(len(clean_labels)), 0
        rows = order[position : position + batch_size]
        position += batch_size
        if b < onset:
            raw, labels = clean_raw[rows], clean_labels[rows]
            poisoned = np.zeros(len(rows), dtype=bool)
        else:
            raw, labels, poisoned = (
                scenario.raw[rows],
                scenario.labels[rows],
                scenario.poisoned[rows],
            )
            if scenario.drip:
                raw, labels, poisoned = _drip(
                    raw, labels, poisoned, scenario.drip, scenario.trigger, rng
                )

        images, targets = normalise(raw).to(device), torch.as_tensor(labels)
        keep = guard.check(images, targets)
        kept = keep.numpy()
        log.poison_trained += int((poisoned & kept).sum())
        log.poison_dropped += int((poisoned & ~kept).sum())
        log.clean_dropped += int((~poisoned & ~kept).sum())
        loss = None
        if keep.sum() > 1:
            loss = online_step(
                model, optimizer, loss_fn, images[keep.to(device)], targets[keep].to(device)
            )
        action = guard.update(loss)
        if action is not None:
            event = audit.events[-1]
            if action != Action.CONTINUE:
                log.actions.append((b + 1, action.value))
            log.windows.append(
                {
                    "batch": b + 1,
                    "type": event["attack_type"],
                    "confidence": event["confidence"],
                    "action": action.value,
                    "target_label": event["target_label"],
                    "mix_pair": event["mix_pair"],
                }
            )
        if (b + 1) % measure_every == 0:
            log.curve.append({"batch": b + 1, **measure(model)})
    return log
