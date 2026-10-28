"""Training-stream episodes: a live training run where an attack starts part-way.

The model learns online, one update per incoming batch (`online.py`). For the
first `onset` batches the stream is clean; after that it comes from the
scenario's (poisoned) data. The guard's monitor scores every batch; after
every window the policy chooses an action, which is carried out here:
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

from mlecho.actions.checkpoints import CheckpointManager
from mlecho.actions.quarantine import quarantine_mask
from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.bench.datasets import normalise
from mlecho.bench.online import online_step
from mlecho.classification.monitor import Monitor
from mlecho.response.base import Action, ResponsePolicy, State


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
    manager = CheckpointManager()
    manager.save(model, optimizer, batch=0, clean=True)
    log = EpisodeLog(curve=[{"batch": 0, **measure(model)}])

    order = rng.permutation(len(clean_labels))
    position = 0
    quarantine = None  # (attack type, pair) once a quarantine is in force
    streak, incident_start, clean_streak = 0, 0, 0
    in_incident = acted = escalated = False
    last_type, losses, previous_loss = "clean", [], None

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
        evidence = monitor.evidence(images)
        keep = torch.ones(len(rows), dtype=torch.bool)
        if quarantine is not None:
            keep = ~quarantine_mask(evidence, targets, monitor.calibration, *quarantine)
        kept_poison = poisoned & keep.numpy()
        log.poison_trained += int(kept_poison.sum())
        log.poison_dropped += int((poisoned & ~keep.numpy()).sum())
        log.clean_dropped += int((~poisoned & ~keep.numpy()).sum())
        if keep.sum() > 1:
            losses.append(
                online_step(
                    model, optimizer, loss_fn, images[keep.to(device)], targets[keep].to(device)
                )
            )

        verdict = monitor.add(evidence, targets)
        if verdict is not None:
            if verdict.attack_type == "clean":
                streak, clean_streak = 0, clean_streak + 1
                if clean_streak >= 2:  # two clean windows in a row: the incident is over
                    in_incident = acted = escalated = False
            else:
                clean_streak = 0
                streak = streak + 1 if verdict.attack_type == last_type else 1
                if not in_incident:  # first suspicious window: remember where it began
                    in_incident, incident_start = True, b + 1 - monitor.window
            last_type = verdict.attack_type
            mean_loss = float(np.mean(losses)) if losses else 0.0
            state = State(
                attack_type=verdict.attack_type,
                confidence=verdict.confidence,
                probabilities=verdict.probabilities,
                streak=streak,
                acted=acted,
                escalated=escalated,
                loss_change=0.0 if previous_loss is None else mean_loss - previous_loss,
            )
            previous_loss, losses = mean_loss, []
            action = policy.decide(state) if policy is not None else Action.CONTINUE
            if action in (Action.ROLLBACK, Action.RETRAIN):
                checkpoint = (
                    manager.last_clean(before=incident_start - monitor.window)
                    if action == Action.ROLLBACK
                    else manager.oldest_clean()
                )
                manager.restore(checkpoint, model, optimizer)
            if action in (Action.QUARANTINE, Action.ROLLBACK, Action.RETRAIN):
                quarantine, acted = (verdict.attack_type, verdict.pair), True
            if action == Action.ESCALATE:
                escalated = True
            if action != Action.CONTINUE:
                log.actions.append((b + 1, action.value))
            manager.save(model, optimizer, batch=b + 1, clean=verdict.attack_type == "clean")
            log.windows.append(
                {
                    "batch": b + 1,
                    "type": verdict.attack_type,
                    "confidence": verdict.confidence,
                    "action": action.value,
                }
            )
        if (b + 1) % measure_every == 0:
            log.curve.append({"batch": b + 1, **measure(model)})
    return log
