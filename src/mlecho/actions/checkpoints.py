"""Checkpoints of the model being protected, so the guard can go back in time.

A copy of the model and optimizer is kept after every window, marked clean
when the monitor saw nothing wrong in it. Rollback and retrain restore one of
these copies. Kept in memory (on the CPU); only the newest `keep` copies plus
the very first one are kept.
"""

import copy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Checkpoint:
    batch: int  # number of batches trained on when it was taken
    clean: bool  # the window just before it looked clean
    model: dict
    optimizer: dict


def _cpu_copy(value):
    """A copy of a (nested) state dict with every tensor moved to the CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


class CheckpointManager:
    """Saves and restores (model, optimizer) states."""

    def __init__(self, keep: int = 50):
        self.keep = keep
        self.saved: list[Checkpoint] = []

    def save(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, batch: int, clean: bool
    ) -> None:
        self.saved.append(
            Checkpoint(
                batch, clean, _cpu_copy(model.state_dict()), _cpu_copy(optimizer.state_dict())
            )
        )
        if len(self.saved) > self.keep + 1:  # always keep the first (the starting model)
            del self.saved[1]

    def last_clean(self, before: int) -> Checkpoint:
        """Newest clean checkpoint taken at or before batch `before` (else the first)."""
        for checkpoint in reversed(self.saved):
            if checkpoint.clean and checkpoint.batch <= before:
                return checkpoint
        return self.saved[0]

    def oldest_clean(self) -> Checkpoint:
        return next((c for c in self.saved if c.clean), self.saved[0])

    @staticmethod
    def restore(checkpoint: Checkpoint, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Put the saved weights and optimizer state back (onto the model's device)."""
        model.load_state_dict(checkpoint.model)
        optimizer.load_state_dict(checkpoint.optimizer)
