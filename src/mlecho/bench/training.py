"""Train and evaluate a classifier, one epoch at a time."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochResult:
    """Average loss and accuracy over one pass of a data loader."""

    loss: float
    accuracy: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochResult:
    """Show the model every training batch once, updating its weights."""
    model.train()  # turns dropout on
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)  # 1. forward: a score per class
        loss = loss_fn(logits, labels)  # 2. loss: how wrong were we?
        optimizer.zero_grad()  # (clear the previous batch's directions)
        loss.backward()  # 3. backward: which way to nudge each weight
        optimizer.step()  # 4. step: nudge the weights

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return EpochResult(loss=total_loss / seen, accuracy=correct / seen)


@torch.no_grad()  # measuring only, so skip the gradient bookkeeping
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> EpochResult:
    """Measure loss and accuracy without changing the model."""
    model.eval()  # turns dropout off
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return EpochResult(loss=total_loss / seen, accuracy=correct / seen)
