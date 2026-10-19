"""Online (real-time) learning: the model takes one update step per incoming batch.

This simulates a deployed model that keeps learning from a live data stream,
the setting of the accumulative poisoning attack (Pang et al., 2021).
"""

import torch
from torch import nn


def online_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Learn from one incoming batch with a single update step. Returns the loss."""
    model.train()
    loss = loss_fn(model(images), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
