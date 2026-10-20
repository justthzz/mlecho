"""Small maths helpers for attacks that compare model gradients.

A model's gradient is a list of tensors, one per weight group (layer weights,
layer biases, ...). These helpers treat that list as one long vector, so two
gradients can be compared: are they pointing the same way, and how long are they?
"""

from collections.abc import Sequence

import torch
from torch import nn

Gradient = Sequence[torch.Tensor]  # one tensor per weight group


def trainable(model: nn.Module) -> list[torch.Tensor]:
    """The weights that learning changes (the ones a gradient is taken for)."""
    return [p for p in model.parameters() if p.requires_grad]


def dot(first: Gradient, second: Gradient) -> torch.Tensor:
    """Dot product of two gradients: large when they are long and point the same way."""
    return sum((a * b).sum() for a, b in zip(first, second, strict=True))


def norm(gradient: Gradient) -> torch.Tensor:
    """Length of a gradient (all weight groups together)."""
    return torch.sqrt(dot(gradient, gradient)) + 1e-12


def unit(gradient: Gradient) -> list[torch.Tensor]:
    """The same direction, scaled to length 1."""
    length = norm(gradient)
    return [g / length for g in gradient]


def cosine(first: Gradient, second: Gradient) -> torch.Tensor:
    """How well two gradients point the same way: 1 same, 0 unrelated, -1 opposite."""
    return dot(first, second) / (norm(first) * norm(second))
