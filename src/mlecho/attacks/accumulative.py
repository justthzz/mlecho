"""Accumulative poisoning attack in online learning (Pang et al., NeurIPS 2021).

Algorithm 1 of the paper, with a clean trigger batch. While a model learns
from a live stream (one update per incoming batch), the attacker slightly
perturbs each incoming batch so that the model:
  (a) keeps its normal accuracy (the attack stays secret), and
  (b) becomes more and more sensitive to one future "trigger" batch.
When the trigger batch finally arrives, a single update step breaks the
model's accuracy.

Per incoming batch S_t, the attacker aligns the update on the crafted batch
with a target direction (paper, Eq. 9):
    target = unit(grad L(S_t))  +  lambda * unit(G)
with G = grad( grad L(S_val) . grad L(S_trigger) ), using a few steps of
projected gradient ascent on the pixels, within a small per-pixel bound.

Both parts of the target are normalised to unit length (the paper's optional
"normalise the gradients" step). Without it, on our model G was 3.5x larger
than the clean gradient, the crafted batches became adversarial (loss
0.4 -> 8.3, update 29x larger) and accuracy broke before the trigger (see
docs/ROADMAP.md). By default the attacker maximises the cosine between the
crafted gradient and the target (direction only); `objective="dot"` keeps the
paper's raw dot product (direction and size).
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class AccumulativeConfig:
    """Settings for crafting one accumulative batch."""

    epsilon: float = 8 / 255  # largest change per pixel (pixels scaled 0-1)
    pgd_steps: int = 10  # nudges per batch (the paper uses 100)
    trade_off: float = 0.02  # lambda: weight of "make the trigger harmful" vs "keep accuracy"
    objective: Literal["cosine", "dot"] = "cosine"  # align direction only, or direction and size

    @property
    def step_size(self) -> float:
        """Size of one nudge (the paper's choice: 2 * epsilon / steps)."""
        return 2 * self.epsilon / self.pgd_steps


Batch = tuple[torch.Tensor, torch.Tensor]  # (images, labels)


def trigger_sensitivity(
    model: nn.Module, loss_fn: nn.Module, val_batch: Batch, trigger_batch: Batch
) -> list[torch.Tensor]:
    """G: the direction in weight space that makes the trigger batch most harmful.

    It is the gradient of (validation gradient . trigger gradient). Moving the
    weights along -G makes one future step on the trigger batch increase the
    validation loss the most.
    """
    params = _trainable(model)
    with _keep_buffers(model):
        val_grad = torch.autograd.grad(
            loss_fn(model(val_batch[0]), val_batch[1]), params, create_graph=True
        )
        trigger_grad = torch.autograd.grad(
            loss_fn(model(trigger_batch[0]), trigger_batch[1]), params, create_graph=True
        )
        alignment = _dot(val_grad, trigger_grad)
        return [g.detach() for g in torch.autograd.grad(alignment, params)]


def craft_accumulative_batch(
    model: nn.Module,
    loss_fn: nn.Module,
    batch: Batch,
    sensitivity: Sequence[torch.Tensor],
    config: AccumulativeConfig,
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
) -> torch.Tensor:
    """Return a slightly perturbed copy of the incoming (normalised) images.

    `pixel_mean` and `pixel_std` describe how the images were normalised, so
    the per-pixel bound `epsilon` and the valid range [0, 1] are applied in
    real pixel units. Labels are never changed. The model is not updated.
    """
    images, labels = batch
    params = _trainable(model)
    std = torch.tensor(pixel_std, device=images.device).view(1, -1, 1, 1)
    mean = torch.tensor(pixel_mean, device=images.device).view(1, -1, 1, 1)
    bound, step = config.epsilon / std, config.step_size / std
    lowest, highest = (0 - mean) / std, (1 - mean) / std  # valid pixels, normalised

    with _keep_buffers(model):
        clean_grad = torch.autograd.grad(loss_fn(model(images), labels), params)
        target = [
            c + config.trade_off * g
            for c, g in zip(_unit(clean_grad), _unit(sensitivity), strict=True)
        ]

        delta = torch.zeros_like(images)
        for _ in range(config.pgd_steps):
            delta.requires_grad_(True)
            poisoned_grad = torch.autograd.grad(
                loss_fn(model(images + delta), labels), params, create_graph=True
            )
            alignment = _dot(poisoned_grad, target)
            if config.objective == "cosine":
                alignment = alignment / _norm(poisoned_grad)  # direction only, up to scale
            (pixel_grad,) = torch.autograd.grad(alignment, delta)
            delta = (delta + step * pixel_grad.sign()).detach()
            delta = torch.max(torch.min(delta, bound), -bound)  # stay within epsilon
            delta = (images + delta).clamp(lowest, highest) - images  # stay valid pixels

    return (images + delta).detach()


def _trainable(model: nn.Module) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _dot(first: Sequence[torch.Tensor], second: Sequence[torch.Tensor]) -> torch.Tensor:
    """Dot product of two gradients, each given as one tensor per weight group."""
    return sum((a * b).sum() for a, b in zip(first, second, strict=True))


def _norm(gradient: Sequence[torch.Tensor]) -> torch.Tensor:
    """Length of a gradient (all weight groups together)."""
    return torch.sqrt(_dot(gradient, gradient)) + 1e-12


def _unit(gradient: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """The same direction, scaled to length 1."""
    length = _norm(gradient)
    return [g / length for g in gradient]


@contextmanager
def _keep_buffers(model: nn.Module) -> Iterator[None]:
    """Undo any change to BatchNorm running statistics made inside the block.

    The attacker's practice forward passes must not change the model; only the
    real update step on the crafted batch should.
    """
    saved = [b.detach().clone() for b in model.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, value in zip(model.buffers(), saved, strict=True):
                buffer.copy_(value)
