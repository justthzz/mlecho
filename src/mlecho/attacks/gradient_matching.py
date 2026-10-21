"""Gradient-matching poisoning ("Witches' Brew", Geiping et al., ICLR 2021).

A clean-label, targeted attack. The attacker picks one target image (e.g. one
particular dog photo) and a wrong class for it (e.g. frog). They then slightly
change a few training images that really are frogs, keeping their correct
labels, so that learning from them pushes the weights the same way as learning
"this dog is a frog" would. After normal training on the poisoned set, the
model calls the target a frog and behaves normally on everything else.

Crafting (the paper's Algorithm 1, with one restart instead of eight):
    target_grad = grad L(f(x_target), wrong_class)
    minimise  1 - cos( grad L(f(x_poison + delta), y_poison), target_grad )
over delta with signed Adam, keeping every pixel change within epsilon. The
poisons are randomly mirrored and shifted at each step, so they still work
when the victim's training augments its images.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mlecho.attacks.base import ImageDataset
from mlecho.attacks.gradients import cosine, trainable


@dataclass(frozen=True)
class GradientMatchingConfig:
    """Settings for crafting one set of gradient-matching poisons."""

    epsilon: float = 16 / 255  # largest change per pixel (pixels scaled 0-1), as in the paper
    steps: int = 240  # optimisation steps, as in the paper
    augment: bool = True  # mirror and shift the poisons while crafting
    max_shift: int = 4  # pixels; matches the training augmentation (RandomCrop padding 4)

    @property
    def step_size(self) -> float:
        """Starting step size; divided by 10 at 3/8, 5/8 and 7/8 of the steps."""
        return self.epsilon / 10


def craft_gradient_matching(
    model: nn.Module,
    loss_fn: nn.Module,
    poisons: tuple[torch.Tensor, torch.Tensor],
    target: tuple[torch.Tensor, torch.Tensor],
    config: GradientMatchingConfig,
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
    generator: torch.Generator,
) -> torch.Tensor:
    """Return slightly changed copies of the (normalised) poison images.

    `poisons` are (images, their true labels); the labels are never changed.
    `target` is (target image(s), the wrong label the attacker wants for them).
    `pixel_mean` and `pixel_std` describe how the images were normalised, so
    epsilon and the valid range [0, 1] apply in real pixel units. The model is
    used in eval mode and is not changed.
    """
    images, labels = poisons
    params = trainable(model)
    std = torch.tensor(pixel_std, device=images.device).view(1, -1, 1, 1)
    mean = torch.tensor(pixel_mean, device=images.device).view(1, -1, 1, 1)
    pixels = images * std + mean  # back to real pixels, 0-1

    was_training = model.training
    model.eval()  # fixed BatchNorm statistics: crafting must not change the model
    try:
        target_grad = [
            g.detach() for g in torch.autograd.grad(loss_fn(model(target[0]), target[1]), params)
        ]

        delta = torch.zeros_like(pixels, requires_grad=True)  # the change, in pixel units
        optimizer = torch.optim.Adam([delta], lr=config.step_size)
        milestones = [config.steps * k // 8 for k in (3, 5, 7)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

        for _ in range(config.steps):
            poisoned = pixels + delta
            if config.augment:
                poisoned = _flip_and_shift(poisoned, config.max_shift, generator)
            poison_grad = torch.autograd.grad(
                loss_fn(model((poisoned - mean) / std), labels), params, create_graph=True
            )
            mismatch = 1 - cosine(poison_grad, target_grad)
            (pixel_grad,) = torch.autograd.grad(mismatch, delta)
            delta.grad = pixel_grad.sign()  # "signed Adam": only each pixel's direction counts
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                delta.clamp_(-config.epsilon, config.epsilon)  # stay within epsilon
                delta.copy_((pixels + delta).clamp(0, 1) - pixels)  # stay valid pixels
    finally:
        model.train(was_training)

    return ((pixels + delta - mean) / std).detach()


def choose_poisons(
    dataset: ImageDataset,
    candidates: Sequence[int],
    poison_class: int,
    count: int,
    seed: int,
) -> np.ndarray:
    """Pick `count` positions (sorted) among `candidates` whose label is `poison_class`.

    These are the images the attacker will change. Their labels already are the
    poison class and stay that way (clean-label).
    """
    pool = [i for i in candidates if dataset.targets[i] == poison_class]
    return np.sort(np.random.default_rng(seed).choice(pool, count, replace=False))


def read_images(
    dataset: ImageDataset,
    positions: Sequence[int],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
) -> torch.Tensor:
    """Raw pixels at `positions` as normalised tensors (N, 3, H, W): the opposite of
    `write_poisons`."""
    raw = torch.from_numpy(dataset.data[np.asarray(positions)]).permute(0, 3, 1, 2)
    std = torch.tensor(pixel_std).view(1, -1, 1, 1)
    mean = torch.tensor(pixel_mean).view(1, -1, 1, 1)
    return (raw.float() / 255 - mean) / std


def write_poisons(
    dataset: ImageDataset,
    positions: Sequence[int],
    images: torch.Tensor,
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
) -> None:
    """Store crafted (normalised) images back into the dataset's raw pixels.

    Pixels are rounded to whole numbers 0-255, as when saving a real image file.
    """
    std = torch.tensor(pixel_std).view(1, -1, 1, 1)
    mean = torch.tensor(pixel_mean).view(1, -1, 1, 1)
    pixels = (images.detach().cpu() * std + mean).clamp(0, 1)
    raw = (pixels * 255).round().to(torch.uint8).permute(0, 2, 3, 1)  # (N, H, W, 3)
    dataset.data[np.asarray(positions)] = raw.numpy()


def _flip_and_shift(images: torch.Tensor, max_shift: int, generator: torch.Generator):
    """Mirror a random half of the images and shift the batch by up to `max_shift` pixels.

    Built from plain tensor operations, so gradients flow back to the pixels.
    The empty border is filled with black, like RandomCrop's padding.
    """
    flip = (torch.rand(len(images), generator=generator) < 0.5).to(images.device)
    images = torch.where(flip.view(-1, 1, 1, 1), images.flip(3), images)
    dy, dx = (torch.randint(-max_shift, max_shift + 1, (2,), generator=generator)).tolist()
    height, width = images.shape[2:]
    padded = F.pad(images, [max_shift] * 4)
    top, left = max_shift + dy, max_shift + dx
    return padded[:, :, top : top + height, left : left + width]
