import torch
from torch import nn

from mlecho.attacks.accumulative import (
    AccumulativeConfig,
    craft_accumulative_batch,
    trigger_sensitivity,
)

MEAN, STD = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
CONFIG = AccumulativeConfig(epsilon=8 / 255, pgd_steps=3, trade_off=0.02)


def _tiny_model() -> nn.Module:
    """A small CNN with BatchNorm (to check its running statistics are kept)."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 10),
    ).train()


def _batch(n: int = 6, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, 8, 8, generator=generator)  # pixels in [0, 1]
    normalised = (images - 0.5) / 0.25
    return normalised, torch.randint(0, 10, (n,), generator=generator)


def _craft(model: nn.Module, batch, config=CONFIG) -> torch.Tensor:
    loss_fn = nn.CrossEntropyLoss()
    sensitivity = trigger_sensitivity(model, loss_fn, _batch(seed=2), _batch(seed=3))
    return craft_accumulative_batch(model, loss_fn, batch, sensitivity, config, MEAN, STD)


def test_sensitivity_has_one_tensor_per_weight_group():
    model = _tiny_model()
    sensitivity = trigger_sensitivity(model, nn.CrossEntropyLoss(), _batch(seed=2), _batch(seed=3))
    shapes = [p.shape for p in model.parameters()]
    assert [g.shape for g in sensitivity] == shapes


def test_crafted_images_change_but_stay_within_epsilon_and_valid_pixels():
    images, labels = _batch()
    crafted = _craft(_tiny_model(), (images, labels))

    pixel_change = (crafted - images) * 0.25  # back to pixel units
    assert not torch.equal(crafted, images)
    assert pixel_change.abs().max() <= CONFIG.epsilon + 1e-6
    pixels = crafted * 0.25 + 0.5
    assert pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6


def test_crafting_does_not_change_the_model():
    model = _tiny_model()
    weights_before = [p.clone() for p in model.parameters()]
    buffers_before = [b.clone() for b in model.buffers()]

    _craft(model, _batch())

    assert all(torch.equal(a, b) for a, b in zip(weights_before, model.parameters(), strict=True))
    assert all(torch.equal(a, b) for a, b in zip(buffers_before, model.buffers(), strict=True))


def test_zero_epsilon_returns_the_images_unchanged():
    images, labels = _batch()
    crafted = _craft(_tiny_model(), (images, labels), AccumulativeConfig(epsilon=0.0, pgd_steps=2))
    assert torch.equal(crafted, images)


def test_dot_objective_also_respects_the_pixel_bound():
    images, labels = _batch()
    config = AccumulativeConfig(epsilon=4 / 255, pgd_steps=3, objective="dot")
    crafted = _craft(_tiny_model(), (images, labels), config)
    assert not torch.equal(crafted, images)
    assert ((crafted - images) * 0.25).abs().max() <= config.epsilon + 1e-6
