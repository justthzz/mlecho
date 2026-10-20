import numpy as np
import torch
from torch import nn

from mlecho.attacks.gradient_matching import (
    GradientMatchingConfig,
    craft_gradient_matching,
    write_poisons,
)
from mlecho.attacks.gradients import cosine, trainable

MEAN, STD = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
CONFIG = GradientMatchingConfig(epsilon=16 / 255, steps=30, max_shift=1)


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


def _batch(n: int, label: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, 8, 8, generator=generator)  # pixels in [0, 1]
    return (images - 0.5) / 0.25, torch.full((n,), label)


POISONS = _batch(8, label=6, seed=1)  # e.g. frogs, with their true label
TARGET = _batch(1, label=6, seed=2)  # one target image, with the wrong label wanted


def _craft(model: nn.Module, config=CONFIG) -> torch.Tensor:
    return craft_gradient_matching(
        model,
        nn.CrossEntropyLoss(),
        POISONS,
        TARGET,
        config,
        MEAN,
        STD,
        torch.Generator().manual_seed(0),
    )


def _match(model: nn.Module, images: torch.Tensor) -> float:
    """Cosine between the poisons' gradient and the target's gradient."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    params = trainable(model)
    poison_grad = torch.autograd.grad(loss_fn(model(images), POISONS[1]), params)
    target_grad = torch.autograd.grad(loss_fn(model(TARGET[0]), TARGET[1]), params)
    return cosine(poison_grad, target_grad).item()


def test_poisons_change_but_stay_within_epsilon_and_valid_pixels():
    crafted = _craft(_tiny_model())

    pixel_change = (crafted - POISONS[0]) * 0.25  # back to pixel units
    assert not torch.equal(crafted, POISONS[0])
    assert pixel_change.abs().max() <= CONFIG.epsilon + 1e-6
    pixels = crafted * 0.25 + 0.5
    assert pixels.min() >= -1e-6 and pixels.max() <= 1 + 1e-6


def test_poison_gradient_matches_the_target_better_after_crafting():
    model = _tiny_model()
    no_augment = GradientMatchingConfig(epsilon=16 / 255, steps=30, augment=False)
    crafted = _craft(model, no_augment)
    assert _match(model, crafted) > _match(model, POISONS[0])


def test_crafting_does_not_change_the_model():
    model = _tiny_model()
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    _craft(model)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key]), key
    assert model.training  # mode restored


def test_zero_epsilon_returns_the_original_images():
    crafted = _craft(_tiny_model(), GradientMatchingConfig(epsilon=0, steps=5))
    assert torch.allclose(crafted, POISONS[0], atol=1e-6)


class _FakeDataset:
    def __init__(self) -> None:
        self.data = np.zeros((5, 8, 8, 3), dtype=np.uint8)
        self.targets = [6] * 5


def test_write_poisons_stores_rounded_pixels_at_the_given_positions():
    dataset = _FakeDataset()
    images = (torch.full((2, 3, 8, 8), 0.2) - 0.5) / 0.25  # pixel value 0.2 everywhere
    write_poisons(dataset, [1, 3], images, MEAN, STD)

    assert (dataset.data[[1, 3]] == round(0.2 * 255)).all()
    assert (dataset.data[[0, 2, 4]] == 0).all()  # other images untouched
    assert dataset.targets == [6] * 5  # labels untouched (clean-label)
