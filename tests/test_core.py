import torch

from mlecho.core import get_device, seed_everything


def test_same_seed_gives_same_random_numbers():
    seed_everything(123)
    first = torch.rand(5)
    seed_everything(123)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_different_seeds_give_different_random_numbers():
    seed_everything(1)
    first = torch.rand(5)
    seed_everything(2)
    second = torch.rand(5)
    assert not torch.equal(first, second)


def test_get_device_returns_a_usable_device():
    device = get_device()
    assert torch.zeros(1, device=device).device.type == device.type
