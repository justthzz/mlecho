import torch
from torch import nn

from mlecho.core import get_device, load_weights, save_weights, seed_everything


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


def test_saved_weights_load_back_identically(tmp_path):
    original = nn.Linear(4, 2)
    path = tmp_path / "sub" / "weights.pt"  # folder is created automatically
    save_weights(original, path)

    copy = load_weights(nn.Linear(4, 2), path, torch.device("cpu"))
    assert torch.equal(copy.weight, original.weight)
    assert torch.equal(copy.bias, original.bias)
