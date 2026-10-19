"""Save and load a model's weights (its learned numbers) to and from a file."""

from pathlib import Path

import torch
from torch import nn


def save_weights(model: nn.Module, path: str | Path) -> None:
    """Write the model's weights to `path`, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    """Fill `model` (same architecture) with the weights saved at `path`."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device)
