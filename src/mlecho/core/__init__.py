"""Shared building blocks used by every other part of MLEcho."""

from mlecho.core.device import get_device
from mlecho.core.reproducibility import seed_everything

__all__ = ["get_device", "seed_everything"]
