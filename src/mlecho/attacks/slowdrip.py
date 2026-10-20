"""Slow-drip poisoning: a few poisoned images in every incoming batch.

A simple accumulative attack for a model that learns from a live stream. No
single batch carries much poison, but the backdoor builds up batch after batch.
Here the poison is the backdoor trigger (`attacks/backdoor.py`), stamped on a
small share of each batch with the label changed to the target class.
"""

import torch

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger_tensor


def drip_backdoor(
    images: torch.Tensor,
    labels: torch.Tensor,
    per_batch_rate: float,
    backdoor: BackdoorConfig,
    pixel_mean: tuple[float, ...],
    pixel_std: tuple[float, ...],
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poison a random `per_batch_rate` share of one incoming batch.

    Only images not already in the target class are chosen. Returns copies of
    (images, labels) plus a True/False mask marking the poisoned images, so
    later steps know exactly which ones were poisoned.
    """
    candidates = torch.nonzero(labels.cpu() != backdoor.target_class).flatten()
    count = min(round(per_batch_rate * len(labels)), len(candidates))
    chosen = candidates[torch.randperm(len(candidates), generator=generator)[:count]]

    is_poisoned = torch.zeros(len(labels), dtype=torch.bool)
    is_poisoned[chosen] = True
    mask = is_poisoned.to(labels.device)

    images, labels = images.clone(), labels.clone()
    images[mask] = add_trigger_tensor(images[mask], backdoor, pixel_mean, pixel_std)
    labels[mask] = backdoor.target_class
    return images, labels, mask
