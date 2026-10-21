"""Turning anomaly scores into alarms, with a chosen false-alarm rate.

Every threshold is set from scores of trusted clean images only:
- an image threshold: only `rate` of clean images score above it;
- a count threshold: for a group of images (a batch, or a window of batches),
  how many flagged images is "more than chance" for clean data.
"""

import torch


def score_threshold(clean_scores: torch.Tensor, rate: float) -> float:
    """Score above which only `rate` of clean images fall (e.g. 0.01 = 1% false alarms)."""
    return torch.quantile(clean_scores.flatten().double(), 1 - rate).item()


def count_threshold(
    clean_scores: torch.Tensor,
    threshold: float,
    group_size: int,
    rate: float,
    generator: torch.Generator,
    trials: int = 20000,
) -> int:
    """Smallest flag count that a group of clean images reaches at most `rate` of the time.

    Simulates `trials` groups of `group_size` clean images (drawn at random from
    `clean_scores`) and counts how many score above `threshold` in each. An alarm
    for "at least this many flags" then rings for at most `rate` of clean groups.
    """
    picks = torch.randint(len(clean_scores), (trials, group_size), generator=generator)
    counts = (clean_scores.flatten()[picks] > threshold).sum(dim=1)
    for needed in range(group_size + 1):
        if (counts >= needed).float().mean().item() <= rate:
            return needed
    return group_size + 1  # never alarm (only if rate is 0)
