"""Statistical drift detectors: the common practice MLEcho is compared with.

They look at whole groups of data, not single images:
- KS test on simple picture statistics (mean and spread of each colour channel):
  has the distribution moved away from the trusted data?
- PSI on the label counts: has the mix of classes changed?
Thresholds are the usual practitioner choices, not tuned on our attacks.
"""

import math

import numpy as np
import torch


def image_statistics(raw: np.ndarray) -> torch.Tensor:
    """Per image: mean and standard deviation of R, G and B (raw pixels 0-255) -> (N, 6)."""
    pixels = torch.from_numpy(raw).float().flatten(1, 2)  # (N, H*W, 3)
    return torch.cat([pixels.mean(dim=1), pixels.std(dim=1)], dim=1)


def ks_p_value(reference: torch.Tensor, sample: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov test: chance of a gap this large between the
    two distributions if they were really the same (asymptotic formula)."""
    reference, sample = reference.double().sort().values, sample.double().sort().values
    values = torch.cat([reference, sample])
    cdf_reference = torch.searchsorted(reference, values, right=True) / len(reference)
    cdf_sample = torch.searchsorted(sample, values, right=True) / len(sample)
    gap = (cdf_reference - cdf_sample).abs().max().item()
    n = len(reference) * len(sample) / (len(reference) + len(sample))
    scaled = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * gap
    if scaled < 0.2:
        return 1.0
    total = sum((-1) ** (k - 1) * math.exp(-2 * k * k * scaled * scaled) for k in range(1, 101))
    return min(1.0, max(0.0, 2 * total))


def ks_alarm(reference: torch.Tensor, sample: torch.Tensor, rate: float = 0.01) -> bool:
    """Alarm if any feature column's KS p-value is below rate / number of features
    (Bonferroni: all features tested at once)."""
    features = reference.shape[1]
    return any(ks_p_value(reference[:, f], sample[:, f]) < rate / features for f in range(features))


def psi(reference_counts: torch.Tensor, sample_counts: torch.Tensor, floor: float = 1e-4) -> float:
    """Population stability index between two histograms (e.g. label counts).

    Common rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 large shift.
    `floor` keeps empty bins from dividing by zero.
    """
    expected = (reference_counts.double() / reference_counts.sum()).clamp_min(floor)
    actual = (sample_counts.double() / sample_counts.sum()).clamp_min(floor)
    return ((actual - expected) * torch.log(actual / expected)).sum().item()
