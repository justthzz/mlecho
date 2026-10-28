"""Window features: the numbers the attack classifier looks at.

A window is a few consecutive training batches. For each image the detector
gives a picture score (CNN-VAE) and the reference model's log-probabilities
(label check). This module turns one window of them into a short list of
numbers, each measured against what trusted clean data looks like, so the
numbers mean the same thing on any dataset.
"""

from dataclasses import dataclass

import torch

from mlecho.detection.label_check import agreement_counts, pair_surprise, psi

FEATURES = (
    "picture_rate",  # share of pictures flagged, divided by the clean share (1 = normal)
    "picture_dispersion",  # how uneven the flags are across batches (about 1 if random)
    "picture_median",  # typical picture score, divided by the trusted one
    "picture_top_label",  # share of flagged pictures that carry the most common label
    "picture_surprise",  # how badly the labels of flagged pictures fit (vs trusted)
    "label_rate",  # share of labels flagged as surprising, divided by the clean share
    "label_median",  # typical label surprise, minus the trusted one
    "label_top_label",  # share of surprising labels that are the same class
    "pattern_z",  # strongest "labelled j, looks like i" pair (z-test, experiment 09)
    "label_mix",  # change in the mix of labels (PSI), divided by its clean level
)


@dataclass(frozen=True)
class Calibration:
    """What clean data looks like, measured once on trusted images."""

    picture_threshold: float  # picture score above which an image is flagged
    picture_median: float
    surprise_threshold: float  # label surprise above which a label is flagged
    surprise_median: float
    reference_counts: torch.Tensor  # agreement counts (given label, predicted class)
    label_counts: torch.Tensor  # how many trusted images have each label
    mix_level: float  # label-mix PSI that only `rate` of clean windows exceed
    rate: float  # share of clean images each threshold flags


def surprise(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """-log P(label | picture), for any shape (..., classes) with labels of shape (...)."""
    return -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def stream_rows(num_trusted: int, size: int, generator: torch.Generator) -> torch.Tensor:
    """Rows of trusted images for one made-up clean window of `size` images.

    The real stream's class mix is only estimated by the trusted images (a
    sample of a few thousand), so a window drawn straight from them would look
    more like them than real data does. First resample the trusted set with
    replacement (a bootstrap: "another sample like it"), then draw the window
    from that resample.
    """
    pool = torch.randint(num_trusted, (num_trusted,), generator=generator)
    return pool[torch.randint(num_trusted, (size,), generator=generator)]


def calibrate(
    picture: torch.Tensor,
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    window_size: int,
    rate: float,
    generator: torch.Generator,
    trials: int = 1000,
) -> Calibration:
    """Measure normal behaviour on trusted images (scores should be out-of-fold).

    `picture` (N,), `log_probs` (N, classes), `labels` (N,). The label-mix level
    comes from `trials` made-up clean windows of `window_size` images
    (`stream_rows`), so it allows for the trusted mix being only an estimate.
    """
    num_classes = log_probs.shape[1]
    scores = surprise(log_probs, labels)
    label_counts = torch.bincount(labels, minlength=num_classes)
    mixes = [
        psi(label_counts, torch.bincount(labels[rows], minlength=num_classes))
        for rows in (stream_rows(len(labels), window_size, generator) for _ in range(trials))
    ]
    return Calibration(
        picture_threshold=torch.quantile(picture.double(), 1 - rate).item(),
        picture_median=picture.median().item(),
        surprise_threshold=torch.quantile(scores.double(), 1 - rate).item(),
        surprise_median=scores.median().item(),
        reference_counts=agreement_counts(log_probs.argmax(dim=1), labels, num_classes),
        label_counts=label_counts,
        mix_level=torch.quantile(torch.tensor(mixes, dtype=torch.float64), 1 - rate).item(),
        rate=rate,
    )


def _top_label_share(labels: torch.Tensor, num_classes: int) -> float:
    """Share of `labels` that are the most common one (0 if there are none)."""
    if len(labels) == 0:
        return 0.0
    return torch.bincount(labels, minlength=num_classes).max().item() / len(labels)


def window_features(
    picture: torch.Tensor,
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    calibration: Calibration,
) -> torch.Tensor:
    """The FEATURES of one window, in that order.

    `picture` and `labels` have shape (batches, batch size); `log_probs` has
    shape (batches, batch size, classes).
    """
    num_classes = log_probs.shape[-1]
    scores = surprise(log_probs, labels)
    flagged = picture > calibration.picture_threshold
    per_batch = flagged.sum(dim=1).double()
    mean = per_batch.mean().item()
    dispersion = per_batch.var().item() / mean if mean > 0 else 1.0
    odd_label = scores > calibration.surprise_threshold

    flat_labels = labels.flatten()
    z, _, _ = pair_surprise(
        log_probs.argmax(dim=-1).flatten(), flat_labels, calibration.reference_counts
    )
    mix = psi(calibration.label_counts, torch.bincount(flat_labels, minlength=num_classes))
    flagged_surprise = (
        scores[flagged].mean().item() if flagged.any() else calibration.surprise_median
    )
    return torch.tensor(
        [
            flagged.float().mean().item() / calibration.rate,
            dispersion,
            picture.median().item() / calibration.picture_median,
            _top_label_share(labels[flagged], num_classes),
            flagged_surprise - calibration.surprise_median,
            odd_label.float().mean().item() / calibration.rate,
            scores.median().item() - calibration.surprise_median,
            _top_label_share(labels[odd_label], num_classes),
            z,
            mix / calibration.mix_level,
        ]
    )
