"""Quarantine: which images of a batch to leave out of training.

Works per image, not per batch: at a 5% poison rate almost every batch holds
a few poisoned images, so skipping whole batches would stop training. Which
images count as suspicious depends on the attack type the classifier named.
"""

import torch

from mlecho.classification.features import Calibration, surprise
from mlecho.classification.monitor import Evidence


def quarantine_mask(
    evidence: Evidence,
    labels: torch.Tensor,
    calibration: Calibration,
    attack_type: str,
    pair: tuple[int, int],
) -> torch.Tensor:
    """True for every image to leave out.

    - label-flip: images with the suspicious (given label, looks-like class)
      pair, plus labels the reference model finds very surprising;
    - anything else: images flagged by the picture check or the label check.
    """
    labels = labels.cpu()
    odd_label = surprise(evidence.log_probs, labels) > calibration.surprise_threshold
    if attack_type == "label_flip":
        given, looks_like = pair
        return odd_label | ((labels == given) & (evidence.log_probs.argmax(dim=1) == looks_like))
    return odd_label | (evidence.picture > calibration.picture_threshold)
