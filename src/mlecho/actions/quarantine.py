"""Quarantine: which images of a batch to leave out of training.

Works per image, not per batch: at a 5% poison rate almost every batch holds
a few poisoned images, so skipping whole batches would stop training. Which
images count as suspicious depends on the attack type the classifier named.
"""

import torch

from mlecho.classification.features import Calibration, surprise
from mlecho.classification.monitor import Evidence, Verdict


def quarantine_mask(
    evidence: Evidence, labels: torch.Tensor, calibration: Calibration, verdict: Verdict
) -> torch.Tensor:
    """True for every image to leave out, given the verdict that started the quarantine.

    Always: labels the reference model finds very surprising. Then by type:
    - label-flip (labels moved from class `fell` to class `rose`): images
      labelled `rose` that look more like `fell` than like `rose`;
    - backdoor / accumulative (trigger poison with one target label): pictures
      the picture check flags, and images with the target label that the
      reference model calls something else (a "plane" that looks like a dog);
    - anything else: pictures the picture check flags.
    """
    labels = labels.cpu()
    log_probs = evidence.log_probs
    drop = surprise(log_probs, labels) > calibration.surprise_threshold
    if verdict.attack_type == "label_flip":
        rose, fell = verdict.mix_pair
        return drop | ((labels == rose) & (log_probs[:, fell] > log_probs[:, rose]))
    drop |= evidence.picture > calibration.picture_threshold
    if verdict.attack_type in ("backdoor", "accumulative") and verdict.target_label >= 0:
        target = verdict.target_label
        drop |= (labels == target) & (log_probs.argmax(dim=1) != target)
    return drop
