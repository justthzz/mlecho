"""Watching the training stream: evidence for every batch, a verdict for every window.

The monitor holds the guard's detector models (cross-fitted CNN-VAEs and
reference models), the calibration and the attack classifier. For each
incoming batch it scores every image; after `window` batches it turns the
collected evidence into features and asks the classifier which type it is.
"""

from dataclasses import dataclass

import torch
from torch import nn

from mlecho.classification.classifier import ATTACK_TYPES, AttackClassifier
from mlecho.classification.features import Calibration, surprise, window_features
from mlecho.detection.label_check import pair_surprise
from mlecho.detection.vae import reconstruction_scores


@dataclass(frozen=True)
class Evidence:
    """What the detector says about one batch: a picture score and the reference
    model's log-probabilities for every image (on the CPU)."""

    picture: torch.Tensor  # (N,)
    log_probs: torch.Tensor  # (N, classes)


@dataclass(frozen=True)
class Verdict:
    """The classifier's answer for one window."""

    attack_type: str  # one of ATTACK_TYPES
    confidence: float  # its probability
    probabilities: tuple[float, ...]  # all types, in ATTACK_TYPES order
    pair: tuple[int, int]  # most over-represented (given label, looks-like class) pair
    target_label: int  # most common label among flagged pictures (else surprising labels; -1)
    mix_pair: tuple[int, int]  # (label whose share rose most, label whose share fell most)


class Monitor:
    """Scores batches and gives a Verdict every `window` batches.

    Each image is scored by one of the fold models, picked at random, exactly
    as in the calibration (experiments 09, 13, 14).
    """

    def __init__(
        self,
        vaes: list[nn.Module],
        references: list[nn.Module],
        calibration: Calibration,
        classifier: AttackClassifier,
        window: int = 10,
        patch: int = 4,
        seed: int = 0,
    ):
        self.vaes = [m.eval() for m in vaes]
        self.references = [m.eval() for m in references]
        self.calibration, self.classifier = calibration, classifier.eval()
        self.window, self.patch = window, patch
        self.generator = torch.Generator().manual_seed(seed)
        self._pending: list[tuple[Evidence, torch.Tensor]] = []

    @torch.no_grad()
    def evidence(self, images: torch.Tensor) -> Evidence:
        """Score a batch of normalised images (on the models' device)."""
        which = torch.randint(len(self.vaes), (len(images),), generator=self.generator)
        picture = torch.empty(len(images))
        log_probs = None
        for f, (vae, reference) in enumerate(zip(self.vaes, self.references, strict=True)):
            rows = torch.nonzero(which == f).flatten()
            if len(rows) == 0:
                continue
            part = images[rows.to(images.device)]
            picture[rows] = reconstruction_scores(vae, part, self.patch).cpu()
            scores = torch.log_softmax(reference(part), dim=1).cpu()
            if log_probs is None:
                log_probs = torch.empty(len(images), scores.shape[1])
            log_probs[rows] = scores
        return Evidence(picture=picture, log_probs=log_probs)

    def add(self, evidence: Evidence, labels: torch.Tensor) -> Verdict | None:
        """Collect one batch; when the window is full, return its Verdict (else None)."""
        self._pending.append((evidence, labels.cpu()))
        if len(self._pending) < self.window:
            return None
        batch = min(len(labels) for _, labels in self._pending)  # equal-sized batches
        picture = torch.stack([e.picture[:batch] for e, _ in self._pending])
        log_probs = torch.stack([e.log_probs[:batch] for e, _ in self._pending])
        given = torch.stack([labels[:batch] for _, labels in self._pending])
        self._pending = []
        features = window_features(picture, log_probs, given, self.calibration)
        probabilities = self.classifier.probabilities(features.unsqueeze(0))[0]
        best = int(probabilities.argmax())
        _, j, i = pair_surprise(
            log_probs.argmax(dim=-1).flatten(), given.flatten(), self.calibration.reference_counts
        )
        num_classes = log_probs.shape[-1]
        cal = self.calibration
        flagged = given[picture > cal.picture_threshold]
        if len(flagged) == 0:
            flagged = given[surprise(log_probs, given) > cal.surprise_threshold]
        target = (
            int(torch.bincount(flagged, minlength=num_classes).argmax()) if len(flagged) else -1
        )
        share = torch.bincount(given.flatten(), minlength=num_classes) / given.numel()
        change = share - cal.label_counts / cal.label_counts.sum()
        return Verdict(
            attack_type=ATTACK_TYPES[best],
            confidence=probabilities[best].item(),
            probabilities=tuple(probabilities.tolist()),
            pair=(j, i),
            target_label=target,
            mix_pair=(int(change.argmax()), int(change.argmin())),
        )
