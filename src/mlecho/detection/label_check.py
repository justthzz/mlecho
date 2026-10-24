"""The label check: does each label fit its picture?

A small reference classifier is trained only on trusted clean data. For each
(image, label) pair, the score is how surprised the reference model is by the
label: minus the log of the probability it gives that label. A cat picture
labelled "dog" gets a high score. The picture check (CNN-VAE) can't see such
poison, because the picture itself is normal.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ReferenceCNN(nn.Module):
    """Small classifier for the label check. Adaptive pooling: any image size works."""

    def __init__(self, num_classes: int = 10, width: int = 32):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            block(3, width), block(width, 2 * width), block(2 * width, 4 * width)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(4 * width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_reference_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One pass over trusted (images, labels). Returns the mean loss."""
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss = F.cross_entropy(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def label_scores(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Surprise per image: -log P(given label | picture). Higher = label fits worse."""
    model.eval()
    log_probabilities = F.log_softmax(model(images), dim=1)
    return -log_probabilities.gather(1, labels.view(-1, 1)).squeeze(1)


def agreement_counts(
    predictions: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Row j, column i: how many images labelled j the reference model calls i."""
    counts = torch.bincount(labels * num_classes + predictions, minlength=num_classes**2)
    return counts.view(num_classes, num_classes).double()


def pair_surprise(
    predictions: torch.Tensor, labels: torch.Tensor, reference_counts: torch.Tensor
) -> tuple[float, int, int]:
    """The most over-represented (given label, looks-like class) pair in a group of images.

    For each pair j != i, compares the share of images labelled j that the
    reference calls i, in this group and in the trusted data, with a pooled
    two-proportion z-test. The test allows for the trusted data being a sample
    too (its shares are estimates, not exact). Returns (largest z, j, i): e.g. many
    pictures that look like cats but are labelled dog give a large z for (dog, cat).
    Single images can't show this; a batch or window can.
    """
    num_classes = reference_counts.shape[0]
    group = agreement_counts(predictions, labels, num_classes)
    has_label = group.sum(dim=1, keepdim=True)  # images per given label in the group
    n_group = has_label.clamp_min(1)  # avoid dividing by zero for labels not in the group
    n_ref = reference_counts.sum(dim=1, keepdim=True).clamp_min(1)
    pooled = (group + reference_counts) / (n_group + n_ref)  # shared share if nothing changed
    spread = torch.sqrt(pooled * (1 - pooled) * (1 / n_group + 1 / n_ref)).clamp_min(1e-12)
    z = (group / n_group - reference_counts / n_ref) / spread
    z[has_label.expand_as(z) == 0] = 0.0  # no images with this label in the group
    z.fill_diagonal_(float("-inf"))  # agreements are not suspicious
    best = int(z.argmax())
    return z.flatten()[best].item(), best // num_classes, best % num_classes


def pair_threshold(num_classes: int, rate: float) -> float:
    """z level that clean groups exceed only about `rate` of the time.

    One-sided normal quantile, divided over all num_classes * (num_classes - 1)
    pairs tested at once (Bonferroni), so the chance that *any* pair crosses it
    is at most `rate`.
    """
    pairs = num_classes * (num_classes - 1)
    normal = torch.distributions.Normal(0.0, 1.0)
    return normal.icdf(torch.tensor(1 - rate / pairs, dtype=torch.float64)).item()
