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
