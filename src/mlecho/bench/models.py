"""The image classifier that MLEcho protects in experiments.

This is the *victim* model: the one an attacker wants to poison. It is not
part of MLEcho's detector.
"""

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Convolution -> batch norm -> ReLU -> max pool (halves width and height)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """A small convolutional network for 32x32 colour images.

    Image size goes 32 -> 16 -> 8 -> 4 while channels (pattern detectors)
    grow 3 -> width -> 2*width -> 4*width. The classifier then turns those
    patterns into one score per class. A bigger `width` means more pattern
    detectors per layer: more capacity, but slower.
    """

    def __init__(self, num_classes: int = 10, width: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, width),
            _conv_block(width, 2 * width),
            _conv_block(2 * width, 4 * width),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (4*width, 4, 4) -> one long row of numbers
            nn.Dropout(0.3),  # randomly drop 30% during training to reduce overfitting
            nn.Linear(4 * width * 4 * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
