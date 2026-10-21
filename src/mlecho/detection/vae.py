"""CNN-VAE: the picture check's model.

A variational autoencoder squeezes each image into a few numbers (the latent
code) and rebuilds it from them. Trained only on clean images, it learns to
rebuild what normal images look like. Anything unusual (a stuck-on square,
adversarial noise) is rebuilt badly, so the reconstruction error is the
image's anomaly score.

The encoder ends with adaptive pooling, so any image size works; the decoder
rebuilds at 32x32 and is resized to the input size when they differ.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VAEConfig:
    """Size of the CNN-VAE. Kept small so checking batches is cheap."""

    channels: int = 16  # channels after the first layer; doubles twice (16, 32, 64)
    latent: int = 32  # numbers in the latent code
    beta: float = 1.0  # weight of the KL term (how strongly codes are kept "tidy")


class ConvVAE(nn.Module):
    """Convolutional VAE for colour images (input already normalised)."""

    def __init__(self, config: VAEConfig | None = None):
        super().__init__()
        config = config or VAEConfig()  # default settings when none are given
        c = config.channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # any input size -> 4x4
            nn.Flatten(),
        )
        self.to_mean = nn.Linear(4 * c * 16, config.latent)
        self.to_log_var = nn.Linear(4 * c * 16, config.latent)
        self.from_latent = nn.Linear(config.latent, 4 * c * 16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (4 * c, 4, 4)),
            nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1),  # 16 -> 32
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reconstruction, mean, log variance) of the latent code."""
        hidden = self.encoder(x)
        mean, log_var = self.to_mean(hidden), self.to_log_var(hidden)
        if self.training:  # sample a code near the mean (the "variational" part)
            code = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
        else:  # when scoring, use the mean so the score is the same every time
            code = mean
        rebuilt = self.decoder(self.from_latent(code))
        if rebuilt.shape[-2:] != x.shape[-2:]:
            rebuilt = F.interpolate(rebuilt, size=x.shape[-2:], mode="bilinear")
        return rebuilt, mean, log_var


def vae_loss(
    x: torch.Tensor,
    rebuilt: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Average per image of: reconstruction error + beta * KL divergence.

    The KL term keeps latent codes close to a standard normal distribution,
    so similar images get similar codes.
    """
    reconstruction = F.mse_loss(rebuilt, x, reduction="none").flatten(1).sum(dim=1)
    kl = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).sum(dim=1)
    return (reconstruction + beta * kl).mean()


def train_vae_epoch(
    model: ConvVAE,
    images: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float = 1.0,
) -> float:
    """Train for one pass over `images` (batches of (images, labels)). Returns the mean loss."""
    model.train()
    total, count = 0.0, 0
    for batch, _ in images:
        batch = batch.to(device)
        loss = vae_loss(batch, *model(batch), beta=beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch)
        count += len(batch)
    return total / count


@torch.no_grad()
def reconstruction_scores(model: ConvVAE, images: torch.Tensor) -> torch.Tensor:
    """Anomaly score per image: mean squared reconstruction error (higher = stranger)."""
    model.eval()
    rebuilt, _, _ = model(images)
    return F.mse_loss(rebuilt, images, reduction="none").flatten(1).mean(dim=1)
