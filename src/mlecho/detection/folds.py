"""Cross-fitting the detector on the user's trusted images (used by `Guard.calibrate`).

The trusted images are split into `folds` parts. Fold model f (one CNN-VAE
and one reference classifier) trains on every part except f, so each trusted
image can be scored by a model that never saw it ("out-of-fold"). That keeps
the calibration honest (experiments 09 and 13). Later, each new image is
scored by one fold model picked at random, so it is treated the same way.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from mlecho.detection.label_check import ReferenceCNN, train_reference_epoch
from mlecho.detection.vae import ConvVAE, VAEConfig, reconstruction_scores, train_vae_epoch


def fold_numbers(count: int, folds: int, generator: torch.Generator) -> torch.Tensor:
    """Fold number (0 .. folds-1) of each of `count` images, in equal shares at random."""
    return torch.randperm(count, generator=generator) % folds


class _Shifted(Dataset):
    """Images moved up to `shift` pixels and mirrored half the time (training only)."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, shift: int = 4):
        self.images, self.labels, self.shift = images, labels, shift

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[i]
        if torch.rand(()) < 0.5:
            image = image.flip(-1)
        s = self.shift
        padded = F.pad(image, (s, s, s, s))
        row, col = torch.randint(0, 2 * s + 1, (2,)).tolist()
        return padded[:, row : row + image.shape[1], col : col + image.shape[2]], self.labels[i]


def train_fold_models(
    images: torch.Tensor,
    labels: torch.Tensor,
    fold: torch.Tensor,
    num_classes: int,
    device: torch.device,
    vae_epochs: int = 60,
    reference_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[list[nn.Module], list[nn.Module]]:
    """Train one CNN-VAE and one reference model per fold (settings of experiments 07, 09).

    `images`: normalised trusted images (N, 3, H, W) on the CPU; `labels` (N,);
    `fold`: each image's fold number. Returns (vaes, references), in eval mode.
    """
    vaes, references = [], []
    for f in range(int(fold.max()) + 1):
        train = fold != f
        torch.manual_seed(seed + f)
        vae = ConvVAE(VAEConfig()).to(device)
        loader = DataLoader(
            TensorDataset(images[train], labels[train]),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + f),
        )
        optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
        for _ in range(vae_epochs):
            train_vae_epoch(vae, loader, optimizer, device)
        vaes.append(vae.eval())

        reference = ReferenceCNN(num_classes=num_classes).to(device)
        loader = DataLoader(
            _Shifted(images[train], labels[train]),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + f),
        )
        optimizer = torch.optim.Adam(reference.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=reference_epochs)
        for _ in range(reference_epochs):
            train_reference_epoch(reference, loader, optimizer, device)
            scheduler.step()
        references.append(reference.eval())
    return vaes, references


@torch.no_grad()
def fold_scores(
    vaes: list[nn.Module],
    references: list[nn.Module],
    images: torch.Tensor,
    which: torch.Tensor,
    patch: int = 4,
    chunk: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(picture scores (N,), log-probabilities (N, classes)) on the CPU; image k is
    scored by fold models `which[k]`. `images` may be on the CPU or the models' device.

    The images are grouped by fold, each group is scored in chunks, and all
    results come back to the CPU in one copy (each copy makes the CPU wait for
    the GPU, so fewer copies is faster).
    """
    device = next(vaes[0].parameters()).device
    which = which.cpu()
    parts, outputs = [], []
    for f, (vae, reference) in enumerate(zip(vaes, references, strict=True)):
        for part in torch.nonzero(which == f).flatten().split(chunk):
            batch = images[part.to(images.device)].to(device)
            picture = reconstruction_scores(vae.eval(), batch, patch)
            log_probs = torch.log_softmax(reference.eval()(batch), dim=1)
            outputs.append(torch.cat([picture.unsqueeze(1), log_probs], dim=1))
            parts.append(part)
    scores = torch.empty(len(images), outputs[0].shape[1])
    scores[torch.cat(parts)] = torch.cat(outputs).cpu()
    return scores[:, 0].contiguous(), scores[:, 1:].contiguous()
