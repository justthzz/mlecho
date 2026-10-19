"""Shared pieces for the experiment scripts: common options and the training loop.

Lives in experiments/ (not src/) because it uses MLflow and prints progress.
"""

import argparse
import time

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlecho.bench.training import evaluate, train_one_epoch


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add the options every training experiment shares."""
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=64, help="SmallCNN channels in block 1")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,  # gives both --augment and --no-augment
        default=True,
        help="randomly shift and mirror training images",
    )


def train_and_log(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    """Train for `epochs`, logging train/val metrics to the active MLflow run."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Big learning steps early, then smoothly smaller ones (cosine curve) to settle in.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = evaluate(model, val_loader, loss_fn, device)
        seconds = time.perf_counter() - start
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        mlflow.log_metrics(
            {
                "train_loss": train.loss,
                "train_acc": train.accuracy,
                "val_loss": val.loss,
                "val_acc": val.accuracy,
                "epoch_seconds": seconds,
                "lr": current_lr,
            },
            step=epoch,
        )
        print(
            f"epoch {epoch}/{epochs}  "
            f"train loss {train.loss:.3f} acc {train.accuracy:.1%}  |  "
            f"val loss {val.loss:.3f} acc {val.accuracy:.1%}  |  {seconds:.0f}s"
        )


def log_model(model: nn.Module, example_loader: DataLoader) -> None:
    """Save the trained model to the active MLflow run.

    MLflow records the model by running one example image through it.
    """
    example_images, _ = next(iter(example_loader))
    mlflow.pytorch.log_model(model.cpu(), name="model", input_example=example_images[:1].numpy())
