"""Shared pieces for the experiment scripts: common options and the training loop.

Lives in experiments/ (not src/) because it uses MLflow and prints progress.
"""

import argparse
import time

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlecho.bench.training import evaluate, train_one_epoch


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add the options every training experiment shares."""
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    parser.add_argument(
        "--schedule",
        choices=["cosine", "constant"],
        default="cosine",
        help="cosine: learning rate shrinks to ~0 by the end; constant: stays at --lr",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=64, help="SmallCNN channels in block 1")
    parser.add_argument(
        "--mlflow-experiment", help="override the MLflow experiment, e.g. mlecho-tuning"
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,  # gives both --augment and --no-augment
        default=True,
        help="randomly shift and mirror training images",
    )


def make_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build the optimizer and learning-rate schedule chosen on the command line."""
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.schedule == "cosine":
        # Big learning steps early, then smoothly smaller ones (cosine curve) to settle in.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)  # unchanged
    return optimizer, scheduler


def train_and_log(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Train for `args.epochs`, logging train/val metrics to the active MLflow run."""
    optimizer, scheduler = make_optimizer(model, args)
    epochs = args.epochs

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


def image_strip(images: np.ndarray, count: int = 8, scale: int = 4) -> np.ndarray:
    """Place `count` raw images side by side, enlarged `scale` times, for viewing."""
    strip = np.concatenate(images[:count], axis=1)
    return strip.repeat(scale, axis=0).repeat(scale, axis=1)


def log_model(model: nn.Module, example_loader: DataLoader) -> None:
    """Save the trained model to the active MLflow run.

    Saved as a normal PyTorch model ("pickle"), so it loads back as the same
    SmallCNN and accepts any batch size. (The default "pt2" format freezes
    the shape of the example input, so it would only accept 1 image.)
    """
    example_images, _ = next(iter(example_loader))
    mlflow.pytorch.log_model(
        model.cpu(),
        name="model",
        input_example=example_images[:2].numpy(),  # example input, recorded with the model
        serialization_format="pickle",
    )
