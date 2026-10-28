"""Shared pieces for the experiment scripts: common options and the training loop.

Lives in experiments/ (not src/) because it uses MLflow and prints progress.
"""

import argparse
import json
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torchvision.transforms.functional as TF
from mlflow.tracking import MlflowClient
from torch import nn
from torch.utils.data import DataLoader

from mlecho.attacks.backdoor import BackdoorConfig, poison_train_set
from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.bench.datasets import get_cifar10_datasets, normalise
from mlecho.bench.evaluation import alarm_rates, image_metrics, simulate_flag_counts
from mlecho.bench.training import evaluate, train_one_epoch
from mlecho.core import load_weights
from mlecho.detection.label_check import ReferenceCNN, label_scores
from mlecho.detection.thresholds import count_threshold, score_threshold
from mlecho.detection.vae import ConvVAE, reconstruction_scores


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
        "--norm", choices=["batch", "group"], default="batch", help="SmallCNN normalisation"
    )
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


# ---------- step 3: shared pieces of the detection experiments ----------


def add_detection_args(parser: argparse.ArgumentParser) -> None:
    """Options every detection experiment shares (thresholds and simulated streams)."""
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-size", type=int, default=4000, help="trusted images to train on")
    parser.add_argument("--image-rate", type=float, default=0.01, help="clean images flagged")
    parser.add_argument("--batch-rate", type=float, default=0.01, help="clean batches alarmed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=10, help="batches per window")
    parser.add_argument("--streams", type=int, default=1000, help="simulated batches per case")
    parser.add_argument("--gm-cache", default="data/poisons/gradient-matching-cat8183.npz")
    parser.add_argument("--mlflow-experiment", default="mlecho-detection")


def detection_data(seed: int, fit_size: int, gm_cache: str) -> SimpleNamespace:
    """Everything the detection experiments score, built the same way every time.

    - fit / held_out: dataset positions of the trusted validation images
    - train: positions of the 45,000 training images (rows of every score vector)
    - backdoor / label_flip / gradient_matching: which rows each attack poisons
    - gm_pixels: the crafted gradient-matching images (from experiment 07's cache)
    - non_plane: rows whose true label isn't plane (slow-drip picks from these)
    """
    train_set, val_set, _ = get_cifar10_datasets(seed=seed, augment=False)
    full = train_set.dataset
    trusted = np.random.default_rng(seed).permutation(val_set.indices)
    train = np.asarray(train_set.indices)
    row = {position: k for k, position in enumerate(train)}
    backdoor_view, _, _ = get_cifar10_datasets(seed=seed, augment=False)
    flip_view, _, _ = get_cifar10_datasets(seed=seed, augment=False)
    gm = np.load(gm_cache)
    return SimpleNamespace(
        full=full,
        fit=trusted[:fit_size],
        held_out=trusted[fit_size:],
        train=train,
        backdoor=[row[i] for i in poison_train_set(backdoor_view, BackdoorConfig(), seed)],
        label_flip=[row[i] for i in flip_labels(flip_view, LabelFlipConfig(), seed)],
        gradient_matching=[row[i] for i in gm["positions"]],
        gm_pixels=gm["pixels"],
        non_plane=np.flatnonzero(np.asarray(full.targets)[train] != 0),
    )


def report_detection(
    reference: torch.Tensor,
    clean: torch.Tensor,
    attacks: dict[str, tuple[list[int], torch.Tensor]],
    drip_pool: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Set thresholds on trusted `reference` scores, then report and log detection.

    `clean`: score of every training image as clean. `attacks`: name -> (rows
    poisoned, their scores once poisoned). `drip_pool`: scores of slow-drip
    images. Logs to the active MLflow run and prints tables.
    """
    generator = torch.Generator().manual_seed(args.seed)
    image_t = score_threshold(reference, args.image_rate)
    batch_t = count_threshold(reference, image_t, args.batch_size, args.batch_rate, generator)
    window_size = args.batch_size * args.window
    window_t = count_threshold(reference, image_t, window_size, args.batch_rate, generator)
    mlflow.log_metrics(
        {"image_threshold": image_t, "batch_flags_needed": batch_t, "window_flags_needed": window_t}
    )
    print(
        f"image threshold {image_t:.4f}; batch alarm at >= {batch_t} flags of {args.batch_size};"
        f" window alarm at >= {window_t} flags in {args.window} batches"
    )

    print(f"\nPER IMAGE (threshold: {args.image_rate:.0%} of trusted clean images)")
    for name, (rows, scores) in attacks.items():
        others = clean[np.setdiff1d(np.arange(len(clean)), rows)]
        metrics = image_metrics(others, scores, image_t)
        mlflow.log_metrics({f"image_{k}_{name}": v for k, v in metrics.items()})
        print(
            f"{name:18s} recall {metrics['recall']:6.1%}  false-positive rate "
            f"{metrics['fpr']:5.2%}  precision {metrics['precision']:6.1%}"
        )

    def poisoned_population(rows, scores):
        population = clean.clone()
        population[rows] = scores
        return population

    cases = {"clean": (clean, 0)}
    cases |= {name: (poisoned_population(*a), 0) for name, a in attacks.items()}
    cases |= {"slow_drip_1": (clean, 1), "slow_drip_5": (clean, 5)}
    print(f"\nPER BATCH of {args.batch_size} / PER WINDOW of {args.window} batches")
    for name, (population, per_batch) in cases.items():
        counts = simulate_flag_counts(
            population, image_t, args.streams, args.batch_size, generator, drip_pool, per_batch
        )
        batch_rate, window_rate = alarm_rates(counts, batch_t, args.window, window_t)
        mlflow.log_metrics({f"batch_alarm_{name}": batch_rate, f"window_alarm_{name}": window_rate})
        print(
            f"{name:18s} mean flags/batch {counts.float().mean():5.2f}  "
            f"batch alarm {batch_rate:6.1%}  window alarm {window_rate:6.1%}"
        )


def picture_scores(model: nn.Module, raw: np.ndarray, patch: int, device) -> torch.Tensor:
    """Picture-check score (worst `patch` x `patch` reconstruction error) for raw images."""
    return torch.cat(
        [
            reconstruction_scores(
                model, normalise(raw[start : start + 1000]).to(device), patch
            ).cpu()
            for start in range(0, len(raw), 1000)
        ]
    )


def label_check_scores(
    models: list[nn.Module], raw: np.ndarray, labels: np.ndarray, which: np.ndarray, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(surprise scores, predicted classes) for raw images; image k uses models[which[k]]."""
    scores = torch.empty(len(raw))
    predictions = torch.empty(len(raw), dtype=torch.long)
    for f, model in enumerate(models):
        rows = np.flatnonzero(which == f)
        for start in range(0, len(rows), 1000):
            part = rows[start : start + 1000]
            images = normalise(raw[part]).to(device)
            batch_labels = torch.as_tensor(labels[part]).to(device)
            scores[part] = label_scores(model, images, batch_labels).cpu()
            with torch.no_grad():
                predictions[part] = model(images).argmax(dim=1).cpu()
    return scores, predictions


# Harmless photo changes (experiment 11; also the unseen test shifts of experiment 14).
SHIFTS = {  # name -> change applied to pixels in 0-1, shape (N, 3, H, W)
    "none": lambda x: x,
    "brighter x1.3": lambda x: TF.adjust_brightness(x, 1.3),
    "darker x0.7": lambda x: TF.adjust_brightness(x, 0.7),
    "more contrast x1.5": lambda x: TF.adjust_contrast(x, 1.5),
    "less contrast x0.6": lambda x: TF.adjust_contrast(x, 0.6),
    "more colour x1.5": lambda x: TF.adjust_saturation(x, 1.5),
    "less colour x0.5": lambda x: TF.adjust_saturation(x, 0.5),
    "hue shift 0.05": lambda x: TF.adjust_hue(x, 0.05),
    "blur (sigma 0.8)": lambda x: TF.gaussian_blur(x, kernel_size=3, sigma=0.8),
    "sensor noise (4/255)": lambda x: (x + torch.randn_like(x) * 4 / 255).clamp(0, 1),
}


def shifted(raw: np.ndarray, change) -> np.ndarray:
    """Apply a pixel change to raw images (N, H, W, 3) and round back to 0-255."""
    torch.manual_seed(0)  # same sensor noise every run
    pixels = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255
    out = change(pixels).clamp(0, 1) * 255
    return out.round().byte().permute(0, 2, 3, 1).numpy()


def fold_picture_scores(
    models, raw: np.ndarray, which: np.ndarray, patch: int, device
) -> torch.Tensor:
    """Picture score of each raw image k, using the cross-fitted VAE models[which[k]]."""
    scores = torch.empty(len(raw))
    for f, model in enumerate(models):
        rows = np.flatnonzero(which == f)
        scores[rows] = picture_scores(model, raw[rows], patch, device)
    return scores


@torch.no_grad()
def fold_log_probabilities(models, raw: np.ndarray, which: np.ndarray, device) -> torch.Tensor:
    """Reference log-probabilities (N, classes) of raw images; image k uses models[which[k]]."""
    out = None
    for f, model in enumerate(models):
        rows = np.flatnonzero(which == f)
        for start in range(0, len(rows), 1000):
            part = rows[start : start + 1000]
            logits = model.eval()(normalise(raw[part]).to(device))
            if out is None:
                out = torch.empty(len(raw), logits.shape[1])
            out[part] = torch.log_softmax(logits, dim=1).cpu()
    return out


class FoldScorer:
    """Out-of-fold evidence (picture score, log-probabilities) for raw images, from
    the 5 cross-fitted VAEs (experiment 13) and reference models (experiment 09)."""

    def __init__(self, vae_prefix: str, reference_prefix: str, folds: int, patch: int, device):
        self.vaes = [
            load_weights(ConvVAE(), f"{vae_prefix}{f}.pt", device).eval() for f in range(folds)
        ]
        self.references = [
            load_weights(ReferenceCNN(), f"{reference_prefix}{f}.pt", device).eval()
            for f in range(folds)
        ]
        self.patch, self.device = patch, device

    def __call__(self, raw: np.ndarray, which: np.ndarray) -> SimpleNamespace:
        """Evidence of raw image k from fold models `which[k]`."""
        return SimpleNamespace(
            picture=fold_picture_scores(self.vaes, raw, which, self.patch, self.device),
            log_probs=fold_log_probabilities(self.references, raw, which, self.device),
        )


# ---------- figures: shared by the figures_stepNN.py scripts ----------

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERMILLION, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#999999"
CLEAN_COLOUR, ATTACK_COLOUR = BLUE, ORANGE


class Results:
    """Small helper to read MLflow runs by the 8-character IDs used in RESULTS.md."""

    def __init__(self) -> None:
        self.client = MlflowClient()
        self._ids: dict[str, str] = {}

    def full_id(self, short: str) -> str:
        if short not in self._ids:
            runs = mlflow.search_runs(
                search_all_experiments=True, filter_string=f"attributes.run_id LIKE '{short}%'"
            )
            self._ids[short] = runs.run_id[0]
        return self._ids[short]

    def metric(self, short: str, key: str) -> float:
        return self.client.get_run(self.full_id(short)).data.metrics[key]

    def history(self, short: str, key: str) -> tuple[list[int], list[float]]:
        points = sorted(
            self.client.get_metric_history(self.full_id(short), key), key=lambda m: m.step
        )
        return [m.step for m in points], [m.value for m in points]

    def model(self, short: str) -> torch.nn.Module:
        return mlflow.pytorch.load_model(f"runs:/{self.full_id(short)}/model", map_location="cpu")

    def json(self, short: str, path: str):
        """A JSON artifact of the run (e.g. one saved with mlflow.log_dict)."""
        with open(self.client.download_artifacts(self.full_id(short), path)) as file:
            return json.load(file)

    def image(self, short: str, path: str) -> np.ndarray:
        local = self.client.download_artifacts(self.full_id(short), path)
        return plt.imread(local)


def show_row(axes, images, titles, row_label=None) -> None:
    for ax, image, title in zip(axes, images, titles, strict=True):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    if row_label:
        axes[0].set_ylabel(row_label, fontsize=9)


def percent(value: float) -> str:
    """One decimal, or two near 100% (so 99.96% isn't shown as 100.0%)."""
    return f"{value:.2f}%" if 99.5 < value < 100 else f"{value:.1f}%"


def label_bars(ax, bars, fmt=percent) -> None:
    """Write each bar's value above it. `fmt` is a format string or a function."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt(height) if callable(fmt) else fmt.format(height),
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=7,
        )
