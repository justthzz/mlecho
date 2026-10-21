"""Experiment 07: can the CNN-VAE tell poisoned images from clean ones? (step 3.2)

The VAE learns what normal images look like from the trusted clean data only
(4,000 validation images; 1,000 more are kept aside). It never sees the
training images. We then score training images, clean and poisoned by each
attack, and measure the separation with ROC AUC (1.0 perfect, 0.5 coin flip).

Attacks scored (all on training images):
- backdoor: the 4x4 yellow square (slow-drip images look exactly the same)
- label-flip: pictures untouched, so the picture check should NOT see it
- gradient matching: the 450 dogs crafted against cat 8183 (experiment 06,
  cat -> dog). Crafted once (~5-20 min) and cached in data/poisons/.

Run:   python experiments/07_detector_feasibility.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from common import image_strip
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlecho.attacks.backdoor import BackdoorConfig, poison_train_set
from mlecho.attacks.gradient_matching import (
    GradientMatchingConfig,
    choose_poisons,
    craft_gradient_matching,
    read_images,
    write_poisons,
)
from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.bench.datasets import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_datasets
from mlecho.bench.evaluation import roc_auc
from mlecho.bench.models import SmallCNN
from mlecho.core import get_device, load_weights, save_weights, seed_everything
from mlecho.detection.vae import ConvVAE, VAEConfig, reconstruction_scores, train_vae_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--fit-size", type=int, default=4000, help="trusted images to train on")
    parser.add_argument("--clean-sample", type=int, default=5000, help="clean train images scored")
    parser.add_argument(
        "--attacker-model", default="data/checkpoints/burn-in-smallcnn-w64-seed42.pt"
    )
    parser.add_argument("--gm-target", type=int, default=8183, help="test image (a cat)")
    parser.add_argument("--gm-cache", default="data/poisons/gradient-matching-cat8183.npz")
    parser.add_argument("--save-vae", default="data/checkpoints/vae-c16-z32-seed42.pt")
    parser.add_argument("--mlflow-experiment", default="mlecho-detection")
    return parser.parse_args()


def gradient_matching_poisons(args, device) -> tuple[np.ndarray, np.ndarray]:
    """(positions, poisoned raw pixels) of experiment 06's cat -> dog attack, cached."""
    cache = Path(args.gm_cache)
    if cache.exists():
        saved = np.load(cache)
        return saved["positions"], saved["pixels"]
    train_set, _, test_set = get_cifar10_datasets(seed=args.seed, augment=False)
    full = train_set.dataset
    positions = choose_poisons(full, train_set.indices, 5, round(0.01 * len(train_set)), args.seed)
    attacker = load_weights(SmallCNN(), args.attacker_model, device)
    crafted = craft_gradient_matching(
        attacker,
        nn.CrossEntropyLoss(),
        (
            read_images(full, positions, CIFAR10_MEAN, CIFAR10_STD).to(device),
            torch.full((len(positions),), 5, device=device),
        ),
        (test_set[args.gm_target][0].unsqueeze(0).to(device), torch.tensor([5], device=device)),
        GradientMatchingConfig(),
        CIFAR10_MEAN,
        CIFAR10_STD,
        torch.Generator().manual_seed(args.seed),
    )
    write_poisons(full, positions, crafted, CIFAR10_MEAN, CIFAR10_STD)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, positions=positions, pixels=full.data[positions])
    return positions, full.data[positions]


SCORINGS = {"whole-image mean": None, "worst 2x2": 2, "worst 4x4": 4, "worst 8x8": 8}


def synthetic_attacks(raw: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Made-up attacks the guard could create itself, for choosing the scoring.

    Deliberately NOT the real attacks: a 4x4 square of random colour at a random
    place, and faint random +-8/255 noise on every pixel.
    """
    square = raw.copy()
    for image in square:
        row, col = rng.integers(0, raw.shape[1] - 4, size=2)
        image[row : row + 4, col : col + 4] = rng.integers(0, 256, size=3)
    noise = rng.choice([-8, 8], size=raw.shape)
    noisy = np.clip(raw.astype(int) + noise, 0, 255).astype(np.uint8)
    return {"random square": square, "faint noise": noisy}


def score(model: ConvVAE, raw: np.ndarray, device, patch: int | None) -> torch.Tensor:
    """Anomaly score for raw images (N, H, W, 3), in batches."""
    images = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    images = (images - mean) / std
    return torch.cat(
        [reconstruction_scores(model, chunk.to(device), patch).cpu() for chunk in images.split(500)]
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    train_set, val_set, _ = get_cifar10_datasets(seed=args.seed, augment=False)
    full = train_set.dataset
    rng = np.random.default_rng(args.seed)

    # Trusted clean data: the validation images, split into "fit" and "held out".
    trusted = rng.permutation(val_set.indices)
    fit, held_out = trusted[: args.fit_size], trusted[args.fit_size :]
    fit_images = read_images(full, fit, CIFAR10_MEAN, CIFAR10_STD)
    loader = DataLoader(
        TensorDataset(fit_images, torch.zeros(len(fit))),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # The images to score: clean training images and each attack's poisoned images.
    clean = np.sort(rng.choice(train_set.indices, args.clean_sample, replace=False))
    backdoor_view, _, _ = get_cifar10_datasets(seed=args.seed, augment=False)
    backdoor_at = poison_train_set(backdoor_view, BackdoorConfig(), seed=args.seed)
    flip_view, _, _ = get_cifar10_datasets(seed=args.seed, augment=False)
    flipped_at = flip_labels(flip_view, LabelFlipConfig(), seed=args.seed)
    gm_at, gm_pixels = gradient_matching_poisons(args, device)
    sets = {
        "clean_held_out": full.data[held_out],  # trusted but unseen: sanity check, AUC ~0.5
        "backdoor": backdoor_view.dataset.data[backdoor_at],
        "label_flip": flip_view.dataset.data[flipped_at],
        "gradient_matching": gm_pixels,
    }
    originals = {"backdoor": full.data[backdoor_at], "gradient_matching": full.data[gm_at]}

    config = VAEConfig(channels=args.channels, latent=args.latent, beta=args.beta)
    model = ConvVAE(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="vae-feasibility"):
        mlflow.log_params(vars(args) | {"device": str(device)})
        for epoch in range(1, args.epochs + 1):
            loss = train_vae_epoch(model, loader, optimizer, device, beta=args.beta)
            mlflow.log_metric("vae_train_loss", loss, step=epoch)
            if epoch % 10 == 0:
                print(f"epoch {epoch}/{args.epochs}  VAE loss {loss:.1f}")
        save_weights(model, args.save_vae)

        # 1. Choose the scoring on the trusted held-out images + made-up attacks only.
        held_out_raw = full.data[held_out]
        dev = synthetic_attacks(held_out_raw, np.random.default_rng(args.seed + 1))
        dev_auc = {}
        for name, patch in SCORINGS.items():
            reference = score(model, held_out_raw, device, patch)
            aucs = [roc_auc(reference, score(model, raw, device, patch)) for raw in dev.values()]
            dev_auc[name] = float(np.mean(aucs))
            print(
                f"development  {name:16s} "
                + "  ".join(f"{d}: AUC {a:.3f}" for d, a in zip(dev, aucs, strict=True))
            )
        chosen = max(dev_auc, key=dev_auc.get)
        patch = SCORINGS[chosen]
        mlflow.log_params({"chosen_scoring": chosen})
        print(f"chosen on development data: {chosen}\n")

        # 2. Report every scoring on the real attacks (the chosen one is the official result).
        results = {}
        for name, p in SCORINGS.items():
            clean_scores = score(model, full.data[clean], device, p)
            tag = "worst" + str(p) if p else "mean"
            for attack, raw in sets.items():
                scores = score(model, raw, device, p)
                auc = roc_auc(clean_scores, scores)
                mlflow.log_metric(f"auc_{attack}_{tag}", auc)
                if p == patch:
                    results[attack] = scores
                print(f"{name:16s} {attack:18s} AUC {auc:.3f}")
        clean_scores = score(model, full.data[clean], device, patch)
        for attack, raw in originals.items():  # same images, before vs after poisoning
            higher = (results[attack] > score(model, raw, device, patch)).float().mean().item()
            mlflow.log_metric(f"higher_than_own_original_{attack}", higher)
            print(f"{attack}: poisoned scores higher than its own original {higher:.1%} ({chosen})")

        # Pictures: score distributions, and a few images with their reconstructions.
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        bins = np.linspace(0, float(np.quantile(clean_scores, 0.995)) * 1.5, 60)
        ax.hist(clean_scores, bins, density=True, alpha=0.6, label="clean training images")
        for name in ("backdoor", "gradient_matching", "label_flip"):
            ax.hist(results[name], bins, density=True, histtype="step", linewidth=1.5, label=name)
        ax.set_xlabel(f"anomaly score ({chosen} reconstruction error)")
        ax.set_ylabel("density")
        ax.legend()
        mlflow.log_figure(fig, "score_histograms.png")
        plt.close(fig)
        strip = np.concatenate([image_strip(sets["backdoor"][:8]), image_strip(gm_pixels[:8])])
        mlflow.log_image(strip, "examples_backdoor_top_gradient_matching_bottom.png")


if __name__ == "__main__":
    main()
