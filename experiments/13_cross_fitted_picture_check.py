"""Experiment 13: the picture check with 5-fold cross-fitting (step 4.1).

The CNN-VAE from experiment 07 partly remembers the 4,000 trusted images it
trained on: it flags 0.4% of them, against 1% of new images. The attack
classifier (step 4) learns from windows built out of trusted images, so their
scores must look like scores of new images. Same fix as the label check
(experiment 09): 5 VAEs, each trained on 4,000 trusted images and scoring the
other 1,000, with the same folds. Every trusted image then has an out-of-fold
score, and the thresholds use all 5,000 of them.

Also re-runs the experiment 08 report with these thresholds, to check that the
cross-fitted picture check behaves like the single VAE.

Run:   python experiments/13_cross_fitted_picture_check.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
from common import add_detection_args, detection_data, fold_picture_scores, report_detection
from torch.utils.data import DataLoader, TensorDataset

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.attacks.gradient_matching import read_images
from mlecho.bench.datasets import CIFAR10_MEAN, CIFAR10_STD
from mlecho.core import get_device, load_weights, save_weights, seed_everything
from mlecho.detection.vae import ConvVAE, VAEConfig, train_vae_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch", type=int, default=4, help="chosen in experiment 07")
    parser.add_argument("--save-prefix", default="data/checkpoints/vae-c16-z32-seed42-fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    trusted = np.concatenate([data.fit, data.held_out])
    fold_of_trusted = np.arange(len(trusted)) % args.folds  # same folds as experiment 09

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="cross-fitted-picture-check"):
        mlflow.log_params(vars(args) | {"device": str(device)})

        models = []
        for f in range(args.folds):
            model = ConvVAE(VAEConfig()).to(device)
            path = f"{args.save_prefix}{f}.pt"
            if Path(path).exists():  # trained by an earlier run with the same settings
                models.append(load_weights(model, path, device).eval())
                continue
            images = read_images(
                data.full, trusted[fold_of_trusted != f], CIFAR10_MEAN, CIFAR10_STD
            )
            loader = DataLoader(
                TensorDataset(images, torch.zeros(len(images))),
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(args.seed + f),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for _ in range(args.epochs):
                loss = train_vae_epoch(model, loader, optimizer, device)
            mlflow.log_metric(f"vae_fold{f}_final_loss", loss)
            save_weights(model, path)
            models.append(model.eval())
            print(f"VAE {f + 1}/{args.folds} trained (final loss {loss:.1f})")

        reference = fold_picture_scores(
            models, data.full.data[trusted], fold_of_trusted, args.patch, device
        )
        which = np.random.default_rng(args.seed).integers(0, args.folds, len(data.train))
        train_raw = data.full.data[data.train]
        clean = fold_picture_scores(models, train_raw, which, args.patch, device)
        squared = fold_picture_scores(
            models, add_trigger(train_raw, BackdoorConfig()), which, args.patch, device
        )
        gm_rows = data.gradient_matching
        gm_scores = fold_picture_scores(models, data.gm_pixels, which[gm_rows], args.patch, device)

        attacks = {
            "backdoor": (data.backdoor, squared[data.backdoor]),
            "label_flip": (data.label_flip, clean[data.label_flip]),  # pixels untouched
            "gradient_matching": (gm_rows, gm_scores),
        }
        report_detection(reference, clean, attacks, squared[data.non_plane], args)


if __name__ == "__main__":
    main()
