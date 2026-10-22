"""Experiment 08: how many poisons does the picture check catch? (step 3.3)

Uses the CNN-VAE from experiment 07 with its chosen score (worst 4x4 patch).
All thresholds come from the 1,000 trusted held-out images only:
- image threshold: 1% of clean images flagged;
- batch alarm: at least N flags in a batch of 128, with N set so that 1% of
  clean batches alarm;
- window alarm: at least M flags in the last 10 batches (1% of clean windows).

Evaluated on the 45,000 training images (never seen by the detector):
per image (recall, false-positive rate, precision at the attack's real poison
rate) and per batch / window on simulated training streams.

Run:   python experiments/08_picture_check_detection.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse

import mlflow
import numpy as np
import torch
from common import add_detection_args, detection_data, report_detection

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.bench.datasets import CIFAR10_MEAN, CIFAR10_STD
from mlecho.core import get_device, load_weights, seed_everything
from mlecho.detection.vae import ConvVAE, VAEConfig, reconstruction_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    parser.add_argument("--vae", default="data/checkpoints/vae-c16-z32-seed42.pt")
    parser.add_argument("--channels", type=int, default=16, help="must match the saved VAE")
    parser.add_argument("--latent", type=int, default=32, help="must match the saved VAE")
    parser.add_argument("--patch", type=int, default=4, help="chosen in experiment 07")
    return parser.parse_args()


def picture_scores(model: ConvVAE, raw: np.ndarray, patch: int, device) -> torch.Tensor:
    """Worst-patch score for raw images (N, H, W, 3), in chunks."""
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    out = []
    for start in range(0, len(raw), 1000):
        chunk = torch.from_numpy(raw[start : start + 1000]).permute(0, 3, 1, 2).float() / 255
        out.append(reconstruction_scores(model, ((chunk - mean) / std).to(device), patch).cpu())
    return torch.cat(out)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    model = load_weights(ConvVAE(VAEConfig(args.channels, args.latent)), args.vae, device)

    def score(raw):
        return picture_scores(model, raw, args.patch, device)

    reference = score(data.full.data[data.held_out])
    train_raw = data.full.data[data.train]
    clean = score(train_raw)
    squared = score(add_trigger(train_raw, BackdoorConfig()))
    attacks = {
        "backdoor": (data.backdoor, squared[data.backdoor]),
        "label_flip": (data.label_flip, clean[data.label_flip]),  # pictures unchanged
        "gradient_matching": (data.gradient_matching, score(data.gm_pixels)),
    }

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="picture-check-detection"):
        mlflow.log_params(vars(args) | {"device": str(device)})
        report_detection(reference, clean, attacks, squared[data.non_plane], args)


if __name__ == "__main__":
    main()
