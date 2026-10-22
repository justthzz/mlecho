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

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger, poison_train_set
from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.bench.datasets import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_datasets
from mlecho.core import get_device, load_weights, seed_everything
from mlecho.detection.thresholds import count_threshold, score_threshold
from mlecho.detection.vae import ConvVAE, VAEConfig, reconstruction_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", default="data/checkpoints/vae-c16-z32-seed42.pt")
    parser.add_argument("--channels", type=int, default=16, help="must match the saved VAE")
    parser.add_argument("--latent", type=int, default=32, help="must match the saved VAE")
    parser.add_argument("--fit-size", type=int, default=4000, help="as in experiment 07")
    parser.add_argument("--patch", type=int, default=4, help="chosen in experiment 07")
    parser.add_argument("--image-rate", type=float, default=0.01, help="clean images flagged")
    parser.add_argument("--batch-rate", type=float, default=0.01, help="clean batches alarmed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=10, help="batches per window")
    parser.add_argument("--streams", type=int, default=1000, help="simulated batches per case")
    parser.add_argument("--gm-cache", default="data/poisons/gradient-matching-cat8183.npz")
    parser.add_argument("--mlflow-experiment", default="mlecho-detection")
    return parser.parse_args()


def score_raw(model: ConvVAE, raw: np.ndarray, patch: int, device) -> torch.Tensor:
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
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set, _ = get_cifar10_datasets(seed=args.seed, augment=False)
    full = train_set.dataset
    held_out = np.random.default_rng(args.seed).permutation(val_set.indices)[args.fit_size :]

    model = load_weights(ConvVAE(VAEConfig(args.channels, args.latent)), args.vae, device)

    # Thresholds, from trusted held-out images only.
    reference = score_raw(model, full.data[held_out], args.patch, device)
    image_t = score_threshold(reference, args.image_rate)
    batch_t = count_threshold(reference, image_t, args.batch_size, args.batch_rate, generator)
    window_t = count_threshold(
        reference, image_t, args.batch_size * args.window, args.batch_rate, generator
    )
    print(
        f"image threshold {image_t:.4f}; batch alarm at >= {batch_t} flags of {args.batch_size};"
        f" window alarm at >= {window_t} flags in {args.window} batches"
    )

    # Scores of every training image: clean, and with the backdoor square.
    train = np.asarray(train_set.indices)
    clean = score_raw(model, full.data[train], args.patch, device)
    squared = score_raw(model, add_trigger(full.data[train], BackdoorConfig()), args.patch, device)
    where = {position: k for k, position in enumerate(train)}  # dataset position -> row
    backdoor_view, _, _ = get_cifar10_datasets(seed=args.seed, augment=False)
    backdoor_rows = [where[i] for i in poison_train_set(backdoor_view, BackdoorConfig(), args.seed)]
    flip_view, _, _ = get_cifar10_datasets(seed=args.seed, augment=False)
    flip_rows = [where[i] for i in flip_labels(flip_view, LabelFlipConfig(), args.seed)]
    gm = np.load(args.gm_cache)
    gm_rows = [where[i] for i in gm["positions"]]
    gm_scores = score_raw(model, gm["pixels"], args.patch, device)
    non_plane = np.flatnonzero(np.asarray(full.targets)[train] != 0)

    attacks = {  # rows poisoned, and their scores once poisoned
        "backdoor": (backdoor_rows, squared[backdoor_rows]),
        "label_flip": (flip_rows, clean[flip_rows]),  # pictures unchanged
        "gradient_matching": (gm_rows, gm_scores),
    }

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="picture-check-detection"):
        mlflow.log_params(vars(args) | {"device": str(device)})
        mlflow.log_metrics(
            {
                "image_threshold": image_t,
                "batch_flags_needed": batch_t,
                "window_flags_needed": window_t,
            }
        )

        # Per image.
        print("\nPER IMAGE (threshold: 1% of trusted clean images)")
        for name, (rows, scores) in attacks.items():
            others = np.setdiff1d(np.arange(len(train)), rows)
            false_positive = (clean[others] > image_t).float().mean().item()
            recall = (scores > image_t).float().mean().item()
            caught, wrong = recall * len(rows), false_positive * len(others)
            precision = caught / (caught + wrong) if caught + wrong else 0.0
            mlflow.log_metrics(
                {
                    f"image_recall_{name}": recall,
                    f"image_fpr_{name}": false_positive,
                    f"image_precision_{name}": precision,
                }
            )
            print(
                f"{name:18s} recall {recall:6.1%}  false-positive rate {false_positive:5.2%}"
                f"  precision {precision:6.1%}  ({len(rows)} poisoned of {len(train)})"
            )

        # Per batch and per window, on simulated training streams.
        def stream(poisoned_rows, poisoned_scores, per_batch):
            """Flag counts for `streams` batches; `per_batch` None = the attack's natural rate."""
            picks = torch.randint(len(train), (args.streams, args.batch_size), generator=generator)
            scores = clean[picks].clone()
            if per_batch is None:  # batches drawn from the poisoned training set as it is
                lookup = dict(zip(poisoned_rows, poisoned_scores.tolist(), strict=True))
                for b, row in np.argwhere(np.isin(picks.numpy(), poisoned_rows)):
                    scores[b, row] = lookup[int(picks[b, row])]
            elif per_batch:  # slow-drip: this many squared images in every batch
                choice = torch.randint(
                    len(non_plane), (args.streams, per_batch), generator=generator
                )
                scores[:, :per_batch] = squared[non_plane[choice.numpy()]]
            return (scores > image_t).sum(dim=1)

        cases = {
            "clean stream": stream([], torch.empty(0), 0),
            "backdoor (5%)": stream(*attacks["backdoor"], None),
            "label-flip (5%)": stream(*attacks["label_flip"], None),
            "gradient matching (1%)": stream(*attacks["gradient_matching"], None),
            "slow-drip 1 per batch": stream([], torch.empty(0), 1),
            "slow-drip 5 per batch": stream([], torch.empty(0), 5),
        }
        print(f"\nPER BATCH of {args.batch_size} / PER WINDOW of {args.window} batches")
        for name, counts in cases.items():
            batch_alarm = (counts >= batch_t).float().mean().item()
            windows = counts[: len(counts) // args.window * args.window].view(-1, args.window)
            window_alarm = (windows.sum(dim=1) >= window_t).float().mean().item()
            key = name.split(" (")[0].replace(" ", "_").replace("-", "_")
            mlflow.log_metrics(
                {f"batch_alarm_{key}": batch_alarm, f"window_alarm_{key}": window_alarm}
            )
            print(
                f"{name:24s} mean flags/batch {counts.float().mean():5.2f}  "
                f"batch alarm {batch_alarm:6.1%}  window alarm {window_alarm:6.1%}"
            )


if __name__ == "__main__":
    main()
