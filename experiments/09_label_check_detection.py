"""Experiment 09: how many poisons does the label check catch? (step 3.4)

Reference models learn the classes from the 5,000 trusted clean validation
images only, with 5-fold cross-fitting: each of 5 small models trains on 4,000
of them and predicts the other 1,000. So every trusted image gets a prediction
from a model that never saw it ("out-of-fold"), and the normal behaviour of the
check is measured on all 5,000 (500 per class). Each training image is checked
by one of the 5 models, picked at random, so it is treated exactly like the
out-of-fold trusted images.

Part 1, per image: surprise = -log P(label | picture), thresholds from the
out-of-fold trusted scores (as in experiment 08).
Part 2, label pattern: in a window of batches, is any (given label, looks-like
class) pair far more common than on trusted data? (pooled two-proportion z,
Bonferroni over all pairs). Added after part 1 failed on label-flip, so it is
also tested on three flip pairs never looked at before.

Run:   python experiments/09_label_check_detection.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse

import mlflow
import numpy as np
import torch
from common import (
    add_detection_args,
    detection_data,
    label_check_scores,
    report_detection,
)
from torch.utils.data import DataLoader, Subset

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.bench.datasets import CIFAR10_CLASSES, get_cifar10_datasets
from mlecho.core import get_device, save_weights, seed_everything
from mlecho.detection.label_check import (
    ReferenceCNN,
    agreement_counts,
    pair_surprise,
    pair_threshold,
    train_reference_epoch,
)

FLIPS = ((3, 5), (4, 7), (9, 1), (0, 2))  # cat->dog (the attack), then 3 pairs never looked at


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 50], help="batches")
    parser.add_argument("--save-prefix", default="data/checkpoints/reference-w32-seed42-fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    targets = np.asarray(data.full.targets)
    trusted = np.concatenate([data.fit, data.held_out])  # all 5,000 trusted images
    fold_of_trusted = np.arange(len(trusted)) % args.folds
    augmented, _, _ = get_cifar10_datasets(seed=args.seed, augment=True)

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="label-check-detection"):
        mlflow.log_params(vars(args) | {"device": str(device)})

        # Cross-fitting: model f never sees the trusted images of fold f.
        models = []
        for f in range(args.folds):
            loader = DataLoader(
                Subset(augmented.dataset, trusted[fold_of_trusted != f]),
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(args.seed + f),
            )
            model = ReferenceCNN(width=args.width).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            for _ in range(args.epochs):
                train_reference_epoch(model, loader, optimizer, device)
                scheduler.step()
            save_weights(model, f"{args.save_prefix}{f}.pt")
            models.append(model.eval())
            print(f"reference model {f + 1}/{args.folds} trained")

        trusted_labels = targets[trusted]
        reference, trusted_pred = label_check_scores(
            models, data.full.data[trusted], trusted_labels, fold_of_trusted, device
        )
        accuracy = (trusted_pred.numpy() == trusted_labels).mean()
        mlflow.log_metric("reference_out_of_fold_accuracy", accuracy)
        print(f"out-of-fold accuracy on the 5,000 trusted images: {accuracy:.1%}\n")

        # Every training image gets one randomly chosen fold model.
        which = np.random.default_rng(args.seed).integers(0, args.folds, len(data.train))
        train_raw, true = data.full.data[data.train], targets[data.train]
        clean, clean_pred = label_check_scores(models, train_raw, true, which, device)
        planes = np.zeros(len(train_raw), dtype=np.int64)  # backdoor / slow-drip label
        squared, squared_pred = label_check_scores(
            models, add_trigger(train_raw, BackdoorConfig()), planes, which, device
        )
        flip_rows = data.label_flip
        flipped, _ = label_check_scores(
            models, train_raw[flip_rows], np.full(len(flip_rows), 5), which[flip_rows], device
        )
        gm_rows = data.gradient_matching
        gm_scores, gm_pred = label_check_scores(
            models, data.gm_pixels, true[gm_rows], which[gm_rows], device
        )

        # Part 1: per image, per batch, per 10-batch window (same report as experiment 08).
        attacks = {
            "backdoor": (data.backdoor, squared[data.backdoor]),
            "label_flip": (flip_rows, flipped),
            "gradient_matching": (gm_rows, gm_scores),
        }
        report_detection(reference, clean, attacks, squared[data.non_plane], args)

        # Part 2: label pattern over windows of batches.
        reference_counts = agreement_counts(trusted_pred, torch.as_tensor(trusted_labels), 10)
        level = pair_threshold(10, args.batch_rate)
        mlflow.log_metric("pattern_level", level)
        true_t = torch.as_tensor(true)
        cases = {"clean": (clean_pred, true_t, None)}
        backdoor_pred, backdoor_label = clean_pred.clone(), true_t.clone()
        backdoor_pred[data.backdoor], backdoor_label[data.backdoor] = squared_pred[data.backdoor], 0
        cases["backdoor"] = (backdoor_pred, backdoor_label, None)
        with_gm = clean_pred.clone()
        with_gm[gm_rows] = gm_pred
        cases["gradient_matching"] = (with_gm, true_t, None)
        row = {position: k for k, position in enumerate(data.train)}
        for source, target in FLIPS:
            view, _, _ = get_cifar10_datasets(seed=args.seed, augment=False)
            rows = [row[i] for i in flip_labels(view, LabelFlipConfig(source, target), args.seed)]
            labels = true_t.clone()
            labels[rows] = target
            name = f"label_flip_{CIFAR10_CLASSES[source]}_to_{CIFAR10_CLASSES[target]}"
            cases[name] = (clean_pred, labels, (target, source))

        generator = torch.Generator().manual_seed(args.seed)
        print(f"\nLABEL PATTERN (alarm when a pair's z > {level:.2f})")
        for name, (predictions, labels, pair) in cases.items():
            line = f"{name:24s}"
            for window in args.windows:
                size = window * args.batch_size
                picks = torch.randint(len(labels), (args.streams // 10, size), generator=generator)
                found = [pair_surprise(predictions[p], labels[p], reference_counts) for p in picks]
                alarm = np.mean([z > level for z, _, _ in found])
                mlflow.log_metric(f"pattern_alarm_{name}_w{window}", alarm)
                line += f"  {window}-batch window alarm {alarm:6.1%}"
                if pair is not None:
                    named = [(j, i) == pair for z, j, i in found if z > level]
                    right = float(np.mean(named)) if named else 0.0
                    mlflow.log_metric(f"pattern_right_pair_{name}_w{window}", right)
                    line += f" (right pair {right:4.0%})"
            print(line)


if __name__ == "__main__":
    main()
