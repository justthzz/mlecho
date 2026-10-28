"""Experiment 12: statistical drift baselines on the same attacks (step 3.7).

What a practitioner might use instead of MLEcho, with the usual thresholds:
- KS test on simple picture statistics (mean and spread of R, G, B per image),
  Bonferroni over the 6 features, alarm at overall p < 1%;
- PSI on the label counts, alarm at PSI > 0.1 ("moderate shift");
- the same PSI with a calibrated threshold: exceeded by only 1% of windows of
  trusted labels (bootstrapped), like MLEcho's own thresholds.
Reference = the 5,000 trusted images. Each case: 100 windows of 10 or 50
batches (1,280 or 6,400 images) drawn from the training stream.

Run:   python experiments/12_drift_baselines.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse

import mlflow
import numpy as np
import torch
import torchvision.transforms.functional as TF
from common import add_detection_args, detection_data

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.bench.baselines.drift import image_statistics, ks_alarm
from mlecho.core import seed_everything
from mlecho.detection.label_check import psi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    parser.add_argument("--windows", type=int, nargs="+", default=[10, 50], help="batches")
    parser.add_argument("--trials", type=int, default=100, help="windows per case")
    parser.add_argument("--psi-alarm", type=float, default=0.1)
    return parser.parse_args()


def brighter(raw: np.ndarray) -> np.ndarray:
    pixels = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255
    out = TF.adjust_brightness(pixels, 1.3).clamp(0, 1) * 255
    return out.round().byte().permute(0, 2, 3, 1).numpy()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    targets = np.asarray(data.full.targets)

    trusted = np.concatenate([data.fit, data.held_out])
    reference_stats = image_statistics(data.full.data[trusted])
    reference_labels = torch.bincount(torch.as_tensor(targets[trusted]), minlength=10)

    train_raw = data.full.data[data.train]
    clean_stats = image_statistics(train_raw)
    squared_stats = image_statistics(add_trigger(train_raw, BackdoorConfig()))
    true = torch.as_tensor(targets[data.train])

    def poisoned(rows, stats=None, label=None):
        s, lab = clean_stats.clone(), true.clone()
        if stats is not None:
            s[rows] = stats
        if label is not None:
            lab[rows] = label
        return s, lab

    cases = {
        "clean": (clean_stats, true, 0),
        "backdoor": (*poisoned(data.backdoor, squared_stats[data.backdoor], 0), 0),
        "label_flip": (*poisoned(data.label_flip, label=5), 0),
        "gradient_matching": (
            *poisoned(data.gradient_matching, image_statistics(data.gm_pixels)),
            0,
        ),
        "slow_drip_1": (clean_stats, true, 1),
        "slow_drip_5": (clean_stats, true, 5),
        "benign_brighter": (image_statistics(brighter(train_raw)), true, 0),
    }
    drip_rows = torch.as_tensor(data.non_plane)

    # Calibrated PSI threshold per window size, from windows of trusted labels.
    trusted_labels = torch.as_tensor(targets[trusted])
    psi_level = {}
    for window in args.windows:
        size = window * args.batch_size
        levels = [
            psi(reference_labels, torch.bincount(trusted_labels[picks], minlength=10))
            for picks in torch.randint(len(trusted), (1000, size), generator=generator)
        ]
        psi_level[window] = float(np.quantile(levels, 1 - args.batch_rate))
    print(
        "calibrated PSI thresholds: "
        + ", ".join(f"{w} batches {v:.4f}" for w, v in psi_level.items())
    )

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="drift-baselines"):
        mlflow.log_params(vars(args))
        mlflow.log_metrics({f"psi_calibrated_level_w{w}": v for w, v in psi_level.items()})
        print(
            f"{'case':18s} " + "  ".join(f"KS / PSI / PSI-cal ({w} batches)" for w in args.windows)
        )
        for name, (stats, labels, per_batch) in cases.items():
            line = f"{name:18s}"
            for window in args.windows:
                size = window * args.batch_size
                ks_hits, psi_hits, calibrated_hits = 0, 0, 0
                for _ in range(args.trials):
                    pick = torch.randint(len(labels), (size,), generator=generator)
                    s, lab = stats[pick].clone(), labels[pick].clone()
                    if per_batch:  # slow-drip: squared non-plane images labelled plane
                        n = per_batch * window
                        drip = drip_rows[torch.randint(len(drip_rows), (n,), generator=generator)]
                        s[:n], lab[:n] = squared_stats[drip], 0
                    ks_hits += ks_alarm(reference_stats, s, args.batch_rate)
                    value = psi(reference_labels, torch.bincount(lab, minlength=10))
                    psi_hits += value > args.psi_alarm
                    calibrated_hits += value > psi_level[window]
                rates = {
                    f"ks_alarm_{name}_w{window}": ks_hits / args.trials,
                    f"psi_alarm_{name}_w{window}": psi_hits / args.trials,
                    f"psi_calibrated_alarm_{name}_w{window}": calibrated_hits / args.trials,
                }
                mlflow.log_metrics(rates)
                line += "   " + " / ".join(f"{v:6.1%}" for v in rates.values()) + "   "
            print(line)


if __name__ == "__main__":
    main()
