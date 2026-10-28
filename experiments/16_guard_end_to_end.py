"""Experiment 16: the library end to end, as a user would run it (step 6).

1. `Guard.calibrate` on the 5,000 trusted images, from scratch, through the
   public API only (fold models, calibration, made-up attacks, classifier).
2. `save` and `load` it.
3. Protect live training episodes with the loaded guard (the setting of
   experiment 15), logging every event to MLflow and a JSON audit log.

The guard is calibrated independently of experiments 09-14 (its own random
folds and made-up attacks), so the numbers show what a fresh user gets,
not a copy of earlier results.

Run:   python experiments/16_guard_end_to_end.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from common import EpisodeMeasure, add_detection_args, detection_data, episode_scenarios

from mlecho import Guard, GuardConfig
from mlecho.bench.datasets import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_datasets
from mlecho.bench.models import SmallCNN
from mlecho.bench.stream import run_episode
from mlecho.core import get_device, load_weights, seed_everything
from mlecho.tracking.audit import AuditLog
from mlecho.tracking.mlflow_logger import MLflowLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    parser.add_argument("--save", default="data/checkpoints/guard-cifar10-seed42.pt")
    parser.add_argument("--recalibrate", action="store_true", help="even if --save exists")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["clean", "backdoor", "label_flip", "slow_drip_1", "benign_brighter"],
    )
    parser.add_argument("--episode-seed", type=int, default=1)
    parser.add_argument(
        "--burn-in", default="data/checkpoints/burn-in-smallcnn-w64-sgd0.01-seed42.pt"
    )
    parser.add_argument("--batches", type=int, default=400)
    parser.add_argument("--onset", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--measure-every", type=int, default=20)
    parser.add_argument("--measure-images", type=int, default=2000)
    parser.add_argument("--audit-dir", default="data/audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    targets = np.asarray(data.full.targets)
    trusted = np.concatenate([data.fit, data.held_out])
    _, _, test_set = get_cifar10_datasets(seed=args.seed, augment=False)

    mlflow.set_experiment("mlecho-guard")
    with mlflow.start_run(run_name=f"guard-end-to-end-seed{args.episode_seed}"):
        mlflow.log_params(vars(args) | {"device": str(device)})
        if args.recalibrate or not Path(args.save).exists():
            start = time.perf_counter()
            config = GuardConfig(pixel_mean=CIFAR10_MEAN, pixel_std=CIFAR10_STD, seed=args.seed)
            guard = Guard.calibrate(data.full.data[trusted], targets[trusted], 10, device, config)
            minutes = (time.perf_counter() - start) / 60
            guard.save(args.save)
            mlflow.log_metric("calibration_minutes", minutes)
            print(f"calibrated in {minutes:.1f} min, saved to {args.save}")
        calibration = Guard.load(args.save, device).monitor.calibration
        mlflow.log_metrics(
            {
                "picture_threshold": calibration.picture_threshold,
                "surprise_threshold": calibration.surprise_threshold,
                "mix_level": calibration.mix_level,
            }
        )

        curve_measure = EpisodeMeasure(test_set, device, args.measure_images, args.seed)
        final_measure = EpisodeMeasure(test_set, device, None, args.seed)
        cases = episode_scenarios(data, args.seed)
        for name in args.scenarios:
            seed_everything(args.episode_seed)
            model = load_weights(SmallCNN(width=64), args.burn_in, device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            audit = AuditLog(Path(args.audit_dir) / f"guard-{name}-seed{args.episode_seed}.jsonl")
            guard = Guard.load(
                args.save, device, loggers=[audit, MLflowLogger(prefix=f"guard_{name}")]
            )
            log = run_episode(
                model,
                optimizer,
                cases["clean"].raw,
                cases["clean"].labels,
                cases[name],
                guard,
                args.batches,
                args.onset,
                args.batch_size,
                curve_measure,
                args.measure_every,
                device,
                args.episode_seed,
            )
            guard.close()
            final = final_measure(model)
            first = log.actions[0] if log.actions else (-1, "none")
            mlflow.log_metrics(
                {f"{k}_{name}": v for k, v in final.items()}
                | {
                    f"poison_trained_{name}": log.poison_trained,
                    f"clean_dropped_{name}": log.clean_dropped,
                    f"first_action_batch_{name}": first[0],
                }
            )
            mlflow.set_tag(f"first_action_{name}", first[1])
            print(
                f"{name:18s} clean {final['clean_acc']:.1%}  ASR {final['asr']:5.1%}  "
                f"cat {final['cat_acc']:5.1%} | first action {first[1]} at {first[0]} | "
                f"poison learned {log.poison_trained}, clean dropped {log.clean_dropped} | "
                f"audit log {audit.path} ({len(audit.events)} events)"
            )


if __name__ == "__main__":
    main()
