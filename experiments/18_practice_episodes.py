"""Experiment 18: practice episodes for the PPO agent (step 7).

Builds the agent's practice world: `--per-kind` made-up streams of every kind
(clean, backdoor, slow-drip, label-flip, harmless shift; `bench/env/practice.py`),
each with a random onset, scored once by the guard of experiment 15 (same
fold models, calibration and classifier). The real attacks of experiment 15
are not used: random triggers, random flip pairs (never cat -> dog), random
shifts. Saves the scored streams to one file for experiment 19.

Run:   python experiments/18_practice_episodes.py
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import torch
from common import add_detection_args, add_guard_args, detection_data, guard_parts

from mlecho.bench.env.practice import KINDS, cache_episode, random_practice_attack
from mlecho.classification.monitor import Monitor
from mlecho.core import get_device, seed_everything

EVALUATION_FLIP = (3, 5)  # cat -> dog, experiment 15's label-flip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    add_guard_args(parser)
    parser.add_argument("--per-kind", type=int, default=24)
    parser.add_argument("--batches", type=int, default=400)
    parser.add_argument("--onsets", type=int, nargs=2, default=[60, 200], help="low, high")
    parser.add_argument("--practice-seed", type=int, default=7, help="not an evaluation seed")
    parser.add_argument("--out", default="data/env/practice-seed7.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    train_raw = data.full.data[data.train]
    train_labels = np.asarray(data.full.targets)[data.train]
    score, calibration, classifier = guard_parts(args, data, device)

    rng = np.random.default_rng(args.practice_seed)
    plan = [kind for kind in KINDS for _ in range(args.per_kind)]
    episodes = []
    mlflow.set_experiment("mlecho-ppo")
    with mlflow.start_run(run_name=f"practice-episodes-seed{args.practice_seed}"):
        mlflow.log_params(vars(args) | {"device": str(device)})
        start = time.perf_counter()
        for k, kind in enumerate(plan):
            attack = random_practice_attack(
                rng, kind, train_raw.shape[1], 10, tuple(args.onsets), {EVALUATION_FLIP}
            )
            monitor = Monitor(
                score.vaes, score.references, calibration, classifier, args.window, args.patch, k
            )
            episodes.append(
                cache_episode(
                    attack,
                    train_raw,
                    train_labels,
                    monitor,
                    args.batches,
                    args.batch_size,
                    device,
                    seed=args.practice_seed * 1000 + k,
                )
            )
            if (k + 1) % 10 == 0:
                print(f"{k + 1}/{len(plan)} episodes scored", flush=True)
        minutes = (time.perf_counter() - start) / 60
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(episodes, args.out)
        poisoned = Counter()
        for episode in episodes:
            poisoned[episode.attack.kind] += int(episode.poisoned.sum())
        mlflow.log_metrics(
            {"minutes": minutes, "episodes": len(episodes)}
            | {f"poisoned_per_episode_{k}": v / args.per_kind for k, v in poisoned.items()}
        )
        print(f"{len(episodes)} episodes in {minutes:.1f} min -> {args.out}")
        for kind in KINDS:
            print(f"  {kind:13s} poisoned images per episode: {poisoned[kind] / args.per_kind:.0f}")


if __name__ == "__main__":
    main()
