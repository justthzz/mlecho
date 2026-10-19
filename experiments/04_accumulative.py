"""Experiment 04: accumulative poisoning in online learning (Pang et al., 2021).

Starts from a trained (burn-in) model that keeps learning from a stream, one
update per incoming batch:
  1. accumulative phase: each incoming batch is slightly perturbed by the
     attacker (or left clean in the control run), then learned from
  2. trigger: one update on a clean trigger batch
Measured: test accuracy just before and just after the trigger update
(the single-step drop), compared against the control run.

First create the burn-in model (once, ~10 min):
       python experiments/01_clean_baseline.py \
           --save-weights data/checkpoints/burn-in-smallcnn-w64-seed42.pt
Run:   python experiments/04_accumulative.py                (attack)
       python experiments/04_accumulative.py --epsilon 0    (control: clean stream)
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
from collections.abc import Iterator

import mlflow
import numpy as np
import torch
from common import image_strip
from torch import nn
from torch.utils.data import DataLoader

from mlecho.attacks.accumulative import (
    AccumulativeConfig,
    craft_accumulative_batch,
    trigger_sensitivity,
)
from mlecho.bench.datasets import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_datasets
from mlecho.bench.models import SmallCNN
from mlecho.bench.online import online_step
from mlecho.bench.training import evaluate
from mlecho.core import get_device, load_weights, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burn-in", default="data/checkpoints/burn-in-smallcnn-w64-seed42.pt")
    parser.add_argument("--width", type=int, default=64, help="must match the burn-in model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01, help="online SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=200, help="accumulative phase length")
    parser.add_argument("--epsilon", type=float, default=8, help="max pixel change, in 1/255 units")
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--trade-off", type=float, default=0.02, help="lambda in the paper")
    parser.add_argument(
        "--tolerance", type=float, default=0.03, help="max val accuracy drop allowed"
    )
    parser.add_argument("--check-every", type=int, default=20, help="steps between val checks")
    parser.add_argument("--val-batch", type=int, default=500, help="attacker's validation batch")
    return parser.parse_args()


def endless(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches forever, like a live stream (reshuffles each pass)."""
    while True:
        yield from loader


def to_pixels(images: torch.Tensor) -> np.ndarray:
    """Normalised (N, 3, H, W) tensor -> raw (N, H, W, 3) uint8 pixels, for viewing."""
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    pixels = (images.cpu() * std + mean).clamp(0, 1) * 255
    return pixels.round().byte().permute(0, 2, 3, 1).numpy()


def main() -> None:
    args = parse_args()
    attacking = args.epsilon > 0
    config = AccumulativeConfig(
        epsilon=args.epsilon / 255, pgd_steps=args.pgd_steps, trade_off=args.trade_off
    )

    seed_everything(args.seed)
    device = get_device()
    train_set, val_set, test_set = get_cifar10_datasets(seed=args.seed, augment=False)
    stream = endless(
        DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    )
    val_loader = DataLoader(val_set, batch_size=500)
    test_loader = DataLoader(test_set, batch_size=500)

    def to_device(batch):
        return tuple(t.to(device) for t in batch)

    trigger = to_device(next(stream))  # the batch that will arrive at the very end
    attacker_val = to_device(next(iter(DataLoader(val_set, batch_size=args.val_batch))))

    model = load_weights(SmallCNN(width=args.width), args.burn_in, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    mlflow.set_experiment("mlecho-attacks")
    run_name = (
        f"accumulative-eps{args.epsilon:g}-T{args.steps}" if attacking else "accumulative-control"
    )
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(vars(args) | {"device": str(device), "attack": "accumulative"})
        start_val = evaluate(model, val_loader, loss_fn, device).accuracy
        mlflow.log_metric("val_acc", start_val, step=0)
        print(f"burn-in model: val accuracy {start_val:.1%}")

        steps_done = 0
        for step in range(1, args.steps + 1):
            images, labels = to_device(next(stream))
            if attacking:
                sensitivity = trigger_sensitivity(model, loss_fn, attacker_val, trigger)
                crafted = craft_accumulative_batch(
                    model,
                    loss_fn,
                    (images, labels),
                    sensitivity,
                    config,
                    CIFAR10_MEAN,
                    CIFAR10_STD,
                )
                if step == 1:
                    both = np.concatenate([to_pixels(images[:8]), to_pixels(crafted[:8])], axis=1)
                    mlflow.log_image(image_strip(both), "clean_top_vs_crafted_bottom.png")
                images = crafted
            online_step(model, optimizer, loss_fn, images, labels)
            steps_done = step

            if step % args.check_every == 0:
                val = evaluate(model, val_loader, loss_fn, device).accuracy
                mlflow.log_metric("val_acc", val, step=step)
                print(f"step {step}/{args.steps}  val accuracy {val:.1%}")
                if val < start_val - args.tolerance:
                    print("accuracy dropped beyond tolerance: attacker stops to stay secret")
                    break

        # The trigger: one ordinary update on a clean batch.
        before = evaluate(model, test_loader, loss_fn, device).accuracy
        online_step(model, optimizer, loss_fn, *trigger)
        after = evaluate(model, test_loader, loss_fn, device).accuracy

        mlflow.log_metrics(
            {
                "accumulative_steps_done": steps_done,
                "test_acc_before_trigger": before,
                "test_acc_after_trigger": after,
                "single_step_drop": before - after,
            }
        )
        print(f"\nTEST accuracy just BEFORE the trigger: {before:.1%}")
        print(f"TEST accuracy just AFTER  the trigger: {after:.1%}")
        print(f"single-step drop: {(before - after) * 100:.1f} points")


if __name__ == "__main__":
    main()
