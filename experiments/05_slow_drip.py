"""Experiment 05: slow-drip backdoor poisoning in online learning.

A trained (burn-in) model keeps learning from a stream, one update per incoming
batch. In every batch a small share of images gets the backdoor patch and the
label "plane". Every few steps we measure clean test accuracy (does the model
still look healthy?) and the attack success rate (is the backdoor building up?).

First create the burn-in model (once, ~10 min):
       python experiments/01_clean_baseline.py --optimizer sgd --lr 0.01 --schedule constant \\
           --save-weights data/checkpoints/burn-in-smallcnn-w64-sgd0.01-seed42.pt
Run:   python experiments/05_slow_drip.py                     (5% of each batch)
       python experiments/05_slow_drip.py --per-batch-rate 0  (control: clean stream)
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
from collections.abc import Iterator

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from mlecho.attacks.backdoor import BackdoorConfig, make_triggered_test_set
from mlecho.attacks.slowdrip import drip_backdoor
from mlecho.bench.datasets import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_datasets
from mlecho.bench.models import SmallCNN
from mlecho.bench.online import online_step
from mlecho.bench.training import evaluate
from mlecho.core import get_device, load_weights, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--burn-in", default="data/checkpoints/burn-in-smallcnn-w64-sgd0.01-seed42.pt"
    )
    parser.add_argument("--width", type=int, default=64, help="must match the burn-in model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01, help="online SGD learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=450, help="450 x 100 = one pass of train")
    parser.add_argument("--per-batch-rate", type=float, default=0.05)
    parser.add_argument("--target-class", type=int, default=0, help="0 = plane")
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--mlflow-experiment", default="mlecho-attacks")
    return parser.parse_args()


def endless(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches forever, like a live stream (reshuffles each pass)."""
    while True:
        yield from loader


def main() -> None:
    args = parse_args()
    backdoor = BackdoorConfig(target_class=args.target_class)

    seed_everything(args.seed)
    device = get_device()
    train_set, _, test_set = get_cifar10_datasets(seed=args.seed, augment=False)
    stream = endless(
        DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    )
    test_loader = DataLoader(test_set, batch_size=500)
    triggered_loader = DataLoader(make_triggered_test_set(test_set, backdoor), batch_size=500)
    drip_generator = torch.Generator().manual_seed(args.seed + 1)

    model = load_weights(SmallCNN(width=args.width), args.burn_in, device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    def measure(step: int) -> None:
        clean = evaluate(model, test_loader, loss_fn, device).accuracy
        attack = evaluate(model, triggered_loader, loss_fn, device).accuracy
        mlflow.log_metrics({"test_acc": clean, "attack_success_rate": attack}, step=step)
        print(
            f"step {step:4d}  clean test accuracy {clean:.1%}  |  attack success rate {attack:.1%}"
        )

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=f"slow-drip-{args.per_batch_rate:g}"):
        mlflow.log_params(vars(args) | {"device": str(device), "attack": "slow-drip backdoor"})
        measure(step=0)
        poisoned_total = 0
        for step in range(1, args.steps + 1):
            images, labels = next(stream)
            images, labels, is_poisoned = drip_backdoor(
                images,
                labels,
                args.per_batch_rate,
                backdoor,
                CIFAR10_MEAN,
                CIFAR10_STD,
                drip_generator,
            )
            poisoned_total += int(is_poisoned.sum())
            online_step(model, optimizer, loss_fn, images.to(device), labels.to(device))
            if step % args.check_every == 0 or step == args.steps:
                measure(step)

        mlflow.log_metric("poisoned_images_total", poisoned_total)
        print(f"\npoisoned images fed in: {poisoned_total} of {args.steps * args.batch_size}")


if __name__ == "__main__":
    main()
