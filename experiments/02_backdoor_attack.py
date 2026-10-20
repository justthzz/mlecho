"""Experiment 02: train SmallCNN on backdoor-poisoned CIFAR-10.

Shows that the attack is both hidden and effective:
- clean test accuracy stays close to the clean baseline (experiment 01)
- the attack success rate (ASR) on triggered test images is high

Run:   python experiments/02_backdoor_attack.py
       python experiments/02_backdoor_attack.py --poison-rate 0.01
       python experiments/02_backdoor_attack.py --poison-rate 0   (clean model's ASR)
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse

import mlflow
from common import add_training_args, image_strip, log_model, train_and_log
from torch import nn
from torch.utils.data import DataLoader

from mlecho.attacks.backdoor import (
    BackdoorConfig,
    make_triggered_test_set,
    poison_train_set,
)
from mlecho.bench.datasets import get_cifar10_datasets, make_loaders
from mlecho.bench.models import SmallCNN
from mlecho.bench.training import evaluate
from mlecho.core import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser)
    parser.add_argument("--poison-rate", type=float, default=0.05)
    parser.add_argument("--target-class", type=int, default=0, help="0 = plane")
    parser.add_argument("--patch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BackdoorConfig(
        target_class=args.target_class,
        poison_rate=args.poison_rate,
        patch_size=args.patch_size,
    )

    seed_everything(args.seed)
    device = get_device()
    train_set, val_set, test_set = get_cifar10_datasets(seed=args.seed, augment=args.augment)
    poisoned = poison_train_set(train_set, config, seed=args.seed)
    train_loader, val_loader, test_loader = make_loaders(
        train_set, val_set, test_set, args.batch_size, args.seed
    )
    triggered_loader = DataLoader(
        make_triggered_test_set(test_set, config), batch_size=args.batch_size
    )
    model = SmallCNN(width=args.width).to(device)
    loss_fn = nn.CrossEntropyLoss()
    print(f"poisoned {len(poisoned)} of {len(train_set)} training images")

    mlflow.set_experiment("mlecho-attacks")
    with mlflow.start_run(run_name=f"backdoor-{args.poison_rate:g}"):
        mlflow.log_params(
            vars(args)
            | {"device": str(device), "model": "SmallCNN", "attack": "backdoor"}
            | {"poisoned_count": len(poisoned)}
        )
        if poisoned:
            mlflow.log_image(image_strip(train_set.dataset.data[poisoned]), "poisoned_examples.png")
        train_and_log(model, train_loader, val_loader, loss_fn, device, args)

        # Both test sets are used exactly once, after training is finished.
        clean = evaluate(model, test_loader, loss_fn, device)
        attack = evaluate(model, triggered_loader, loss_fn, device)
        mlflow.log_metrics(
            {
                "test_acc": clean.accuracy,
                "test_loss": clean.loss,
                "attack_success_rate": attack.accuracy,
            }
        )
        log_model(model, test_loader)
        print(f"\nCLEAN test accuracy:  {clean.accuracy:.1%}")
        print(f"ATTACK success rate:  {attack.accuracy:.1%}  (triggered images called the target)")


if __name__ == "__main__":
    main()
