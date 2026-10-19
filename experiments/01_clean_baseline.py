"""Experiment 01: train SmallCNN on clean CIFAR-10 and log everything to MLflow.

This is the "healthy model" reference. Later experiments poison the data
and compare against these numbers.

Run:   python experiments/01_clean_baseline.py            (30 epochs, augmented)
       python experiments/01_clean_baseline.py --epochs 5 --no-augment
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
import time

import mlflow
import torch
from torch import nn

from mlecho.bench.datasets import get_cifar10_loaders
from mlecho.bench.models import SmallCNN
from mlecho.bench.training import evaluate, train_one_epoch
from mlecho.core import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=64, help="SmallCNN channels in block 1")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,  # gives both --augment and --no-augment
        default=True,
        help="randomly shift and mirror training images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, seed=args.seed, augment=args.augment
    )

    model = SmallCNN(width=args.width).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Big learning steps early, then smoothly smaller ones (cosine curve) to settle in.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mlflow.set_experiment("mlecho-baselines")
    with mlflow.start_run(run_name="clean-smallcnn"):
        mlflow.log_params(vars(args) | {"device": str(device), "model": "SmallCNN"})

        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val = evaluate(model, val_loader, loss_fn, device)
            seconds = time.perf_counter() - start
            lr = scheduler.get_last_lr()[0]
            scheduler.step()

            mlflow.log_metrics(
                {
                    "train_loss": train.loss,
                    "train_acc": train.accuracy,
                    "val_loss": val.loss,
                    "val_acc": val.accuracy,
                    "epoch_seconds": seconds,
                    "lr": lr,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch}/{args.epochs}  "
                f"train loss {train.loss:.3f} acc {train.accuracy:.1%}  |  "
                f"val loss {val.loss:.3f} acc {val.accuracy:.1%}  |  {seconds:.0f}s"
            )

        # The test set is used exactly once, after all decisions are made.
        test = evaluate(model, test_loader, loss_fn, device)
        mlflow.log_metrics({"test_loss": test.loss, "test_acc": test.accuracy})

        # MLflow saves the model by running one example image through it.
        example_images, _ = next(iter(test_loader))
        mlflow.pytorch.log_model(
            model.cpu(), name="model", input_example=example_images[:1].numpy()
        )
        print(f"\nTEST accuracy: {test.accuracy:.1%}  (logged to MLflow)")


if __name__ == "__main__":
    main()
