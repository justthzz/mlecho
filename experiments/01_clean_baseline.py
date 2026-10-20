"""Experiment 01: train SmallCNN on clean CIFAR-10 and log everything to MLflow.

This is the "healthy model" reference. Later experiments poison the data
and compare against these numbers.

Run:   python experiments/01_clean_baseline.py            (30 epochs, augmented)
       python experiments/01_clean_baseline.py --epochs 5 --no-augment
       python experiments/01_clean_baseline.py --save-weights data/checkpoints/burn-in.pt
       python experiments/01_clean_baseline.py --optimizer sgd --lr 0.05 --schedule constant
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse

import mlflow
from common import add_training_args, log_model, train_and_log
from torch import nn

from mlecho.bench.datasets import get_cifar10_loaders
from mlecho.bench.models import SmallCNN
from mlecho.bench.training import evaluate
from mlecho.core import get_device, save_weights, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser)
    parser.add_argument(
        "--save-weights",
        metavar="PATH",
        help="also save the trained weights here (e.g. as the burn-in model for experiment 04)",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, seed=args.seed, augment=args.augment
    )
    model = SmallCNN(width=args.width).to(device)
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_experiment(args.mlflow_experiment or "mlecho-baselines")
    with mlflow.start_run(run_name="clean-smallcnn"):
        mlflow.log_params(vars(args) | {"device": str(device), "model": "SmallCNN"})
        train_and_log(model, train_loader, val_loader, loss_fn, device, args)

        # The test set is used exactly once, after all decisions are made.
        test = evaluate(model, test_loader, loss_fn, device)
        mlflow.log_metrics({"test_loss": test.loss, "test_acc": test.accuracy})
        log_model(model, test_loader)
        print(f"\nTEST accuracy: {test.accuracy:.1%}  (logged to MLflow)")
        if args.save_weights:
            save_weights(model, args.save_weights)
            print(f"weights saved to {args.save_weights}")


if __name__ == "__main__":
    main()
