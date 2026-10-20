"""Experiment 03: train SmallCNN on CIFAR-10 with targeted label-flipping.

Some training cats are relabelled "dog" (pixels untouched). Measured on the
1,000 clean cat test images:
- cat accuracy: share still predicted "cat" (drops if the attack works)
- flip rate: share predicted "dog" (rises if the attack works)
plus overall clean test accuracy.

Run:   python experiments/03_label_flip.py
       python experiments/03_label_flip.py --poison-rate 0   (clean model's numbers)
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse

import mlflow
from common import add_training_args, image_strip, log_model, train_and_log
from torch import nn

from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.bench.datasets import CIFAR10_CLASSES, get_cifar10_datasets, make_loaders
from mlecho.bench.models import SmallCNN
from mlecho.bench.training import evaluate, predict
from mlecho.core import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser)
    parser.add_argument("--poison-rate", type=float, default=0.05)
    parser.add_argument("--source-class", type=int, default=3, help="3 = cat")
    parser.add_argument("--target-class", type=int, default=5, help="5 = dog")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabelFlipConfig(
        source_class=args.source_class,
        target_class=args.target_class,
        poison_rate=args.poison_rate,
    )
    source, target = CIFAR10_CLASSES[args.source_class], CIFAR10_CLASSES[args.target_class]

    seed_everything(args.seed)
    device = get_device()
    train_set, val_set, test_set = get_cifar10_datasets(seed=args.seed, augment=args.augment)
    flipped = flip_labels(train_set, config, seed=args.seed)
    train_loader, val_loader, test_loader = make_loaders(
        train_set, val_set, test_set, args.batch_size, args.seed
    )
    model = SmallCNN(width=args.width).to(device)
    loss_fn = nn.CrossEntropyLoss()
    print(f"relabelled {len(flipped)} training {source} images as '{target}'")

    mlflow.set_experiment(args.mlflow_experiment or "mlecho-attacks")
    with mlflow.start_run(run_name=f"label-flip-{args.poison_rate:g}"):
        mlflow.log_params(
            vars(args)
            | {"device": str(device), "model": "SmallCNN", "attack": "label-flip"}
            | {"poisoned_count": len(flipped)}
        )
        if flipped:  # these look like normal cats; only their labels say "dog"
            mlflow.log_image(image_strip(train_set.dataset.data[flipped]), "poisoned_examples.png")
        train_and_log(model, train_loader, val_loader, loss_fn, device, args)

        # The test set is used once, after training is finished.
        clean = evaluate(model, test_loader, loss_fn, device)
        predictions, truths = predict(model, test_loader, device)
        is_source = truths == args.source_class
        source_accuracy = (predictions[is_source] == args.source_class).float().mean().item()
        flip_rate = (predictions[is_source] == args.target_class).float().mean().item()

        mlflow.log_metrics(
            {
                "test_acc": clean.accuracy,
                "test_loss": clean.loss,
                "source_class_acc": source_accuracy,
                "flip_rate": flip_rate,
            }
        )
        log_model(model, test_loader)
        print(f"\nCLEAN test accuracy:      {clean.accuracy:.1%}")
        print(f"{source} test images called '{source}': {source_accuracy:.1%}")
        print(f"{source} test images called '{target}': {flip_rate:.1%}  (flip rate)")


if __name__ == "__main__":
    main()
