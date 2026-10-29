"""Experiment 17: would a stronger reference model fix the label-flip weak spot?

The label check's reference models (experiment 09) get ~65% right, and they
confuse cats and dogs. That is why cat -> dog flips at 2% are missed and why
about a third of the flipped images pass the label-flip quarantine (experiment
15): it drops a dog-labelled image only if the model thinks "cat" is likelier
than "dog".

Here, four reference models are trained on the same 4,000 trusted images
(fold 0's training part) and compared on what the quarantine needs:
- flipped cats caught: test cats with P(cat) > P(dog) (would be dropped if labelled dog)
- real dogs dropped: test dogs with P(cat) > P(dog) (the price)
and on speed: the reference model runs on every checked batch (NFR-P1).

Run:   python experiments/17_reference_model_variants.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
import time

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from mlecho.bench.datasets import get_cifar10_datasets, normalise
from mlecho.core import get_device, seed_everything
from mlecho.detection.folds import _Shifted
from mlecho.detection.label_check import ReferenceCNN

CAT, DOG = 3, 5


class DeepReference(nn.Module):
    """ReferenceCNN with two convolutions per stage (a small VGG)."""

    def __init__(self, num_classes: int = 10, width: int = 32):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, width), block(width, 2 * width), block(2 * width, 4 * width)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(4 * width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


VARIANTS = {  # name -> (model, weight decay, label smoothing)
    "current-w32": (lambda: ReferenceCNN(width=32), 0.0, 0.0),
    "wide-w64": (lambda: ReferenceCNN(width=64), 0.0, 0.0),
    "w32-decay-smoothing": (lambda: ReferenceCNN(width=32), 5e-4, 0.1),
    "deep-w32-decay-smoothing": (lambda: DeepReference(width=32), 5e-4, 0.1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mlflow-experiment", default="mlecho-detection")
    return parser.parse_args()


@torch.no_grad()
def log_probabilities(model: nn.Module, images: torch.Tensor, device) -> torch.Tensor:
    model.eval()
    return torch.cat([F.log_softmax(model(c.to(device)), 1).cpu() for c in images.split(1000)])


@torch.no_grad()
def milliseconds_per_batch(model: nn.Module, batch: torch.Tensor, device) -> float:
    model.eval()
    batch = batch.to(device)
    model(batch)
    if device.type == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(50):
        model(batch)
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - start) / 50 * 1000


def main() -> None:
    args = parse_args()
    device = get_device()
    train_set, val_set, test_set = get_cifar10_datasets(seed=args.seed, augment=False)
    full = train_set.dataset
    trusted = np.random.default_rng(args.seed).permutation(val_set.indices)
    images = normalise(full.data[trusted])
    labels = torch.as_tensor(np.asarray(full.targets)[trusted])
    test_images, test_labels = normalise(test_set.data), torch.as_tensor(test_set.targets)
    cats, dogs = test_labels == CAT, test_labels == DOG

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="reference-model-variants"):
        mlflow.log_params(vars(args) | {"device": str(device), "train_images": 4000})
        for name, (make, decay, smoothing) in VARIANTS.items():
            seed_everything(args.seed)
            model = make().to(device)
            loader = DataLoader(
                _Shifted(images[:4000], labels[:4000]),
                batch_size=args.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(args.seed),
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            start = time.perf_counter()
            for _ in range(args.epochs):
                model.train()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    loss = F.cross_entropy(model(x), y, label_smoothing=smoothing)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            minutes = (time.perf_counter() - start) / 60

            held_out = log_probabilities(model, images[4000:], device)
            test = log_probabilities(model, test_images, device)
            results = {
                "held_out_accuracy": (held_out.argmax(1) == labels[4000:]).float().mean().item(),
                "test_accuracy": (test.argmax(1) == test_labels).float().mean().item(),
                "flipped_cats_caught": (test[cats, CAT] > test[cats, DOG]).float().mean().item(),
                "real_dogs_dropped": (test[dogs, CAT] > test[dogs, DOG]).float().mean().item(),
                "ms_per_batch": milliseconds_per_batch(model, test_images[:128], device),
                "train_minutes": minutes,
            }
            mlflow.log_metrics({f"{name}_{k}": v for k, v in results.items()})
            print(
                f"{name:26s} held-out {results['held_out_accuracy']:.1%} "
                f"test {results['test_accuracy']:.1%} | flipped cats caught "
                f"{results['flipped_cats_caught']:.1%}, real dogs dropped "
                f"{results['real_dogs_dropped']:.1%} | {results['ms_per_batch']:.1f} ms/batch",
                flush=True,
            )


if __name__ == "__main__":
    main()
