"""Experiment 10: how much do the checks slow training down? (step 3.5, NFR-P1 < 5%)

Times the victim's normal training (SmallCNN, or ResNet-18 with --victim
resnet18; batches of 128, augmentation, Adam) for a fixed number of batches,
with and without the detection checks running on each batch: the picture
check (CNN-VAE, worst 4x4 patch) and the label check (one reference model),
and the whole MLEcho Guard as a user runs it (`check` before and `update`
after every step: 5 fold models, quarantine, a verdict, a policy decision and
a checkpoint every 10 batches; saved by experiment 16).
Each setting is repeated, interleaved with the others, so slow-downs of the
machine affect all settings alike. Time includes data loading, as in real
training. Other programs can only make a run slower, never faster, so each
setting's fastest run is the main measure (the median is logged too).

Run:   python experiments/10_detection_overhead.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
import time

import mlflow
import numpy as np
import torch
import torchvision
from torch import nn

from mlecho.bench.datasets import get_cifar10_loaders
from mlecho.bench.models import SmallCNN
from mlecho.core import get_device, load_weights, seed_everything
from mlecho.detection.label_check import ReferenceCNN, label_scores
from mlecho.detection.vae import ConvVAE, reconstruction_scores
from mlecho.guard import Guard

SETTINGS = {  # name -> (run the picture check every n-th batch, label check every n-th); 0 = never
    "no checks": (0, 0),
    "picture check, every batch": (1, 0),
    "label check, every batch": (0, 1),
    "both checks, every batch": (1, 1),
    "both checks, every 4th batch": (4, 4),
    "both checks, every 8th batch": (8, 8),
    "picture every batch, label every 8th": (1, 8),
    "MLEcho Guard, every batch": "guard",
    "MLEcho Guard, every 4th quiet batch": "guard4",
    "MLEcho Guard, every 8th quiet batch": "guard8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=200, help="timed batches per run")
    parser.add_argument("--warmup", type=int, default=20, help="untimed batches first")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--victim", choices=["smallcnn", "resnet18"], default="smallcnn")
    parser.add_argument("--vae", default="data/checkpoints/vae-c16-z32-seed42.pt")
    parser.add_argument("--reference", default="data/checkpoints/reference-w32-seed42-fold0.pt")
    parser.add_argument("--guard", default="data/checkpoints/guard-cifar10-seed42.pt")
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="random shifts and flips (the Guard must see batches before augmentation)",
    )
    parser.add_argument("--mlflow-experiment", default="mlecho-detection")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    """Wait until the GPU has finished, so the clock measures real work."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def make_victim(name: str) -> nn.Module:
    """The model being trained: our SmallCNN, or a standard ResNet-18 (11M weights)."""
    if name == "resnet18":
        return torchvision.models.resnet18(num_classes=10)
    return SmallCNN()


def timed_training(loader, device, vae, reference, setting, args) -> float:
    """Seconds for `args.batches` training steps with the given checks."""
    guard = None
    if isinstance(setting, str):  # "guard", "guard4", "guard8": the whole Guard
        every = int(setting[5:] or 1)
        guard, setting = Guard.load(args.guard, device, check_every=every), (0, 0)
    picture_every, label_every = setting
    seed_everything(args.seed)
    model = make_victim(args.victim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if guard is not None:
        guard.watch(model, optimizer)
    loss_fn = nn.CrossEntropyLoss()
    batches = iter(loader)
    model.train()
    for step in range(args.warmup + args.batches):
        if step == args.warmup:
            synchronize(device)
            start = time.perf_counter()
        images, labels = next(batches)
        images, labels = images.to(device), labels.to(device)
        if guard is not None:
            keep = guard.check(images, labels)
            images, labels = images[keep], labels[keep]
        if picture_every and step % picture_every == 0:
            reconstruction_scores(vae, images, patch=4)
        if label_every and step % label_every == 0:
            label_scores(reference, images, labels)
        loss = loss_fn(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if guard is not None:
            guard.update(loss.item())
    synchronize(device)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    device = get_device()
    loader, _, _ = get_cifar10_loaders(batch_size=128, seed=args.seed, augment=args.augment)
    vae = load_weights(ConvVAE(), args.vae, device).eval()
    reference = load_weights(ReferenceCNN(), args.reference, device).eval()

    times = {name: [] for name in SETTINGS}
    for repeat in range(args.repeats):  # interleaved: every setting once per round
        for name, setting in SETTINGS.items():
            seconds = timed_training(loader, device, vae, reference, setting, args)
            times[name].append(seconds)
            print(f"round {repeat + 1}: {name:30s} {seconds:6.2f} s")

    mlflow.set_experiment(args.mlflow_experiment)
    plain = "" if args.augment else "-no-augment"
    with mlflow.start_run(run_name=f"detection-overhead-{args.victim}{plain}"):
        mlflow.log_params(vars(args) | {"device": str(device)})
        fastest = {name: min(values) for name, values in times.items()}
        typical = {name: float(np.median(values)) for name, values in times.items()}
        print(f"\nfastest and median of {args.repeats} runs of {args.batches} batches:")
        for name in times:
            overhead = fastest[name] / fastest["no checks"] - 1
            median_overhead = typical[name] / typical["no checks"] - 1
            key = name.replace(", ", "_").replace(" ", "_").replace("-", "")
            mlflow.log_metrics(
                {
                    f"seconds_{key}": fastest[name],
                    f"overhead_{key}": overhead,
                    f"median_overhead_{key}": median_overhead,
                }
            )
            print(
                f"{name:38s} fastest {fastest[name]:6.2f} s  overhead {overhead:+6.1%}"
                f"   (median {median_overhead:+6.1%})"
            )


if __name__ == "__main__":
    main()
