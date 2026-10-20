"""Experiment 06: gradient-matching poisoning (Witches' Brew), clean-label and targeted.

The attacker wants ONE chosen test image (a dog) to be called a chosen wrong
class (frog). Using their own model (the clean burn-in model, trained with seed
42), they slightly change 1% of the training images, all real frogs that keep
their correct label. The victim then trains a new model from scratch, as
normal, on the poisoned training set (with a different seed). Success means
the victim calls the target dog a frog, while clean accuracy stays normal.

First create the attacker's model (once, ~10 min):
       python experiments/01_clean_baseline.py \\
           --save-weights data/checkpoints/burn-in-smallcnn-w64-seed42.pt
Run:   python experiments/06_gradient_matching.py --target-pick 0   (try 0, 1, 2, ...)
       python experiments/06_gradient_matching.py --epsilon 0       (control: no poison)
Easier setup (cat -> dog, the cat closest to "dog", victim fine-tunes the attacker's model):
       python experiments/06_gradient_matching.py --target-class 3 --poison-class 5 \\
           --target-order closest --victim-start attacker --epochs 5 --lr 1e-4 --schedule constant
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse

import mlflow
import numpy as np
import torch
from common import add_training_args, image_strip, log_model, train_and_log
from torch import nn

from mlecho.attacks.gradient_matching import (
    GradientMatchingConfig,
    craft_gradient_matching,
    write_poisons,
)
from mlecho.bench.datasets import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    cifar10_transform,
    get_cifar10_datasets,
    make_loaders,
)
from mlecho.bench.models import SmallCNN
from mlecho.bench.training import evaluate
from mlecho.core import get_device, load_weights, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser)
    parser.set_defaults(seed=1)  # the victim's seed; the attacker's model used 42
    parser.add_argument(
        "--attacker-model", default="data/checkpoints/burn-in-smallcnn-w64-seed42.pt"
    )
    parser.add_argument("--data-seed", type=int, default=42, help="train/val split and choices")
    parser.add_argument("--target-class", type=int, default=5, help="5 = dog")
    parser.add_argument("--poison-class", type=int, default=6, help="6 = frog (the wrong class)")
    parser.add_argument("--target-pick", type=int, default=0, help="which candidate target")
    parser.add_argument("--budget", type=float, default=0.01, help="share of train to poison")
    parser.add_argument("--epsilon", type=float, default=16, help="max pixel change, in /255")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument(
        "--control-picks", type=int, default=5, help="control: targets to check (no poison)"
    )
    parser.add_argument(
        "--target-order",
        choices=["random", "closest"],
        default="random",
        help="closest: targets the attacker's model already finds most like the poison class",
    )
    parser.add_argument(
        "--victim-start",
        choices=["scratch", "attacker"],
        default="scratch",
        help="victim trains a new model, or fine-tunes the attacker's (easiest case)",
    )
    return parser.parse_args()


def as_tensors(raw: np.ndarray) -> torch.Tensor:
    """Raw pixels (N, H, W, 3), 0-255 -> normalised tensors (N, 3, H, W), no augmentation."""
    plain = cifar10_transform(augment=False)
    return torch.stack([plain(image) for image in raw])


def candidate_targets(model, test_set, args, device) -> list[int]:
    """Test images of the target class that the attacker's model gets right.

    In a fixed shuffled order, or (`closest`) ordered by how likely the model
    already thinks they are the poison class, most likely first.
    """
    loader = torch.utils.data.DataLoader(test_set, batch_size=500)
    model.eval()
    with torch.no_grad():
        probabilities = torch.cat([model(x.to(device)).softmax(dim=1).cpu() for x, _ in loader])
    labels = torch.tensor(test_set.targets)
    correct = (labels == args.target_class) & (probabilities.argmax(dim=1) == args.target_class)
    indices = np.flatnonzero(correct.numpy())
    if args.target_order == "closest":
        return indices[np.argsort(-probabilities[indices, args.poison_class].numpy())].tolist()
    return np.random.default_rng(args.data_seed).permutation(indices).tolist()


def main() -> None:
    args = parse_args()
    control = args.epsilon == 0
    config = GradientMatchingConfig(epsilon=args.epsilon / 255, steps=args.steps)

    seed_everything(args.data_seed)
    device = get_device()
    train_set, val_set, test_set = get_cifar10_datasets(seed=args.data_seed, augment=args.augment)
    full = train_set.dataset
    loss_fn = nn.CrossEntropyLoss()

    # 1. The attacker picks the target(s) and the poison-class images to change.
    attacker = load_weights(SmallCNN(width=args.width, norm=args.norm), args.attacker_model, device)
    candidates = candidate_targets(attacker, test_set, args, device)
    targets = candidates[: args.control_picks] if control else [candidates[args.target_pick]]
    rng = np.random.default_rng(args.data_seed)
    pool = [i for i in train_set.indices if full.targets[i] == args.poison_class]
    positions = np.sort(rng.choice(pool, round(args.budget * len(train_set)), replace=False))
    if control:
        positions = positions[:0]  # nothing poisoned

    mlflow.set_experiment(args.mlflow_experiment or "mlecho-attacks")
    name = "control" if control else f"pick{args.target_pick}-eps{args.epsilon:g}"
    name += f"-{CIFAR10_CLASSES[args.target_class]}-to-{CIFAR10_CLASSES[args.poison_class]}"
    name += f"-{args.target_order}-{args.victim_start}"
    with mlflow.start_run(run_name=f"gradient-matching-{name}"):
        mlflow.log_params(
            vars(args)
            | {"device": str(device), "model": "SmallCNN", "attack": "gradient matching"}
            | {"poisoned_count": len(positions), "target_test_index": targets[0]}
        )

        # 2. Craft the poisons and put them into the training set (labels unchanged).
        if not control:
            originals = full.data[positions].copy()
            target_image = test_set[targets[0]][0].unsqueeze(0).to(device)
            wrong_label = torch.tensor([args.poison_class], device=device)
            poisons = (
                as_tensors(originals).to(device),
                torch.full((len(positions),), args.poison_class, device=device),
            )
            crafted = craft_gradient_matching(
                attacker,
                loss_fn,
                poisons,
                (target_image, wrong_label),
                config,
                CIFAR10_MEAN,
                CIFAR10_STD,
                torch.Generator().manual_seed(args.data_seed),
            )
            write_poisons(full, positions, crafted, CIFAR10_MEAN, CIFAR10_STD)
            change = np.abs(full.data[positions].astype(int) - originals.astype(int))
            mlflow.log_metrics(
                {"max_pixel_change": change.max(), "mean_pixel_change": change.mean()}
            )
            difference = (128 + 4 * (full.data[positions].astype(int) - originals)).clip(0, 255)
            examples = np.concatenate(
                [
                    image_strip(originals),
                    image_strip(full.data[positions]),
                    image_strip(difference.astype(np.uint8)),
                ]
            )
            mlflow.log_image(examples, "poisons_before_after_difference.png")
            mlflow.log_image(image_strip(test_set.data[targets], count=1), "target.png")
            print(f"poisoned {len(positions)} images; largest pixel change {change.max()}/255")

        # 3. The victim trains as normal: a new model, or fine-tuning the attacker's.
        seed_everything(args.seed)
        train_loader, val_loader, test_loader = make_loaders(
            train_set, val_set, test_set, args.batch_size, args.seed
        )
        victim = SmallCNN(width=args.width, norm=args.norm).to(device)
        if args.victim_start == "attacker":
            victim.load_state_dict(attacker.state_dict())
        train_and_log(victim, train_loader, val_loader, loss_fn, device, args)

        # 4. Did the target get the wrong class? Is clean accuracy normal?
        clean = evaluate(victim, test_loader, loss_fn, device)
        victim.eval()
        with torch.no_grad():
            images = torch.stack([test_set[i][0] for i in targets]).to(device)
            probabilities = victim(images).softmax(dim=1).cpu()
        predicted = probabilities.argmax(dim=1)
        fooled = (predicted == args.poison_class).float()
        mlflow.log_metrics(
            {
                "test_acc": clean.accuracy,
                "test_loss": clean.loss,
                "target_fooled_rate": fooled.mean().item(),
                "target_prob_poison_class": probabilities[:, args.poison_class].mean().item(),
            }
        )
        log_model(victim, test_loader)

        print(f"\nCLEAN test accuracy: {clean.accuracy:.1%}")
        for index, pred, prob in zip(targets, predicted, probabilities, strict=True):
            print(
                f"target test image {index}: called {CIFAR10_CLASSES[pred]} "
                f"(P({CIFAR10_CLASSES[args.poison_class]}) = {prob[args.poison_class]:.1%})"
            )
        print(f"targets fooled: {int(fooled.sum())} of {len(targets)}")


if __name__ == "__main__":
    main()
