"""Experiment 15: does responding help? Live training episodes (step 5, RQ3).

A burn-in model (SGD, 83.3% test, experiment 01) keeps learning online, one
update per batch of 128 (SGD lr 0.01, momentum 0.9, as experiment 05). The
first `--onset` batches are clean; then the attack starts and lasts until
the end. The guard (cross-fitted detectors, calibration and the attack
classifier from experiment 14) watches every batch. Three policies:
- none: no response (the control);
- rules: MLEcho's rule table (`mlecho.response.rules`);
- quarantine-everything: the same alarms, no attack type (ablation).

Scenarios: clean, backdoor 5%, label-flip cat -> dog 5%, slow-drip 1 and 5
per batch, gradient matching, and a harmless brightness change (x1.3).
Measured every `--measure-every` batches on test subsets, and at the end on
the full test set: clean accuracy, ASR (yellow trigger), cat accuracy, cats
called dog, and P(dog) of the gradient-matching target (test cat 8183).

Run:   python experiments/15_response_episodes.py --policy rules --episode-seed 0
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
from dataclasses import asdict

import mlflow
import numpy as np
import torch
from common import FoldScorer, add_detection_args, detection_data

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger, poison_train_set
from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.bench.baselines.response import QuarantineEverything
from mlecho.bench.datasets import get_cifar10_datasets, normalise
from mlecho.bench.models import SmallCNN
from mlecho.bench.stream import Scenario, run_episode
from mlecho.classification.classifier import AttackClassifier
from mlecho.classification.features import FEATURES, calibrate
from mlecho.classification.monitor import Monitor
from mlecho.classification.synthetic import Shift, apply_shift
from mlecho.core import get_device, load_weights, seed_everything
from mlecho.response.rules import RulePolicy

POLICIES = {"none": None, "rules": RulePolicy(), "quarantine-everything": QuarantineEverything()}
GM_TARGET = 8183  # test cat the gradient-matching poisons push towards "dog" (experiment 06)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    parser.add_argument("--policy", choices=list(POLICIES), default="rules")
    parser.add_argument("--episode-seed", type=int, default=0, help="stream order, drip, folds")
    parser.add_argument("--scenarios", nargs="+", default=None, help="default: all")
    parser.add_argument(
        "--burn-in", default="data/checkpoints/burn-in-smallcnn-w64-sgd0.01-seed42.pt"
    )
    parser.add_argument("--batches", type=int, default=400)
    parser.add_argument("--onset", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--measure-every", type=int, default=20)
    parser.add_argument("--measure-images", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--vae-prefix", default="data/checkpoints/vae-c16-z32-seed42-fold")
    parser.add_argument("--reference-prefix", default="data/checkpoints/reference-w32-seed42-fold")
    parser.add_argument("--classifier", default="data/checkpoints/attack-classifier-seed42.pt")
    return parser.parse_args()


class Measure:
    """Clean accuracy, ASR, cat accuracy, cats called dog, and the target's P(dog)."""

    def __init__(self, test_set, device, images: int | None, seed: int):
        raw, labels = test_set.data, np.asarray(test_set.targets)
        rows = np.arange(len(labels))
        if images is not None:  # a fixed random subset, for the curves
            rows = np.random.default_rng(seed).choice(len(labels), images, replace=False)
        non_plane = rows[labels[rows] != 0]
        cats = np.flatnonzero(labels == 3)
        self.device = device
        self.clean = (normalise(raw[rows]), torch.as_tensor(labels[rows]))
        self.triggered = normalise(add_trigger(raw[non_plane], BackdoorConfig()))
        self.cats = normalise(raw[cats if images is None else cats[: images // 2]])
        self.target = normalise(raw[[GM_TARGET]])

    @torch.no_grad()
    def predict(self, model, images) -> torch.Tensor:
        return torch.cat(
            [model(chunk.to(self.device)).softmax(dim=1).cpu() for chunk in images.split(1000)]
        )

    def __call__(self, model) -> dict:
        model.eval()
        clean_images, clean_labels = self.clean
        cats = self.predict(model, self.cats).argmax(dim=1)
        return {
            "clean_acc": (self.predict(model, clean_images).argmax(1) == clean_labels)
            .float()
            .mean()
            .item(),
            "asr": (self.predict(model, self.triggered).argmax(1) == 0).float().mean().item(),
            "cat_acc": (cats == 3).float().mean().item(),
            "cat_as_dog": (cats == 5).float().mean().item(),
            "target_dog": self.predict(model, self.target)[0, 5].item(),
        }


def scenarios(data, seed) -> dict[str, Scenario]:
    """The training data after the onset, for every scenario."""
    full, rows = data.full, data.train
    raw, labels = full.data[rows], np.asarray(full.targets)[rows]
    row_of = {position: k for k, position in enumerate(rows)}
    none = np.zeros(len(rows), dtype=bool)
    out = {"clean": Scenario("clean", raw, labels, none)}

    view, _, _ = get_cifar10_datasets(seed=seed, augment=False)
    chosen = [row_of[i] for i in poison_train_set(view, BackdoorConfig(), seed)]
    mask = none.copy()
    mask[chosen] = True
    out["backdoor"] = Scenario(
        "backdoor", view.dataset.data[rows], np.asarray(view.dataset.targets)[rows], mask
    )

    view, _, _ = get_cifar10_datasets(seed=seed, augment=False)
    chosen = [row_of[i] for i in flip_labels(view, LabelFlipConfig(), seed)]
    mask = none.copy()
    mask[chosen] = True
    out["label_flip"] = Scenario("label_flip", raw, np.asarray(view.dataset.targets)[rows], mask)

    out["slow_drip_1"] = Scenario("slow_drip_1", raw, labels, none, drip=1)
    out["slow_drip_5"] = Scenario("slow_drip_5", raw, labels, none, drip=5)

    gm_raw, mask = raw.copy(), none.copy()
    gm_raw[data.gradient_matching], mask[data.gradient_matching] = data.gm_pixels, True
    out["gradient_matching"] = Scenario("gradient_matching", gm_raw, labels, mask)

    brighter = apply_shift(raw, Shift("brightness", 1.3))
    out["benign_brighter"] = Scenario("benign_brighter", brighter, labels, none)
    return out


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    targets = np.asarray(data.full.targets)
    _, _, test_set = get_cifar10_datasets(seed=args.seed, augment=False)

    # The guard: same detectors, calibration and classifier as experiment 14.
    score = FoldScorer(args.vae_prefix, args.reference_prefix, args.folds, args.patch, device)
    trusted_rows = np.concatenate([data.fit, data.held_out])
    trusted = score(data.full.data[trusted_rows], np.arange(len(trusted_rows)) % args.folds)
    calibration = calibrate(
        trusted.picture,
        trusted.log_probs,
        torch.as_tensor(targets[trusted_rows]),
        args.window * args.batch_size,
        args.image_rate,
        torch.Generator().manual_seed(42),  # experiment 14's classifier seed
    )
    classifier = load_weights(AttackClassifier(len(FEATURES)), args.classifier, device).cpu()

    curve_measure = Measure(test_set, device, args.measure_images, args.seed)
    final_measure = Measure(test_set, device, None, args.seed)
    cases = scenarios(data, args.seed)
    names = args.scenarios or list(cases)

    mlflow.set_experiment("mlecho-response")
    with mlflow.start_run(run_name=f"episodes-{args.policy}-seed{args.episode_seed}"):
        mlflow.log_params(vars(args) | {"device": str(device), "rules": str(RulePolicy())})
        for name in names:
            seed_everything(args.episode_seed)
            model = load_weights(SmallCNN(width=64), args.burn_in, device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
            monitor = Monitor(
                score.vaes,
                score.references,
                calibration,
                classifier,
                args.window,
                args.patch,
                args.episode_seed,
            )
            log = run_episode(
                model,
                optimizer,
                cases["clean"].raw,
                cases["clean"].labels,
                cases[name],
                monitor,
                POLICIES[args.policy],
                args.batches,
                args.onset,
                args.batch_size,
                curve_measure,
                args.measure_every,
                device,
                args.episode_seed,
            )
            final = final_measure(model)
            first = log.actions[0] if log.actions else (None, "none")
            summary = final | {
                "poison_trained": log.poison_trained,
                "poison_dropped": log.poison_dropped,
                "clean_dropped": log.clean_dropped,
                "actions": len(log.actions),
                "first_action_batch": first[0] if first[0] is not None else -1,
            }
            mlflow.log_metrics({f"{k}_{name}": v for k, v in summary.items()})
            mlflow.log_dict(asdict(log), f"episode_{name}.json")
            mlflow.set_tag(f"first_action_{name}", first[1])
            print(
                f"{name:18s} clean {final['clean_acc']:.1%}  ASR {final['asr']:5.1%}  "
                f"cat {final['cat_acc']:5.1%} (as dog {final['cat_as_dog']:5.1%})  "
                f"target P(dog) {final['target_dog']:5.1%} | first action {first[1]} at "
                f"{first[0]} | poison learned {log.poison_trained}, dropped "
                f"{log.poison_dropped} poison / {log.clean_dropped} clean"
            )


if __name__ == "__main__":
    main()
