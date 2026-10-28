"""Experiment 14: which attack is it? The attack classifier (steps 4.2-4.4, RQ2).

Each 10-batch window of the training stream is turned into 10 numbers
(`mlecho.classification.features`), and a small softmax regression names the
type: clean, backdoor, label-flip, accumulative (slow-drip) or harmless shift.

Learns ONLY from made-up attacks on the 5,000 trusted images
(`mlecho.classification.synthetic`): random trigger colours, sizes, places and
target classes; random flipped class pairs (never the four test pairs);
random brightness / contrast / blur changes. Scores are out-of-fold (5 VAEs
from experiment 13, 5 reference models from experiment 09).

Tested on the 45,000 training images, with:
1. the real step-2 attacks (yellow square -> plane, cat -> dog, slow-drip) and
   the harmless changes of experiment 11;
2. variants it never saw: a checkerboard trigger, other poison rates, three
   other flip pairs, slow-drip with the checkerboard;
3. gradient matching, which the detector can't see (experiments 08, 09).

Run:   python experiments/14_attack_classifier.py
View:  mlflow ui --port 5050   (then open http://127.0.0.1:5050)
"""

import argparse
from types import SimpleNamespace

import mlflow
import numpy as np
import torch
from common import (
    SHIFTS,
    FoldScorer,
    add_detection_args,
    detection_data,
    shifted,
)
from sklearn.ensemble import RandomForestClassifier  # comes with mlflow; comparison only

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger, poison_train_set
from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.bench.datasets import get_cifar10_datasets
from mlecho.classification.classifier import ATTACK_TYPES, AttackClassifier, fit_classifier
from mlecho.classification.features import FEATURES, calibrate, stream_rows, window_features
from mlecho.classification.synthetic import (
    apply_shift,
    random_backdoor,
    random_flip,
    random_shift,
)
from mlecho.core import get_device, save_weights, seed_everything

TEST_FLIPS = ((3, 5), (4, 7), (9, 1), (0, 2))  # cat->dog (the attack), then experiment 09's pairs
CHECKER = BackdoorConfig(  # unseen trigger: training only has solid squares
    target_class=3, patch_size=3, patch_color=(255, 0, 0), position=(0, 0), pattern="checker"
)
IN_FAMILY_SHIFTS = ("brighter", "darker", "contrast", "blur")  # kinds trained on


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_detection_args(parser)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--patch", type=int, default=4, help="chosen in experiment 07")
    parser.add_argument("--vae-prefix", default="data/checkpoints/vae-c16-z32-seed42-fold")
    parser.add_argument("--reference-prefix", default="data/checkpoints/reference-w32-seed42-fold")
    parser.add_argument("--configs", type=int, default=40, help="made-up triggers / shifts")
    parser.add_argument("--train-windows", type=int, default=600, help="per type")
    parser.add_argument("--check-windows", type=int, default=200, help="per type, new settings")
    parser.add_argument("--test-windows", type=int, default=200, help="per test case")
    parser.add_argument("--shift-images", type=int, default=10000, help="training images shifted")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument(
        "--model",
        choices=["softmax", "forest"],
        default="softmax",
        help="softmax: MLEcho's classifier; forest: a non-linear comparison (scikit-learn)",
    )
    parser.add_argument(
        "--classifier-seed",
        type=int,
        default=42,
        help="seed of the made-up attacks and windows (--seed keeps the data split and detector)",
    )
    parser.add_argument(
        "--save",
        default="data/checkpoints/attack-classifier-seed{seed}.pt",
        help="{seed}: classifier seed",
    )
    return parser.parse_args()


def features(evidence, labels, rows, batches, calibration) -> torch.Tensor:
    """Window features of the images at `rows` (split into `batches` batches)."""
    return window_features(
        evidence.picture[rows].view(batches, -1),
        evidence.log_probs[rows].view(batches, -1, evidence.log_probs.shape[1]),
        labels[rows].view(batches, -1),
        calibration,
    )


def overlay(base, poison, mask, rows) -> SimpleNamespace:
    """Evidence of the window `rows`, with the images where `mask` is True replaced
    by their poisoned version (`poison`, indexed like `base`)."""
    picture, log_probs = base.picture[rows].clone(), base.log_probs[rows].clone()
    picture[mask], log_probs[mask] = poison.picture[rows[mask]], poison.log_probs[rows[mask]]
    return SimpleNamespace(picture=picture, log_probs=log_probs)


def made_up_window(kind, trusted, labels, settings, rng, generator, args, calibration):
    """One made-up window of type `kind` from the trusted images (see module docstring)."""
    batches, size = args.window, args.window * args.batch_size
    rows = stream_rows(len(labels), size, generator)
    lab = labels[rows].clone()
    evidence = SimpleNamespace(picture=trusted.picture[rows], log_probs=trusted.log_probs[rows])
    if kind == "backdoor":  # a share of the dataset carries the trigger
        config, stamped = settings.triggers[rng.integers(len(settings.triggers))]
        mask = (torch.rand(size, generator=generator) < rng.uniform(0.02, 0.10)) & (
            lab != config.target_class
        )
        evidence, lab[mask] = overlay(trusted, stamped, mask, rows), config.target_class
    elif kind == "accumulative":  # a fixed few triggered images in every batch
        config, stamped = settings.triggers[rng.integers(len(settings.triggers))]
        mask = torch.zeros(batches, args.batch_size, dtype=torch.bool)
        mask[:, : rng.integers(1, 7)] = True
        mask = mask.flatten()
        others = torch.as_tensor(np.flatnonzero(labels.numpy() != config.target_class))
        rows = rows.clone()
        rows[mask] = others[torch.randint(len(others), (int(mask.sum()),), generator=generator)]
        evidence, lab = overlay(trusted, stamped, mask, rows), labels[rows].clone()
        lab[mask] = config.target_class
    elif kind == "label_flip":  # some images of one class get another class's label
        source, target = random_flip(rng, 10, settings.excluded)
        share = rng.uniform(0.01, 0.05) / (labels == source).float().mean().item()
        lab[(lab == source) & (torch.rand(size, generator=generator) < share)] = target
    elif kind == "benign_shift":  # part or all of the data changed in a harmless way
        _, changed = settings.shifts[rng.integers(len(settings.shifts))]
        mask = torch.rand(size, generator=generator) < rng.uniform(0.3, 1.0)
        evidence = overlay(trusted, changed, mask, rows)
    whole = torch.arange(size)
    return features(evidence, lab, whole, batches, calibration)


def made_up_set(trusted, labels, settings, windows, rng, generator, args, calibration):
    """`windows` made-up windows of every type -> (features, type numbers)."""
    x, y = [], []
    for t, kind in enumerate(ATTACK_TYPES):
        for _ in range(windows):
            x.append(
                made_up_window(kind, trusted, labels, settings, rng, generator, args, calibration)
            )
            y.append(t)
    return torch.stack(x), torch.tensor(y)


def made_up_settings(score, raw, which, count, rng, excluded) -> SimpleNamespace:
    """`count` random triggers and harmless shifts, with the trusted images' evidence."""
    triggers = []
    for _ in range(count):
        config = random_backdoor(rng, raw.shape[1], 10)
        triggers.append((config, score(add_trigger(raw, config), which)))
    shifts = []
    for _ in range(count):
        shift = random_shift(rng)
        shifts.append((shift, score(apply_shift(raw, shift), which)))
    return SimpleNamespace(triggers=triggers, shifts=shifts, excluded=excluded)


def report(name, predicted, expected) -> dict[str, float]:
    """Share of windows given each type; `correct` if an expected type is given."""
    shares = {t: (predicted == i).float().mean().item() for i, t in enumerate(ATTACK_TYPES)}
    result = {f"as_{t}": v for t, v in shares.items()}
    result["called_attack"] = 1 - shares["clean"] - shares["benign_shift"]
    if expected:
        result["correct"] = sum(shares[t] for t in expected)
    line = "  ".join(f"{v:5.0%}" for v in shares.values())
    right = f"{result['correct']:6.1%}" if expected else "   n/a"
    print(f"{name:30s} {line}   {right}   {'/'.join(expected) or '-'}")
    return result


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    generator = torch.Generator().manual_seed(args.classifier_seed)
    rng = np.random.default_rng(args.classifier_seed)
    data = detection_data(args.seed, args.fit_size, args.gm_cache)
    targets = np.asarray(data.full.targets)
    score = FoldScorer(args.vae_prefix, args.reference_prefix, args.folds, args.patch, device)
    window_size = args.window * args.batch_size

    # Trusted images, each scored by the fold models that never saw it.
    trusted_rows = np.concatenate([data.fit, data.held_out])
    trusted_raw = data.full.data[trusted_rows]
    fold_of_trusted = np.arange(len(trusted_rows)) % args.folds
    trusted = score(trusted_raw, fold_of_trusted)
    trusted_labels = torch.as_tensor(targets[trusted_rows])
    calibration = calibrate(
        trusted.picture, trusted.log_probs, trusted_labels, window_size, args.image_rate, generator
    )

    excluded = {pair for s, t in TEST_FLIPS for pair in ((s, t), (t, s))}
    train_settings = made_up_settings(
        score, trusted_raw, fold_of_trusted, args.configs, rng, excluded
    )
    check_settings = made_up_settings(
        score, trusted_raw, fold_of_trusted, args.configs // 4, rng, excluded
    )
    print("made-up attacks scored")

    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=f"attack-classifier-{args.model}-seed{args.classifier_seed}"):
        mlflow.log_params(vars(args) | {"device": str(device)})
        # How many triggered trusted images the picture check flags, per made-up trigger.
        caught = []
        for i, (config, stamped) in enumerate(train_settings.triggers):
            caught.append((stamped.picture > calibration.picture_threshold).float().mean().item())
            mlflow.log_metric("made_up_trigger_caught", caught[-1], step=i)
            print(
                f"  trigger {i:2d}: {config.patch_size}px {config.patch_color}"
                f"  caught {caught[-1]:4.0%}"
            )
        print(f"made-up triggers caught: median {np.median(caught):.0%}")
        x, y = made_up_set(
            trusted,
            trusted_labels,
            train_settings,
            args.train_windows,
            rng,
            generator,
            args,
            calibration,
        )
        if args.model == "softmax":
            model = AttackClassifier(len(FEATURES))
            mlflow.log_metric("train_loss", fit_classifier(model, x, y, l2=args.l2))
            save_weights(model, args.save.format(seed=args.classifier_seed))
            probabilities = model.probabilities
        else:
            forest = RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, random_state=args.classifier_seed
            ).fit(x.numpy(), y.numpy())

            def probabilities(features: torch.Tensor) -> torch.Tensor:
                return torch.from_numpy(forest.predict_proba(features.numpy())).float()

        train_accuracy = (probabilities(x).argmax(dim=1) == y).float().mean().item()

        # Development check: new made-up windows with new triggers and shifts.
        cx, cy = made_up_set(
            trusted,
            trusted_labels,
            check_settings,
            args.check_windows,
            rng,
            generator,
            args,
            calibration,
        )
        check_pred = probabilities(cx).argmax(dim=1)
        check_accuracy = (check_pred == cy).float().mean().item()
        mlflow.log_metrics({"train_accuracy": train_accuracy, "check_accuracy": check_accuracy})
        print(
            f"made-up windows: train accuracy {train_accuracy:.1%},"
            f" new settings {check_accuracy:.1%}"
        )
        for t, kind in enumerate(ATTACK_TYPES):
            share = (check_pred[cy == t] == t).float().mean().item()
            mlflow.log_metric(f"check_accuracy_{kind}", share)
            print(f"  {kind:14s} {share:6.1%}")

        if args.model == "softmax":  # which features push towards which type
            weights = model.linear.weight.detach()  # on standardised features
            mlflow.log_dict(
                {
                    kind: dict(zip(FEATURES, weights[t].tolist(), strict=True))
                    for t, kind in enumerate(ATTACK_TYPES)
                },
                "weights.json",
            )
            print("\nweights (rows: type; columns: " + ", ".join(FEATURES) + ")")
            for t, kind in enumerate(ATTACK_TYPES):
                print(f"  {kind:14s} " + " ".join(f"{w:5.1f}" for w in weights[t].tolist()))

        # The real training stream: every training image scored by one random fold model.
        which = np.random.default_rng(args.seed).integers(0, args.folds, len(data.train))
        train_raw = data.full.data[data.train]
        clean = score(train_raw, which)
        true = torch.as_tensor(targets[data.train])
        yellow = score(add_trigger(train_raw, BackdoorConfig()), which)
        checker = score(add_trigger(train_raw, CHECKER), which)
        gm = SimpleNamespace(picture=clean.picture.clone(), log_probs=clean.log_probs.clone())
        gm_scored = score(data.gm_pixels, which[data.gradient_matching])
        gm.picture[data.gradient_matching] = gm_scored.picture
        gm.log_probs[data.gradient_matching] = gm_scored.log_probs
        row_of = {position: k for k, position in enumerate(data.train)}

        def population(trigger_scores, config, rate):
            """Dataset poisoned like `poison_train_set`: `rate` of it triggered."""
            view, _, _ = get_cifar10_datasets(seed=args.seed, augment=False)
            plan = BackdoorConfig(target_class=config.target_class, poison_rate=rate)
            rows = [row_of[i] for i in poison_train_set(view, plan, args.seed)]
            mask = torch.zeros(len(true), dtype=torch.bool)
            mask[rows] = True
            evidence = overlay(clean, trigger_scores, mask, torch.arange(len(true)))
            labels = true.clone()
            labels[mask] = config.target_class
            return evidence, labels

        def flipped(source, target, rate):
            view, _, _ = get_cifar10_datasets(seed=args.seed, augment=False)
            config = LabelFlipConfig(source, target, rate)
            labels = true.clone()
            labels[[row_of[i] for i in flip_labels(view, config, args.seed)]] = target
            return clean, labels

        def stream(evidence, labels, drip=None, per_batch=0, config=None):
            """Test windows from a population; slow-drip adds `per_batch` triggered images."""
            out = []
            if per_batch:  # slow-drip picks images not already in the target class
                others = torch.as_tensor(np.flatnonzero(true.numpy() != config.target_class))
            for _ in range(args.test_windows):
                rows = torch.randint(len(labels), (window_size,), generator=generator)
                lab = labels[rows].clone()
                window = SimpleNamespace(
                    picture=evidence.picture[rows], log_probs=evidence.log_probs[rows]
                )
                if per_batch:
                    mask = torch.zeros(args.window, args.batch_size, dtype=torch.bool)
                    mask[:, :per_batch] = True
                    mask = mask.flatten()
                    rows[mask] = others[
                        torch.randint(len(others), (int(mask.sum()),), generator=generator)
                    ]
                    window = overlay(evidence, drip, mask, rows)
                    lab = labels[rows].clone()
                    lab[mask] = config.target_class
                out.append(
                    features(window, lab, torch.arange(window_size), args.window, calibration)
                )
            return torch.stack(out)

        yellow_config = BackdoorConfig()
        cases = {  # name -> (test windows, expected types); "official" = the step-2 attacks
            "clean": (stream(clean, true), ("clean",)),
            "backdoor (official)": (
                stream(*population(yellow, yellow_config, 0.05)),
                ("backdoor",),
            ),
            "label_flip cat->dog (official)": (stream(*flipped(3, 5, 0.05)), ("label_flip",)),
        }
        for k in (1, 2, 5):
            cases[f"accumulative {k}/batch (official)"] = (
                stream(clean, true, yellow, k, yellow_config),
                ("accumulative",),
            )
        # Unseen variants.
        cases["backdoor checker 5%"] = (stream(*population(checker, CHECKER, 0.05)), ("backdoor",))
        for rate in (0.02, 0.10):
            cases[f"backdoor yellow {rate:.0%}"] = (
                stream(*population(yellow, yellow_config, rate)),
                ("backdoor",),
            )
        cases["accumulative checker 2/batch"] = (
            stream(clean, true, checker, 2, CHECKER),
            ("accumulative",),
        )
        cases["label_flip cat->dog 2%"] = (stream(*flipped(3, 5, 0.02)), ("label_flip",))
        names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        for source, target in TEST_FLIPS[1:]:
            cases[f"label_flip {names[source]}->{names[target]}"] = (
                stream(*flipped(source, target, 0.05)),
                ("label_flip",),
            )
        cases["gradient_matching"] = (stream(gm, true), ())

        # Harmless changes (experiment 11) on a sample of training images.
        sample = np.random.default_rng(args.seed).choice(
            len(true), args.shift_images, replace=False
        )
        sample_labels = true[sample]
        for name, change in SHIFTS.items():
            if name == "none":
                continue
            evidence = score(shifted(train_raw[sample], change), which[sample])
            family = (
                "in-family" if any(word in name for word in IN_FAMILY_SHIFTS) else "unseen kind"
            )
            cases[f"shift {name} ({family})"] = (
                stream(evidence, sample_labels),
                ("benign_shift", "clean"),
            )

        print(
            f"\n{'test case':30s} "
            + "  ".join(f"{t[:5]:>5s}" for t in ATTACK_TYPES)
            + "   correct   expected"
        )
        summary = {}
        for name, (x_test, expected) in cases.items():
            chances = probabilities(x_test)
            result = report(name, chances.argmax(dim=1), expected)
            result["confidence"] = chances.max(dim=1).values.mean().item()
            key = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
            mlflow.log_metrics({f"{k}_{key}": v for k, v in result.items()})
            summary[name] = result
        mlflow.log_dict(summary, "test_results.json")


if __name__ == "__main__":
    main()
