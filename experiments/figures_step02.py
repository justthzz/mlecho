"""Step 2 figures: redraw every attack picture and chart with one command.

Reads the official runs listed in docs/RESULTS.md from MLflow (metrics and
saved models) plus the CIFAR-10 images. Trains nothing; takes about a minute.
Colours are colour-blind safe (Okabe-Ito), and every bar and line also has a
text label or marker, so nothing depends on colour alone.

Run:   python experiments/figures_step02.py                (writes local-docs/figures/)
       python experiments/figures_step02.py --out some/folder
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from common import (
    ATTACK_COLOUR,
    CLEAN_COLOUR,
    GREEN,
    GREY,
    ORANGE,
    VERMILLION,
    Results,
    label_bars,
    show_row,
)
from torch.utils.data import DataLoader

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.attacks.labelflip import LabelFlipConfig, flip_labels
from mlecho.bench.datasets import CIFAR10_CLASSES, cifar10_transform, get_cifar10_datasets
from mlecho.bench.training import predict

# Official runs (docs/RESULTS.md)
RUNS = {
    "backdoor_clean": "3f507100",
    "backdoor_attack": "80c612ac",
    "flip_clean": "0e827c6e",
    "flip_attack": "a67415a6",
    "drip_control": "9c6fa516",
    "drip_1": "be46a900",
    "drip_5": "ed5b5913",
    "gm_control": "004e5cbc",
}
GM_TARGETS = {8183: "dbefe861", 4975: "65b96ff7", 2231: "954b9141", 669: "50a8de63"}


def predict_each(model: torch.nn.Module, raw: np.ndarray) -> list[int]:
    """Predicted class for each raw image, one at a time (older saved models need that)."""
    plain = cifar10_transform(augment=False)
    with torch.no_grad():
        return [int(model(plain(image).unsqueeze(0)).argmax()) for image in raw]


# ---------- attack 1: backdoor ----------


def backdoor_examples(full, train_indices) -> plt.Figure:
    config = BackdoorConfig()
    picks = [i for i in train_indices if full.targets[i] != config.target_class][:6]
    raw = full.data[picks]
    fig, axes = plt.subplots(2, 6, figsize=(9, 3.4), layout="constrained")
    show_row(axes[0], raw, [CIFAR10_CLASSES[full.targets[i]] for i in picks], "original")
    show_row(axes[1], add_trigger(raw, config), ["label → plane"] * 6, "poisoned")
    fig.suptitle("Backdoor: yellow 4×4 square + label changed to 'plane' (5% of training images)")
    return fig


def backdoor_results(results, test_set) -> plt.Figure:
    config = BackdoorConfig()
    fig = plt.figure(figsize=(9, 5.6), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.1, 1])

    ax = top.subplots()
    names = ["clean test accuracy", "attack success rate\n(images with the square called plane)"]
    clean = [
        results.metric(RUNS["backdoor_clean"], k) * 100 for k in ("test_acc", "attack_success_rate")
    ]
    attack = [
        results.metric(RUNS["backdoor_attack"], k) * 100
        for k in ("test_acc", "attack_success_rate")
    ]
    x = np.arange(2)
    label_bars(
        ax,
        ax.bar(x - 0.2, clean, 0.4, color=CLEAN_COLOUR, label="model trained on clean data"),
        "{:.2f}%",
    )
    label_bars(
        ax,
        ax.bar(
            x + 0.2,
            attack,
            0.4,
            color=ATTACK_COLOUR,
            hatch="//",
            label="model trained on poisoned data",
        ),
        "{:.2f}%",
    )
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("%")
    ax.legend(fontsize=8, loc="center")
    top.suptitle("Backdoor: normal on clean images, fooled whenever the square is present")

    keep = [i for i in range(len(test_set.targets)) if test_set.targets[i] != config.target_class][
        :6
    ]
    raw = add_trigger(test_set.data[keep], config)
    clean_preds = predict_each(results.model(RUNS["backdoor_clean"]), raw)
    attack_preds = predict_each(results.model(RUNS["backdoor_attack"]), raw)
    titles = [
        f"true: {CIFAR10_CLASSES[test_set.targets[i]]}\nclean model: {CIFAR10_CLASSES[c]}"
        f"\npoisoned model: {CIFAR10_CLASSES[a]} {'✗' if a == 0 else '✓'}"
        for i, c, a in zip(keep, clean_preds, attack_preds, strict=True)
    ]
    show_row(bottom.subplots(1, 6), raw, titles)
    bottom.suptitle("Test images with the square added (✗ = fooled into 'plane')", fontsize=10)
    return fig


# ---------- attack 2: label-flip ----------


def labelflip_examples(full, flipped) -> plt.Figure:
    fig, axes = plt.subplots(1, 6, figsize=(9, 2.2), layout="constrained")
    show_row(axes, full.data[flipped[:6]], ["picture: cat\nlabel: dog"] * 6)
    fig.suptitle(
        "Label-flip: cat pictures untouched, label changed to 'dog' (2,250 = half the cats)"
    )
    return fig


def labelflip_results(results, test_set) -> plt.Figure:
    loader = DataLoader(test_set, batch_size=500)
    per_class = {}
    for key in ("flip_clean", "flip_attack"):
        preds, labels = predict(results.model(RUNS[key]), loader, torch.device("cpu"))
        per_class[key] = [
            ((preds == labels)[labels == c].float().mean() * 100).item() for c in range(10)
        ]
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    x = np.arange(10)
    ax.bar(x - 0.2, per_class["flip_clean"], 0.4, color=CLEAN_COLOUR, label="clean training data")
    ax.bar(
        x + 0.2,
        per_class["flip_attack"],
        0.4,
        color=ATTACK_COLOUR,
        hatch="//",
        label="label-flipped",
    )
    for c in (3, 5):
        for offset, key in ((-0.2, "flip_clean"), (0.2, "flip_attack")):
            ax.annotate(
                f"{per_class[key][c]:.0f}",
                (c + offset, per_class[key][c]),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, CIFAR10_CLASSES)
    ax.set_ylabel("test accuracy per class (%)")
    ax.set_ylim(0, 118)
    ax.legend(fontsize=8, loc="upper center", ncols=2)
    ax.set_title(
        "Label-flip: only cat accuracy collapses (69% → 19%); the other classes barely move"
    )
    return fig


# ---------- attack 3: accumulative ----------


def accumulative_sweeps() -> plt.Figure:
    runs = mlflow.search_runs(experiment_names=["mlecho-tuning"])
    runs = runs[runs["params.trade_off"].notna() & runs["metrics.val_single_step_drop"].notna()]

    def sweep(row) -> int:
        if row.get("params.norm") == "group":
            return 4
        if row["params.per_layer"] == "True":
            return 3
        return 1 if float(row["params.lr"]) == 0.001 else 2

    runs = runs.assign(sweep=runs.apply(sweep, axis=1))
    # the 200-step control was the burn-in stability check, not part of sweep 2
    runs = runs[~((runs["sweep"] == 2) & (runs["params.steps"] == "200"))]
    control = runs["params.epsilon"].astype(float) == 0
    secret = runs["metrics.accumulative_steps_done"] == runs["params.steps"].astype(float)
    groups = [
        (control, dict(marker="D", color=GREY, label="control (no attack)")),
        (~control & secret, dict(marker="o", color=GREEN, label="attack stayed secret")),
        (
            ~control & ~secret,
            dict(marker="x", color=VERMILLION, label="attack stopped early (not secret)"),
        ),
    ]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    rng = np.random.default_rng(0)  # small sideways spread so points don't hide each other
    for rows, style in groups:
        subset = runs[rows]
        jitter = rng.uniform(-0.15, 0.15, len(subset))
        ax.scatter(
            subset["sweep"] + jitter, subset["metrics.val_single_step_drop"] * 100, s=30, **style
        )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(
        [1, 2, 3, 4],
        ["1\nAdam burn-in", "2\nSGD burn-in", "3\nper-layer", "4\nGroupNorm,\nauthors' settings"],
    )
    ax.set_xlabel("sweep")
    ax.set_ylabel("accuracy drop after one trigger step\n(validation, points)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title(
        "Accumulative (Pang et al.): no setting was both secret and followed by a big drop\n"
        "(paper, ResNet-18: 82% → 28%, a 54-point drop)",
        fontsize=10,
    )
    return fig


def slowdrip_curves(results) -> plt.Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for key, style in {
        "drip_control": dict(color=GREY, marker="s", label="no poison"),
        "drip_1": dict(color=ORANGE, marker="o", label="1 poisoned image per batch"),
        "drip_5": dict(color=VERMILLION, marker="^", label="5 poisoned images per batch"),
    }.items():
        steps, asr = results.history(RUNS[key], "attack_success_rate")
        left.plot(steps, np.array(asr) * 100, markersize=4, **style)
        steps, acc = results.history(RUNS[key], "test_acc")
        right.plot(steps, np.array(acc) * 100, markersize=4, **style)
    left.set_title("Attack success rate builds up")
    left.set_ylabel("%")
    right.set_title("Clean accuracy stays normal")
    right.set_ylim(70, 90)
    for ax in (left, right):
        ax.set_xlabel("online step (batches of 100)")
        ax.legend(fontsize=8)
    fig.suptitle("Slow-drip backdoor: a little poison in every batch")
    return fig


# ---------- attack 4: gradient matching ----------


def gradient_matching_examples(results, test_set) -> plt.Figure:
    strip = results.image(GM_TARGETS[8183], "poisons_before_after_difference.png")
    fig = plt.figure(figsize=(10, 3.4), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1, 5])
    ax = left.subplots()
    ax.imshow(test_set.data[8183])
    ax.set_title("target (test image 8183)\ntrue: cat; wanted: dog", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = right.subplots()
    ax.imshow(strip)
    ax.set_xticks([])
    third = strip.shape[0] / 3
    ax.set_yticks(
        [third * 0.5, third * 1.5, third * 2.5],
        ["original dog", "poisoned dog\n(label still 'dog')", "difference\n(×4, grey = 0)"],
        fontsize=8,
    )
    right.suptitle("8 of the 450 poisoned training dogs (largest change 16/255)", fontsize=10)
    fig.suptitle("Gradient matching (Witches' Brew): clean-label, invisible changes")
    return fig


def gradient_matching_results(results, test_set) -> plt.Figure:
    plain = cifar10_transform(augment=False)
    control = results.model(RUNS["gm_control"]).eval()

    def p_dog(model, index) -> float:
        with torch.no_grad():
            return model(plain(test_set.data[index]).unsqueeze(0)).softmax(dim=1)[0, 5].item() * 100

    before = [p_dog(control, i) for i in GM_TARGETS]
    after = [p_dog(results.model(run).eval(), i) for i, run in GM_TARGETS.items()]
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    x = np.arange(len(GM_TARGETS))
    label_bars(
        ax,
        ax.bar(x - 0.2, before, 0.4, color=CLEAN_COLOUR, label="trained without poison"),
        "{:.0f}%",
    )
    label_bars(
        ax,
        ax.bar(x + 0.2, after, 0.4, color=ATTACK_COLOUR, hatch="//", label="trained with poison"),
        "{:.0f}%",
    )
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8)
    ax.text(-0.45, 51, "50% line", fontsize=7)
    fooled = [results.metric(run, "target_fooled_rate") == 1 for run in GM_TARGETS.values()]
    called = ["dog ✗" if f else "cat" for f in fooled]  # the model's actual answer
    ax.set_xticks(x, [f"cat {i}\nnow called: {c}" for i, c in zip(GM_TARGETS, called, strict=True)])
    ax.set_ylabel("model's P(dog) for the target cat (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "Gradient matching, new model trained from scratch: 3 of 4 target cats called 'dog'",
        fontsize=10,
    )
    return fig


# ---------- all four ----------


def attacks_summary(results) -> plt.Figure:
    gm_clean = np.mean([results.metric(run, "test_acc") for run in GM_TARGETS.values()]) * 100
    gm_fooled = (
        np.mean([results.metric(run, "target_fooled_rate") for run in GM_TARGETS.values()]) * 100
    )
    rows = [
        (
            "backdoor",
            RUNS["backdoor_clean"],
            RUNS["backdoor_attack"],
            "attack_success_rate",
            "attack success",
        ),
        ("label-flip", RUNS["flip_clean"], RUNS["flip_attack"], "flip_rate", "cats called dog"),
        (
            "slow-drip (1/batch)",
            RUNS["drip_control"],
            RUNS["drip_1"],
            "attack_success_rate",
            "attack success, end",
        ),
    ]
    names = [r[0] for r in rows] + ["gradient matching"]
    clean_no = [results.metric(r[1], "test_acc") * 100 for r in rows] + [
        results.metric(RUNS["gm_control"], "test_acc") * 100
    ]
    clean_yes = [results.metric(r[2], "test_acc") * 100 for r in rows] + [gm_clean]
    effect_no = [results.metric(r[1], r[3]) * 100 for r in rows] + [0.0]
    effect_yes = [results.metric(r[2], r[3]) * 100 for r in rows] + [gm_fooled]
    effect_names = [r[4] for r in rows] + ["targets fooled"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    x = np.arange(4)
    for ax, no, yes, title in (
        (left, clean_no, clean_yes, "Clean test accuracy: stays normal (attack is hidden)"),
        (right, effect_no, effect_yes, "Attack effect: large only when poisoned"),
    ):
        label_bars(ax, ax.bar(x - 0.2, no, 0.4, color=CLEAN_COLOUR, label="no poison"))
        label_bars(
            ax,
            ax.bar(x + 0.2, yes, 0.4, color=ATTACK_COLOUR, hatch="//", label="poisoned"),
        )
        ax.set_ylim(0, 125)
        ax.set_ylabel("%")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper center", ncols=2)
    left.set_xticks(x, names, fontsize=8)
    right.set_xticks(
        x, [f"{n}\n({e})" for n, e in zip(names, effect_names, strict=True)], fontsize=7
    )
    fig.suptitle("All four attacks on CIFAR-10 / SmallCNN (accumulative: the slow-drip version)")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="local-docs/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")  # draw to files, no window

    results = Results()
    train_set, _, test_set = get_cifar10_datasets(augment=False)
    full = train_set.dataset
    flip_view, _, _ = get_cifar10_datasets(augment=False)  # a second copy to flip labels in
    flipped = flip_labels(flip_view, LabelFlipConfig(), seed=42)

    figures = {
        "step02-1-backdoor-examples": lambda: backdoor_examples(full, train_set.indices),
        "step02-2-backdoor-results": lambda: backdoor_results(results, test_set),
        "step02-3-labelflip-examples": lambda: labelflip_examples(full, flipped),
        "step02-4-labelflip-results": lambda: labelflip_results(results, test_set),
        "step02-5-accumulative-sweeps": accumulative_sweeps,
        "step02-6-slowdrip-curves": lambda: slowdrip_curves(results),
        "step02-7-gradient-matching-examples": lambda: gradient_matching_examples(
            results, test_set
        ),
        "step02-8-gradient-matching-results": lambda: gradient_matching_results(results, test_set),
        "step02-9-attacks-summary": lambda: attacks_summary(results),
    }
    for name, draw in figures.items():
        fig = draw()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"saved {out / name}.png")


if __name__ == "__main__":
    main()
