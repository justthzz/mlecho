"""Step 4 figures: redraw every attack-classifier chart with one command.

Reads the official runs of experiment 14 listed in docs/RESULTS.md from
MLflow. Trains nothing. Colour-blind safe (viridis and purple-orange maps,
text on every cell).

Run:   python experiments/figures_step04.py                (writes local-docs/figures/)
       python experiments/figures_step04.py --out some/folder
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, GREY, ORANGE, Results

from mlecho.classification.classifier import ATTACK_TYPES
from mlecho.classification.features import FEATURES
from mlecho.classification.synthetic import random_backdoor

RUNS = {  # official runs (docs/RESULTS.md, experiment 14)
    "softmax": {42: "e35fb511", 1: "0bdcd525", 2: "7ae2e890"},
    "forest": "dcf6afa4",
}
OFFICIAL = [  # the step-2 attacks and their expected type
    "clean",
    "backdoor (official)",
    "label_flip cat->dog (official)",
    "accumulative 1/batch (official)",
    "accumulative 2/batch (official)",
    "accumulative 5/batch (official)",
]
SHORT = {
    "clean": "clean",
    "backdoor": "backdoor",
    "label_flip": "label-flip",
    "accumulative": "accumulative",
    "benign_shift": "harmless shift",
}


def text_heatmap(ax, values, rows, columns, cmap, vmin, vmax, fmt, label) -> None:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    middle = (vmin + vmax) / 2
    for r in range(len(rows)):
        for c in range(len(columns)):
            v = values[r, c]
            dark = v > middle if cmap == "viridis" else abs(v - middle) < (vmax - vmin) / 4
            ax.text(
                c,
                r,
                fmt(v),
                ha="center",
                va="center",
                fontsize=7,
                color="black" if dark else "white",
            )
    plt.colorbar(image, ax=ax, label=label)


def confusion(results) -> plt.Figure:
    tests = results.json(RUNS["softmax"][42], "test_results.json")
    names = list(tests)
    values = np.array([[tests[n][f"as_{t}"] * 100 for t in ATTACK_TYPES] for n in names])
    fig, ax = plt.subplots(figsize=(8, 9))
    text_heatmap(
        ax,
        values,
        names,
        [SHORT[t] for t in ATTACK_TYPES],
        "viridis",
        0,
        100,
        lambda v: f"{v:.0f}%",
        "windows given this type (%)",
    )
    ax.set_xlabel("type the classifier gives")
    ax.set_title(
        "Which attack is it? 200 test windows per row (classifier never saw these)", fontsize=9
    )
    fig.tight_layout()
    return fig


def weights(results) -> plt.Figure:
    w = results.json(RUNS["softmax"][42], "weights.json")
    values = np.array([[w[t][f] for f in FEATURES] for t in ATTACK_TYPES])
    limit = np.abs(values).max()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    text_heatmap(
        ax,
        values,
        [SHORT[t] for t in ATTACK_TYPES],
        list(FEATURES),
        "PuOr",
        -limit,
        limit,
        lambda v: f"{v:+.1f}",
        "weight (+ points towards the type)",
    )
    ax.set_title("What the classifier looks at (weights on standardised features)", fontsize=9)
    fig.tight_layout()
    return fig


def trigger_visibility(results) -> plt.Figure:
    """Picture-check catch rate of each made-up trigger, by size and colour."""
    _, caught = results.history(RUNS["softmax"][42], "made_up_trigger_caught")
    rng = np.random.default_rng(42)  # replays the triggers experiment 14 drew first
    configs = [random_backdoor(rng, 32, 10) for _ in caught]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(caught))
    for config, value, j in zip(configs, caught, jitter, strict=True):
        ax.scatter(
            config.patch_size + j,
            value * 100,
            s=70,
            color=np.array(config.patch_color) / 255,
            edgecolors="black",
            linewidths=0.6,
        )
    ax.scatter(
        [4],
        [results.metric("78942c03", "image_recall_backdoor") * 100],
        marker="*",
        s=250,
        color=(1, 1, 0),
        edgecolors="black",
        label="the real attack (yellow 4x4)",
    )
    ax.set_xlabel("trigger size (pixels per side)")
    ax.set_ylabel("triggered images flagged (%)")
    ax.set_title(
        "The picture check sees vivid triggers of 4 pixels or more\n"
        "(each dot = one made-up trigger, drawn in its own colour)",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def stability(results) -> plt.Figure:
    """Official cases: correct share for 3 classifier seeds (mean and range) and the forest."""
    softmax = np.array(
        [
            [results.json(run, "test_results.json")[n]["correct"] * 100 for n in OFFICIAL]
            for run in RUNS["softmax"].values()
        ]
    )
    forest = [
        results.json(RUNS["forest"], "test_results.json")[n]["correct"] * 100 for n in OFFICIAL
    ]
    x = np.arange(len(OFFICIAL))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    mean = softmax.mean(axis=0)
    error = [mean - softmax.min(axis=0), softmax.max(axis=0) - mean]
    ax.bar(
        x - 0.2,
        mean,
        0.4,
        yerr=error,
        capsize=3,
        color=BLUE,
        label="softmax (MLEcho), 3 seeds: mean and range",
    )
    ax.bar(x + 0.2, forest, 0.4, color=ORANGE, label="random forest (comparison)")
    ax.axhline(100, color=GREY, linewidth=0.5)
    labels = [n.replace(" (official)", "").replace("accumulative", "slow-drip") for n in OFFICIAL]
    ax.set_xticks(x, labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("windows named correctly (%)")
    ax.set_ylim(0, 125)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title("The step-2 attacks: right type named, per test case", fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="local-docs/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")

    results = Results()
    figures = {
        "step04-1-confusion": lambda: confusion(results),
        "step04-2-weights": lambda: weights(results),
        "step04-3-trigger-visibility": lambda: trigger_visibility(results),
        "step04-4-stability": lambda: stability(results),
    }
    for name, draw in figures.items():
        fig = draw()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"saved {out / name}.png")


if __name__ == "__main__":
    main()
