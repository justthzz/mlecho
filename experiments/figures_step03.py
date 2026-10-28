"""Step 3 figures: redraw every detection picture and chart with one command.

Reads the official runs listed in docs/RESULTS.md (experiments 07-12) from
MLflow, plus the saved CNN-VAE for the example pictures. Trains nothing.
Colour-blind safe (Okabe-Ito colours, viridis heatmaps, text on every cell).

Run:   python experiments/figures_step03.py                (writes local-docs/figures/)
       python experiments/figures_step03.py --out some/folder
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from common import (
    ATTACK_COLOUR,
    CLEAN_COLOUR,
    GREY,
    ORANGE,
    VERMILLION,
    Results,
    detection_data,
    label_bars,
    picture_scores,
)

from mlecho.attacks.backdoor import BackdoorConfig, add_trigger
from mlecho.bench.datasets import normalise
from mlecho.core import get_device, load_weights
from mlecho.detection.vae import ConvVAE

RUNS = {  # official runs (docs/RESULTS.md)
    "feasibility": "daf37fdc",
    "picture": "613f2371",
    "label": "02cb63dd",
    "overhead_small": "faef6394",
    "overhead_resnet": "b58ab3fa",
    "shift": "7bbc3de8",
    "baselines": "911ca15f",
}
VAE = "data/checkpoints/vae-c16-z32-seed42.pt"
GM_CACHE = "data/poisons/gradient-matching-cat8183.npz"


def heatmap(ax, values: np.ndarray, rows: list[str], columns: list[str]) -> None:
    """Percent heatmap with the number written in every cell; NaN cells are grey "not run"."""
    colours = plt.get_cmap("viridis").copy()
    colours.set_bad("lightgrey")
    masked = np.ma.masked_invalid(values * 100)
    image = ax.imshow(masked, cmap=colours, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(columns)), columns, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for r in range(len(rows)):
        for c in range(len(columns)):
            value = values[r, c] * 100
            if np.isnan(value):
                ax.text(c, r, "not run", ha="center", va="center", fontsize=8)
                continue
            ax.text(
                c,
                r,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if value > 60 else "white",
            )
    plt.colorbar(image, ax=ax, label="windows alarmed (%)")


def reconstruction_examples(data, device) -> plt.Figure:
    model = load_weights(ConvVAE(), VAE, device).eval()
    clean = data.full.data[data.train[:3]]
    squared = add_trigger(data.full.data[data.train[data.backdoor[:3]]], BackdoorConfig())
    gm = data.gm_pixels[:3]
    raw = np.concatenate([clean, squared, gm])
    images = normalise(raw).to(device)
    with torch.no_grad():
        rebuilt = model(images)[0]
    error = (rebuilt - images).pow(2).mean(dim=1).cpu()  # (N, H, W)
    shown = (
        rebuilt.cpu() * torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
        + torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    ).clamp(0, 1)
    kinds = ["clean"] * 3 + ["backdoor"] * 3 + ["gradient matching"] * 3
    fig, axes = plt.subplots(3, 9, figsize=(12, 4.6), layout="constrained")
    for k in range(9):
        axes[0, k].imshow(raw[k])
        axes[0, k].set_title(kinds[k], fontsize=8)
        axes[1, k].imshow(shown[k].permute(1, 2, 0).numpy())
        axes[2, k].imshow(error[k].numpy(), cmap="magma", vmin=0, vmax=float(error.max()))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for row, label in enumerate(["image", "VAE rebuilds", "error (bright = bad)"]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle("Picture check: normal content is rebuilt; the square stays a bright error spot")
    return fig


def auc_bars(results) -> plt.Figure:
    attacks = ["clean_held_out", "backdoor", "label_flip", "gradient_matching"]
    names = ["clean, unseen\n(sanity)", "backdoor", "label-flip", "gradient\nmatching"]
    mean = [results.metric(RUNS["feasibility"], f"auc_{a}_mean") for a in attacks]
    worst = [results.metric(RUNS["feasibility"], f"auc_{a}_worst4") for a in attacks]
    fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
    x = np.arange(len(attacks))
    label_bars(ax, ax.bar(x - 0.2, mean, 0.4, color=GREY, label="whole-image mean error"), "{:.2f}")
    label_bars(
        ax,
        ax.bar(
            x + 0.2,
            worst,
            0.4,
            color=ORANGE,
            hatch="//",
            label="worst 4×4 patch (chosen on made-up attacks)",
        ),
        "{:.2f}",
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="coin flip (0.5)")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("ROC AUC vs clean training images")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(
        "Picture check separation: looking at the worst patch finds the backdoor", fontsize=10
    )
    return fig


def score_histograms(results, data, device) -> plt.Figure:
    model = load_weights(ConvVAE(), VAE, device)
    rng = np.random.default_rng(0)
    clean = picture_scores(
        model, data.full.data[rng.choice(data.train, 5000, replace=False)], 4, device
    )
    squared = picture_scores(
        model, add_trigger(data.full.data[data.train[data.backdoor]], BackdoorConfig()), 4, device
    )
    gm = picture_scores(model, data.gm_pixels, 4, device)
    threshold = results.metric(RUNS["picture"], "image_threshold")
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    bins = np.linspace(0, float(np.quantile(squared, 0.99)), 70)
    ax.hist(clean, bins, density=True, color=CLEAN_COLOUR, alpha=0.6, label="clean training images")
    ax.hist(
        squared,
        bins,
        density=True,
        histtype="step",
        linewidth=2,
        color=VERMILLION,
        label="backdoor images",
    )
    ax.hist(
        gm,
        bins,
        density=True,
        histtype="step",
        linewidth=2,
        linestyle="--",
        color=ORANGE,
        label="gradient-matching images",
    )
    ax.axvline(threshold, color="black", linewidth=1)
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.9,
        "  threshold\n  (1% of trusted\n  images above)",
        fontsize=7,
        va="top",
    )
    ax.set_xlabel("picture-check score (worst 4×4 reconstruction error)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title(
        "Backdoor images score high; gradient-matching images look like clean ones", fontsize=10
    )
    return fig


def who_catches_what(results) -> plt.Figure:
    rows = [
        "clean (false alarms)",
        "backdoor",
        "label-flip cat→dog",
        "gradient matching",
        "slow-drip 1/batch",
        "slow-drip 5/batch",
    ]
    keys = ["clean", "backdoor", "label_flip", "gradient_matching", "slow_drip_1", "slow_drip_5"]
    pattern = ["clean", "backdoor", "label_flip_cat_to_dog", "gradient_matching", None, None]
    baseline = [
        "clean",
        "backdoor",
        "label_flip",
        "gradient_matching",
        "slow_drip_1",
        "slow_drip_5",
    ]
    columns = [
        "picture check\n(MLEcho)",
        "label check\n(MLEcho)",
        "label pattern\n(MLEcho)",
        "KS on pictures\n(baseline)",
        "PSI on labels,\ncalibrated (baseline)",
    ]
    values = np.full((len(rows), len(columns)), np.nan)
    for r, (k, p, b) in enumerate(zip(keys, pattern, baseline, strict=True)):
        values[r, 0] = results.metric(RUNS["picture"], f"window_alarm_{k}")
        values[r, 1] = results.metric(RUNS["label"], f"window_alarm_{k}")
        if p:
            values[r, 2] = results.metric(RUNS["label"], f"pattern_alarm_{p}_w10")
        values[r, 3] = results.metric(RUNS["baselines"], f"ks_alarm_{b}_w10")
        values[r, 4] = results.metric(RUNS["baselines"], f"psi_calibrated_alarm_{b}_w10")
    fig, ax = plt.subplots(figsize=(9, 4.2), layout="constrained")
    heatmap(ax, values, rows, columns)
    ax.set_title(
        "Who catches what: share of 10-batch windows (1,280 images) that alarm\n"
        "(first row should be ~1%; other rows high)",
        fontsize=10,
    )
    return fig


def false_alarms_under_shift(results) -> plt.Figure:
    changes = [
        "none",
        "brighter_x1_3",
        "darker_x0_7",
        "more_contrast_x1_5",
        "less_contrast_x0_6",
        "more_colour_x1_5",
        "less_colour_x0_5",
        "hue_shift_0_05",
        "blur_sigma_0_8",
        "sensor_noise_4_255",
    ]
    names = [
        "none",
        "brighter ×1.3",
        "darker ×0.7",
        "more contrast",
        "less contrast",
        "more colour",
        "less colour",
        "hue shift",
        "blur",
        "sensor noise",
    ]
    checks = ["picture_window", "label_window", "label_pattern"]
    values = np.array(
        [[results.metric(RUNS["shift"], f"{c}_{k}") for c in checks] for k in changes]
    )
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    heatmap(ax, values, names, ["picture check", "label check", "label pattern"])
    ax.set_title(
        "False alarms after harmless changes\n(nothing is poisoned: every alarm is false)",
        fontsize=10,
    )
    return fig


def overhead(results) -> plt.Figure:
    settings = [
        "picture_check_every_batch",
        "label_check_every_batch",
        "both_checks_every_batch",
        "both_checks_every_4th_batch",
        "both_checks_every_8th_batch",
    ]
    names = [
        "picture check\nevery batch",
        "label check\nevery batch",
        "both\nevery batch",
        "both every\n4th batch",
        "both every\n8th batch",
    ]
    small = [results.metric(RUNS["overhead_small"], f"overhead_{s}") * 100 for s in settings]
    resnet = [results.metric(RUNS["overhead_resnet"], f"overhead_{s}") * 100 for s in settings]
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    x = np.arange(len(settings))
    label_bars(
        ax, ax.bar(x - 0.2, small, 0.4, color=CLEAN_COLOUR, label="SmallCNN victim"), "{:.1f}%"
    )
    label_bars(
        ax,
        ax.bar(x + 0.2, resnet, 0.4, color=ATTACK_COLOUR, hatch="//", label="ResNet-18 victim"),
        "{:.1f}%",
    )
    ax.axhline(5, color="black", linestyle="--", linewidth=0.8, label="NFR-P1 target (5%)")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("extra training time (%)")
    ax.set_ylim(0, 30)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Cost of checking (quiet-machine run; laptop timings vary ±5 points)", fontsize=10)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="local-docs/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")

    results = Results()
    device = get_device()
    data = detection_data(42, 4000, GM_CACHE)
    figures = {
        "step03-1-reconstruction-examples": lambda: reconstruction_examples(data, device),
        "step03-2-picture-check-auc": lambda: auc_bars(results),
        "step03-3-picture-score-histograms": lambda: score_histograms(results, data, device),
        "step03-4-who-catches-what": lambda: who_catches_what(results),
        "step03-5-false-alarms-under-shift": lambda: false_alarms_under_shift(results),
        "step03-6-overhead": lambda: overhead(results),
    }
    for name, draw in figures.items():
        fig = draw()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"saved {out / name}.png")


if __name__ == "__main__":
    main()
