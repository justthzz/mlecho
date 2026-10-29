"""Step 5 figures: redraw every response chart with one command.

Reads the final runs of experiment 15 (episode seeds 1-3, listed in
docs/RESULTS.md) from MLflow. Trains nothing. Colour-blind safe (Okabe-Ito).

Run:   python experiments/figures_step05.py                (writes local-docs/figures/)
       python experiments/figures_step05.py --out some/folder
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from common import BLUE, GREY, ORANGE, VERMILLION, Results

RUNS = {  # final runs (docs/RESULTS.md, experiment 15): policy -> one run per episode seed
    "none": ["6c468177", "e424fc83", "068ff409"],
    "rules": ["89584acc", "8d86727c", "8348b294"],
    "quarantine-everything": ["792660a6", "e85b9a5f", "b9e16971"],
}
COLOURS = {"none": VERMILLION, "rules": BLUE, "quarantine-everything": ORANGE}
NAMES = {
    "none": "no response",
    "rules": "MLEcho rules",
    "quarantine-everything": "quarantine all",
}
LINES = {"none": "-", "rules": "-", "quarantine-everything": "--"}  # dashed: overlaps show
ONSET = 100
SCENARIOS = {
    "clean": "clean",
    "backdoor": "backdoor 5%",
    "label_flip": "label-flip cat→dog",
    "slow_drip_1": "slow-drip 1/batch",
    "slow_drip_5": "slow-drip 5/batch",
    "gradient_matching": "gradient matching",
    "benign_brighter": "harmless brighter",
}


def curves(results, policy, scenario, key) -> tuple[np.ndarray, np.ndarray]:
    """Batches and the mean of `key` over the episode seeds."""
    runs = [results.json(r, f"episode_{scenario}.json")["curve"] for r in RUNS[policy]]
    batches = np.array([point["batch"] for point in runs[0]])
    return batches, np.mean([[point[key] for point in run] for run in runs], axis=0) * 100


def over_time(results, panels, key, ylabel, title) -> plt.Figure:
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.4), sharey=True)
    for ax, scenario in zip(np.atleast_1d(axes), panels, strict=True):
        for policy in RUNS:
            batches, values = curves(results, policy, scenario, key)
            ax.plot(
                batches,
                values,
                LINES[policy],
                color=COLOURS[policy],
                label=NAMES[policy],
            )
        ax.axvline(ONSET, color=GREY, linestyle="--", linewidth=1)
        ax.text(ONSET + 4, 50, "attack starts", fontsize=7, color=GREY, rotation=90)
        ax.set_title(SCENARIOS[scenario], fontsize=9)
        ax.set_xlabel("training batches")
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(title + " (mean of 3 episodes)", fontsize=10)
    fig.tight_layout()
    return fig


def final_bars(results) -> plt.Figure:
    """Harm at the end of each episode: ASR for trigger attacks, cats called dog for
    the label-flip, and the clean accuracy everywhere."""
    harm = {
        "backdoor": "asr",
        "slow_drip_1": "asr",
        "slow_drip_5": "asr",
        "label_flip": "cat_as_dog",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (metric, names, ylabel) in zip(
        axes,
        [
            ("harm", list(harm), "attack effect at the end (%)"),
            ("clean_acc", list(SCENARIOS), "clean test accuracy at the end (%)"),
        ],
        strict=True,
    ):
        x = np.arange(len(names))
        for k, policy in enumerate(RUNS):
            values = [
                np.mean(
                    [
                        results.metric(r, f"{harm[n] if metric == 'harm' else metric}_{n}") * 100
                        for r in RUNS[policy]
                    ]
                )
                for n in names
            ]
            ax.bar(x + (k - 1) * 0.27, values, 0.27, color=COLOURS[policy], label=NAMES[policy])
        ax.set_xticks(x, [SCENARIOS[n] for n in names], fontsize=7, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
    axes[0].set_title("ASR (triggers) / cats called dog (label-flip): lower is better", fontsize=9)
    axes[1].set_ylim(70, 90)
    axes[1].set_title("Clean accuracy: the response must not hurt it", fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def cost(results) -> plt.Figure:
    """Images left out by quarantine, and poison the model still learned from."""
    names = list(SCENARIOS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, title in [
        (axes[0], "clean_dropped", "clean images left out (cost)"),
        (axes[1], "poison_trained", "poisoned images still learned from"),
    ]:
        x = np.arange(len(names))
        for k, policy in enumerate(RUNS):
            values = [
                np.mean([results.metric(r, f"{key}_{n}") for r in RUNS[policy]]) for n in names
            ]
            ax.bar(x + (k - 1) * 0.27, values, 0.27, color=COLOURS[policy], label=NAMES[policy])
        ax.set_xticks(x, [SCENARIOS[n] for n in names], fontsize=7, rotation=25, ha="right")
        ax.set_title(title + " (mean of 3 episodes, 300 batches after the onset)", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


RECOVERED = {  # the attack's effect counts as gone when it stays below this until the end
    "backdoor": ("asr", 0.10),
    "slow_drip_1": ("asr", 0.10),
    "slow_drip_5": ("asr", 0.10),
    "label_flip": ("cat_as_dog", 0.25),
}


def recovery_batches(results, policy, scenario) -> list[int | None]:
    """Per episode: batches after the onset until the effect stays below its level.

    None = still above it at the end.
    """
    key, level = RECOVERED[scenario]
    out = []
    for run in RUNS[policy]:
        curve = results.json(run, f"episode_{scenario}.json")["curve"]
        after = [p for p in curve if p["batch"] > ONSET]
        bad = [p["batch"] for p in after if p[key] >= level]
        if bad and bad[-1] == after[-1]["batch"]:
            out.append(None)  # still above the level at the end
        else:
            out.append(
                (bad[-1] if bad else ONSET) - ONSET + (after[1]["batch"] - after[0]["batch"])
            )
    return out


def print_recovery(results) -> None:
    print("batches after the onset until the attack's effect is gone (per episode)")
    for scenario in RECOVERED:
        line = f"  {SCENARIOS[scenario]:20s}"
        for policy in RUNS:
            values = recovery_batches(results, policy, scenario)
            line += f"  {NAMES[policy]}: " + ", ".join(
                "never" if v is None else str(v) for v in values
            )
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="local-docs/figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")

    results = Results()
    figures = {
        "step05-1-asr-over-time": lambda: over_time(
            results,
            ["backdoor", "slow_drip_1", "slow_drip_5"],
            "asr",
            "attack success rate (%)",
            "Backdoor success during training",
        ),
        "step05-2-label-flip-over-time": lambda: over_time(
            results, ["label_flip"], "cat_acc", "cat test accuracy (%)", "Cats recognised"
        ),
        "step05-3-final-effect": lambda: final_bars(results),
        "step05-4-cost": lambda: cost(results),
    }
    print_recovery(results)
    for name, draw in figures.items():
        fig = draw()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"saved {out / name}.png")


if __name__ == "__main__":
    main()
