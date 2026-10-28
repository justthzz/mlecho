# MLEcho

**A Real-Time CNN-VAE and Reinforcement Learning Framework for Detecting and Responding to Training Data Poisoning in MLOps Pipelines.**

> ⚠️ **Work in progress.** MLEcho is being built as a Final Year Project (FYP). Detection and attack classification are built and evaluated as research code; the response module and the `Guard` API are not built yet. See [Status](#status).

## What is MLEcho?

MLEcho monitors training batches as they stream through an MLOps pipeline and flags poisoned data **during** training, rather than before or after it. It is designed to combine:

- A **CNN feature extractor** and **VAE anomaly scorer** that flag anomalous images by reconstruction error. Clean images reconstruct faithfully (a true "echo"); poisoned ones produce a distorted one. A **label check** covers attacks that change only labels.
- An **attack-type classifier** that identifies the likely poisoning strategy (backdoor, label-flip, gradient-based, accumulative).
- Two interchangeable **response policies** that choose an action (continue, quarantine, rollback, retrain, escalate):
  - a deterministic **rule-based** policy
  - a **PPO reinforcement-learning** agent, evaluated against the rule-based baseline
- **MLflow** logging of every detection and response event, for auditability.

## Installation

```bash
pip install mlecho
```

> The release on PyPI (0.0.1) is a placeholder: `mlecho.detect()` raises `NotImplementedError`. To use the current code, install from source (see [Development](#development)).

## Status

| Part | State |
|---|---|
| Research test bench: CIFAR-10 split, SmallCNN victim model, train/eval loop | ✅ working (clean baseline: 85.7% test accuracy, logged to MLflow) |
| Poisoning attacks | ✅ all 4 built and tested: backdoor (attack success rate 99.96%), label-flip (cat accuracy 69% → 19%), accumulative (Pang et al. partially reproduced; slow-drip version: ASR 98% with 1 poisoned image per batch), gradient-based (Witches' Brew: 3 of 4 near targets fooled, clean-label) |
| CNN-VAE detector and label check | ✅ built and evaluated: picture check catches the backdoor (AUC 0.994; 88% of poisoned images at 1% false alarms) and slow-drip; label pattern check catches label-flips between distinct classes; gradient matching and cat→dog not detected; false alarms under harmless data changes measured |
| Attack-type classifier | ✅ built and evaluated: trained only on made-up attacks, names the real backdoor (86.5%), label-flip (98%) and slow-drip (76.5–99%) and calls 97% of clean windows clean; also tells harmless data changes apart; gradient matching can't be classified (not detected) |
| Rule-based response and actions | ⬜ not started |
| `Guard` library API and MLflow event logging | ⬜ not started |
| PPO response agent | ⬜ not started |

Detailed plan: [docs/ROADMAP.md](docs/ROADMAP.md). Results so far: [docs/RESULTS.md](docs/RESULTS.md).

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[mlflow,dev]"

ruff format src tests experiments && ruff check . && pytest   # before every commit
python experiments/01_clean_baseline.py                        # clean CIFAR-10 baseline (30 epochs)
mlflow ui --port 5050                                          # view results at http://127.0.0.1:5050
```

Code layout and rules: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT. See [LICENSE](LICENSE).
