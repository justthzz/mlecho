# MLEcho

**A Real-Time CNN-VAE and Reinforcement Learning Framework for Detecting and Responding to Training Data Poisoning in MLOps Pipelines.**

> ⚠️ **Work in progress.** MLEcho is being built as a Final Year Project (FYP). The detection, classification and response modules are not functional yet. See [Status](#status).

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
| Research test bench: CIFAR-10 split, SmallCNN victim model, train/eval loop | ✅ working (clean baseline: 72.9% test accuracy, logged to MLflow) |
| Poisoning attacks | ⬜ not started |
| CNN-VAE detector and label check | ⬜ not started |
| Attack-type classifier | ⬜ not started |
| Rule-based response and actions | ⬜ not started |
| `Guard` library API and MLflow event logging | ⬜ not started |
| PPO response agent | ⬜ not started |

Detailed plan: [docs/ROADMAP.md](docs/ROADMAP.md).

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[mlflow,dev]"

ruff format src tests experiments && ruff check . && pytest   # before every commit
python experiments/01_clean_baseline.py --epochs 5             # clean CIFAR-10 baseline
mlflow ui                                                      # view results at http://127.0.0.1:5000
```

Code layout and rules: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT. See [LICENSE](LICENSE).
