# MLEcho

**A Real-Time CNN-VAE and Reinforcement Learning Framework for Detecting and Responding to Training Data Poisoning in MLOps Pipelines.**

> ⚠️ **Work in progress.** MLEcho is being built as a Final Year Project (FYP). Detection, attack classification and the rule-based response are built and evaluated as research code; the `Guard` API is being finished. See [Status](#status).

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

> The release on PyPI (0.0.1) is an old placeholder. Version 0.1.0 (the `Guard` below) is not published yet: install from source (see [Development](#development)).

## Usage

```python
from mlecho import Guard, GuardConfig
from mlecho.tracking.audit import AuditLog
from mlecho.tracking.mlflow_logger import MLflowLogger   # needs mlecho[mlflow] and an active mlflow.start_run()

# Once: learn "normal" from your own trusted clean images (raw pixels, N x H x W x 3, 0-255).
config = GuardConfig(pixel_mean=MEAN, pixel_std=STD)     # how your model normalises images
guard = Guard.calibrate(trusted_images, trusted_labels, num_classes=10, device=device, config=config)
guard.save("guard.pt")

# Every training run:
guard = Guard.load("guard.pt", device, loggers=[AuditLog("audit.jsonl"), MLflowLogger()])
guard.watch(model, optimizer)                 # enables rollback and retrain
for images, labels in loader:                 # normalised, BEFORE random augmentation
    keep = guard.check(images, labels)        # False = left out (quarantine)
    loss = train_step(images[keep], labels[keep])
    guard.update(loss)                        # every 10 batches: verdict, action, audit event
guard.close()
```

Each window of 10 batches gets a verdict (clean, backdoor, label-flip, accumulative or harmless shift, with a confidence), and the rule-based policy chooses continue, quarantine, rollback, retrain or escalate. `check_every=n` scores only every n-th batch while nothing looks suspicious, to save time.

## Status

| Part | State |
|---|---|
| Research test bench: CIFAR-10 split, SmallCNN victim model, train/eval loop | ✅ working (clean baseline: 85.7% test accuracy, logged to MLflow) |
| Poisoning attacks | ✅ all 4 built and tested: backdoor (attack success rate 99.96%), label-flip (cat accuracy 69% → 19%), accumulative (Pang et al. partially reproduced; slow-drip version: ASR 98% with 1 poisoned image per batch), gradient-based (Witches' Brew: 3 of 4 near targets fooled, clean-label) |
| CNN-VAE detector and label check | ✅ built and evaluated: picture check catches the backdoor (AUC 0.994; 88% of poisoned images at 1% false alarms) and slow-drip; label pattern check catches label-flips between distinct classes; gradient matching and cat→dog not detected; false alarms under harmless data changes measured |
| Attack-type classifier | ✅ built and evaluated: trained only on made-up attacks, names the real backdoor (86.5%), label-flip (98%) and slow-drip (76.5–99%) and calls 97% of clean windows clean; also tells harmless data changes apart; gradient matching can't be classified (not detected) |
| Rule-based response and actions | ✅ built and evaluated in live training episodes: backdoor and slow-drip success drops from 97–100% to ~1% (clean level) within 20–100 batches; label-flip damage halved; no false actions on clean data; harmless changes escalated to a human, never deleted |
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
