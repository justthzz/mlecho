# MLEcho

**A Real-Time CNN-VAE and Reinforcement Learning Framework for Detecting and Responding to Training Data Poisoning in MLOps Pipelines.**

> ⚠️ **Work in progress.** MLEcho is under active development as part of an ongoing Final Year Project (FYP). Core detection, classification, and response modules are not yet functional — see [Status](#status) below.

## What is MLEcho?

MLEcho monitors training batches as they stream through an MLOps pipeline, flagging poisoned data during active training rather than before or after it. It combines:

- A **CNN feature extractor** and **VAE anomaly scorer** that flag anomalous batches by reconstruction error — clean batches reconstruct faithfully (a true "echo"), poisoned batches produce a distorted one.
- An **attack-type classifier** that identifies the likely poisoning strategy (backdoor, label-flip, gradient-based, accumulative).
- A **deterministic rule-based response module** (quarantine, rollback, retrain, escalate).
- *(Research enhancement, in progress)* A **PPO-based reinforcement-learning response agent**, evaluated against the rule-based baseline.

MLEcho integrates with **MLflow**, **Weights & Biases**, and **Docker** for experiment tracking, live monitoring, and reproducible deployment.

## Installation

```bash
pip install mlecho
```

## Status

This package currently ships a placeholder API only. Calling `mlecho.detect()` raises `NotImplementedError`. Follow the repository for progress as the core CNN-VAE detector, classifier, and response modules are implemented.

## License

MIT — see [LICENSE](LICENSE).