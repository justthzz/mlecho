# MLEcho results

Official results only, one section per experiment. Every number links to its
MLflow run (`mlflow ui --port 5050`), where the full settings, curves and saved
model are stored. Other runs in MLflow are quick checks or failed tries.

Setup for all runs unless stated: CIFAR-10, fixed split (45,000 train /
5,000 val / 10,000 test), seed 42, Apple M2 (`mps`).

## Experiment 01: clean baseline

`python experiments/01_clean_baseline.py` · MLflow experiment `mlecho-baselines`

| Version | Model | Training | Test accuracy | Run ID |
|---|---|---|---|---|
| first | SmallCNN width 32 (114,186 weights) | 5 epochs, no augmentation, constant LR | 72.9% | `8d1c1e88` (repeated: `dc71b1cc`) |
| + augmentation | SmallCNN width 32 | 30 epochs, augmentation, cosine LR | 82.1% | `4775f43d` |
| **current baseline** | **SmallCNN width 64 (412,682 weights)** | **30 epochs, augmentation, cosine LR** | **85.7%** | **`88ed4548`** |

- Reproducibility (NFR-R1): the first version gave 72.9% in repeated runs with the same seed. The current baseline gave 85.72% both in experiment 01 and in experiment 02 with 0% poison.
- Fit: train 87.3% / val 86.8% at epoch 30, so no underfitting or overfitting.
- Time: ~10 minutes per run (~20 s per epoch).

## Experiment 02: backdoor attack (BadNets)

`python experiments/02_backdoor_attack.py` · MLflow experiment `mlecho-attacks`

Attack: 4×4 yellow patch in the bottom-right corner, label changed to class 0 (plane), applied to 5% of the training set (2,250 of 45,000 images, spread evenly over the 9 other classes). Same model and training as the current baseline.

| Model | Poison rate | Clean test accuracy (10,000 normal test images) | ASR (9,000 non-plane test images + patch, % predicted "plane") | Run ID |
|---|---|---|---|---|
| clean | 0% | 85.7% | 1.6% (145 / 9,000) | `3f507100` |
| poisoned | 5% | 85.4% | **99.96%** (8,996 / 9,000) | `80c612ac` |

Finding: a 5% BadNets backdoor reduced clean accuracy by only 0.3 points (85.7% → 85.4%) while raising the attack success rate from 1.6% to 99.96%. The attack is hidden from normal testing and almost always works.

Caveat: one seed per condition so far; repeat with 3 seeds in step 8 before reporting final numbers.
