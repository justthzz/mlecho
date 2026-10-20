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

## Experiment 03: targeted label-flip (cat → dog)

`python experiments/03_label_flip.py` · MLflow experiment `mlecho-attacks`

Attack: 2,250 training images of class 3 (cat) relabelled as class 5 (dog), pixels untouched. Same 5% budget as the backdoor (2,250 of 45,000), which is half of the 4,522 training cats. Same model and training as the current baseline.

| Model | Poison rate | Clean test accuracy | Cat accuracy (1,000 cat test images called "cat") | Flip rate (cat test images called "dog") | Run ID |
|---|---|---|---|---|---|
| clean | 0% | 85.7% | 69.0% | 12.5% | `0e827c6e` |
| attacked | 5% | 81.4% | **18.9%** | **62.1%** | `a67415a6` |

Per-class test accuracy (clean → attacked): cat 69.0% → 18.9% (−50.1), dog 82.2% → 90.9% (+8.7, the model now says "dog" more often), every other class within ±1.8 points.

Finding: relabelling half the training cats as dogs (5% of the training set, no pixel changes) cut cat accuracy from 69.0% to 18.9% and raised the cat→dog flip rate from 12.5% to 62.1%. The damage is confined to cats: other classes barely change, and overall accuracy drops 4.4 points (85.7% → 81.4%), almost all of it from the cat class. Unlike the backdoor, this attack is visible in per-class accuracy, but not in the images.

Caveat: one seed per condition so far; repeat with 3 seeds in step 8.

## Experiment 04: accumulative poisoning (Pang et al., 2021), online learning

`python experiments/04_accumulative.py` · tuning runs in MLflow experiment `mlecho-tuning` · sweep commands in `experiments/sweeps/accumulative_sweep.sh`

Status: **partially reproduced; not reported as a working attack.** Selection used validation accuracy only; no test numbers are reported for a winner because no setting met the secrecy criterion.

Setup: a burn-in model keeps learning online (one SGD step per batch of 100). In the accumulative phase each incoming batch is perturbed with PGD (10 steps, ℓ∞ bound ε). Then one update on a clean trigger batch. Secrecy criterion: validation accuracy stays within 3 points of the start for all steps (otherwise the attacker stops early).

| Sweep | Burn-in | Online lr | Formulation | Settings | Secret settings | Their trigger drop (val) | Largest drop in any setting (val) | Control drop |
|---|---|---|---|---|---|---|---|---|
| 1 | Adam + cosine, 85.7% | 0.001 | global, cosine/dot | 18 | 1 | 0.7 | 5.0 (accuracy already ≤ 65%) | −0.2 |
| 2 | SGD constant 0.01, 83.3% | 0.01 | global, cosine/dot | 12 | 2 | −0.1, 0.0 | 4.0 (from 61–79%) | −0.6 |
| 3 | SGD constant 0.01, 83.3% | 0.01 | per-layer (authors' code) | 9 | 0 | – | 5.3 (from 65%) | −0.6 |
| 4 | **GroupNorm**, SGD constant 0.01, 82.8% | 0.01, **momentum 0** | per-layer, **2 PGD steps** (authors' code) | 8 (ε 8/16/32/51, λ 0.5/1) | 4 (ε 8 λ 0.5/1, ε 32 λ 1, ε 51 λ 1) | 0.0, 0.0, −0.1, −0.1 | 0.0 (none) | 0.0 |

Sweep 4 (MLflow `mlecho-tuning`, control `2acabf43`) matched the authors' code in everything except the model: GroupNorm, online momentum 0, 2 PGD steps, per-layer cosines with equal weight (λ = 1), ε up to 0.2 ≈ 51/255. Burn-in (GroupNorm, SGD 0.01): 82.8% test (paper 83.4%). Half the settings stayed secret, but **no setting produced any trigger drop** (all val drops between −0.3 and +0.0 points); the others stopped early at ~80% val. With these settings each step's push is smaller than in sweeps 1–3, and on SmallCNN it doesn't build up into a crash.

Finding: the accumulative phase makes the model more sensitive to the trigger batch (drops up to ~5 points vs ≤ 0.6 for clean streams), but at our scale (SmallCNN with BatchNorm, 100 accumulative steps, 10 PGD iterations, Apple M2) the perturbations also lowered accuracy before the trigger, so the attack could not stay secret. The paper reports 82.09% → 27.66% for ResNet-18 with 1,000 steps and 100 PGD iterations. After sweep 4, the only remaining difference from the authors' code is the model (SmallCNN, 0.4M weights, vs ResNet-18, 11M) and the stream length, so model capacity is the most likely reason the collapse does not appear at our scale.

Also learned: an online learning rate of 0.01–0.1 broke a model trained with a cosine schedule (Adam) even without any attack; the single-step effect scales with the online learning rate.

## Experiment 05: slow-drip backdoor in online learning

`python experiments/05_slow_drip.py` · MLflow experiment `mlecho-attacks`

The simple accumulative variant used to test MLEcho. A burn-in model (SGD lr 0.01 constant, 83.3% test) keeps learning online (SGD lr 0.01, momentum 0.9, batches of 100, no augmentation). In every incoming batch a share of non-plane images gets the 4×4 yellow patch and the label "plane". Clean test accuracy and ASR (9,000 triggered non-plane test images) are measured during the stream.

| Run | Poison per batch | Steps | Poisoned images fed in | Clean test accuracy (start → end) | ASR over time | Run ID |
|---|---|---|---|---|---|---|
| control | 0 | 450 | 0 | 83.3% → 81.7% (range 81.7–84.3%) | 1.7% → 2.2% (≤ 3.8%) | `9c6fa516` |
| drip 5% | 5 of 100 | 450 | 2,250 | 83.3% → 82.0% (range 81.2–84.6%) | 1.7% → **99.6% by step 50** | `ed5b5913` |
| drip 1% | 1 of 100 | 200 | 200 | 83.3% → 82.7% (range 79.4–84.5%) | 1.7% → 41.1% (step 50) → 89.4% (step 100) → **98.3% (step 150)** | `be46a900` |

Finding: when poison arrives a little at a time in a live stream, the backdoor builds up gradually and is fully learned after about 150 poisoned images (1 per batch) or within 50 batches (5 per batch). Clean accuracy stays in the same range as the clean control throughout, so accuracy monitoring cannot see it. No single batch holds more than 1–5 poisoned images, which is the case MLEcho's per-image detection must handle.

## Experiment 06: gradient matching (Witches' Brew, Geiping et al., 2021), clean-label targeted

`bash experiments/sweeps/gradient_matching_sweep.sh` · MLflow experiment `mlecho-attacks`

Goal: make the victim call one chosen dog test image a frog. The attacker's model is the clean burn-in model (Adam + cosine, seed 42, 85.7%). From it, 450 training frogs (1% of train) are perturbed, labels unchanged, with the paper's settings: ε 16/255, 240 signed-Adam steps, differentiable flip + shift, 1 restart (the paper uses 8). The victim then trains a new SmallCNN from scratch on the poisoned set (30 epochs, Adam + cosine, augmentation) with a different seed (1). Targets: dog test images the attacker's model classifies correctly, in a fixed shuffled order (picks 0–4).

| Run | Target test image | Poisoned | Largest / mean pixel change | Clean test accuracy | Target called | P(frog) | Run ID |
|---|---|---|---|---|---|---|---|
| control | 5047, 2731, 1283, 6513, 6297 | 0 | – | 85.8% | dog (all 5) | 0.0% | `764e4c23` |
| pick 0 | 5047 | 450 | 16 / 10.5 (/255) | 85.7% | dog | 0.0% | `9ef95ea0` |
| pick 1 | 2731 | 450 | 16 / 10.5 | 85.9% | dog | 0.0% | `6abc30ff` |
| pick 2 | 1283 | 450 | 16 / 10.2 | 85.9% | dog | 0.0% | `ee148c05` |
| pick 3 | 6513 | 450 | 16 / 10.5 | 85.8% | dog | 0.0% | `76775ecc` |
| pick 4 | 6297 | 450 | 16 / 10.5 | 85.8% | dog | 0.0% | `2d741185` |

Finding (random dog → frog targets): **0 of 5 targets fooled.** The poison is hidden (clean accuracy 85.7–85.9%, same as the control) but has no measurable effect on these targets. Diagnosis: the crafting works (gradient cosine 0.015 → 0.911 after 240 steps), but the attacker's model is certain these dogs are not frogs (P(frog) ≈ 0), and even fine-tuning the attacker's own model on the poison only moved P(frog) to ≤ 0.2%.

**Easier setup (one bounded extra try):** cat → dog, 450 training dogs poisoned (labels unchanged), same ε, steps and budget. The target is the correctly classified cat test image that the attacker's model already finds most dog-like (`--target-order closest`, test image 8183). Two victims: fine-tuning the attacker's model (5 epochs, Adam 1e-4, constant), and a new model from scratch (seed 1, as above). Each has a no-poison control.

| Victim | Poisoned | Clean test accuracy | Target called | P(dog) | Run ID |
|---|---|---|---|---|---|
| fine-tune, control | 0 | 85.5% | cat | 26.1% | `3b9d12f5` |
| fine-tune, attack | 450 (largest / mean change 16 / 10.4 per 255) | 85.3% | **dog** | **64.2%** | `69ceb34c` |
| from scratch, control | 0 | 85.8% | cat | 22.1% | `152988a1` |
| from scratch, attack | 450 (16 / 10.4) | 85.6% | **dog** | **53.2%** | `dbefe861` |

Finding: **the attack works when the target is not already far from the wrong class.** With the same training and seed, the only difference between control and attack is the 450 invisibly changed, correctly labelled dogs; they flip the target from cat to dog in both fine-tuning (26% → 64%) and from-scratch training (22% → 53%), while clean accuracy stays within 0.3 points of the control. Limits of this first try: one target per setting, chosen as the easiest (most dog-like) cat.

**Success rate (from scratch, more targets).** A target is only a valid test if the clean victim (no poison, same seed) still calls it a cat; otherwise "fooled" would prove nothing. In the `closest` order, the clean victim already calls 11 of the first 15 cats "dog" (control `004e5cbc`: these cats are borderline, and a model with a different seed decides differently from the attacker's). That leaves 4 valid targets: picks 0, 1, 12, 14.

| Pick | Target test image | P(dog), no poison | P(dog), poisoned | Called | Clean test accuracy | Run ID |
|---|---|---|---|---|---|---|
| 0 | 8183 | 22.1% | 53.2% | **dog** ✅ | 85.6% | `dbefe861` |
| 1 | 4975 | 12.2% | 45.6% | cat ❌ | 85.4% | `65b96ff7` |
| 12 | 2231 | 16.1% | 65.2% | **dog** ✅ | 85.8% | `954b9141` |
| 14 | 669 | 44.0% | 52.9% | **dog** ✅ | 85.6% | `50a8de63` |
| control | – | – | – | – | 85.8% | `004e5cbc` |

Final finding: **3 of 4 valid targets fooled** in from-scratch training (the 4th moved from 12% to 46% P(dog) but stayed "cat"); every poisoned target's P(dog) rose by 9 to 49 points, while clean accuracy stayed within 0.4 points of the control (85.4–85.8% vs 85.8%). The attack is clean-label (450 correctly labelled dogs, largest change 16/255, mean ~10/255). It works for targets near the wrong class, and not for random, confidently classified targets (dog → frog: 0 of 5). The paper uses ResNet-18 and 8 restarts; we use SmallCNN and 1 restart, which likely explains the narrower range of targets.
