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

## Experiment 07: CNN-VAE picture check, feasibility (step 3.2)

`python experiments/07_detector_feasibility.py` · MLflow experiment `mlecho-detection` · run `daf37fdc`

Setup: the CNN-VAE (183k weights; latent 32; β 1; Adam 1e-3, 60 epochs) is trained on 4,000 trusted clean **validation** images only; it never sees training images. 1,000 more validation images are held out. Score = reconstruction error. Separation is measured with ROC AUC against 5,000 clean **training** images (1.0 perfect, 0.5 coin flip).

**Choosing the score fairly.** Candidate scores: whole-image mean error, or the worst k×k patch (k = 2, 4, 8). The choice is made on development data only: the 1,000 held-out trusted images with made-up attacks the guard could create itself (a 4×4 square of random colour at a random place; ±8/255 random noise), never on the real attacks. Development AUC (square / noise): mean 0.575 / 0.544, worst 2×2 0.625 / 0.508, **worst 4×4 0.760 / 0.510**, worst 8×8 0.658 / 0.518 → **worst 4×4 chosen**.

| Images scored | n | AUC, whole-image mean | AUC, worst 4×4 (chosen) | Poisoned scores above its own original (4×4) |
|---|---|---|---|---|
| clean held-out (sanity) | 1,000 | 0.526 | 0.520 | – |
| backdoor (4×4 yellow square) | 2,250 | 0.682 | **0.994** | 99.9% |
| label-flip (pictures untouched) | 2,250 | 0.498 | 0.502 | – |
| gradient matching (cat 8183 → dog) | 450 | 0.587 | 0.570 | 55.3% |

(All four scorings are logged; worst 2×2 / 8×8 give backdoor AUC 0.915 / 0.912.)

Finding: the picture check **separates the backdoor almost perfectly** once it looks at the worst local patch instead of the whole-image average (0.68 → 0.99): the square always adds error, but a whole-image mean dilutes 16 of 1,024 pixels below the natural spread between images. It does **not** see label-flip (expected: the pictures are normal; that is the label check's job) and barely sees **gradient matching** (0.57; invisible ±16/255 changes look like natural variation to the VAE), matching hypothesis H2 that clean-label attacks are the hardest. Clean unseen images are not flagged (0.52). An earlier run with only the whole-image mean (`5567d473`) gave the same mean-score numbers (backdoor 0.682, gradient matching 0.586).

## Experiment 08: picture check detection rates (step 3.3)

`python experiments/08_picture_check_detection.py` · MLflow experiment `mlecho-detection` · run `b3727d2f` (rerun after refactoring, identical numbers: `613f2371`)

The CNN-VAE from experiment 07 with its chosen score (worst 4×4 patch). All thresholds come from the 1,000 trusted held-out images only: **image threshold** = 1% of clean images flagged (score 2.73); **batch alarm** = at least 5 flags in a batch of 128 (≤ 1% of clean batches); **window alarm** = at least 23 flags in 10 batches (≤ 1% of clean windows). Evaluated on the 45,000 training images, which the detector never saw.

Per image:

| Attack | Poisoned / total | Recall | False-positive rate | Precision |
|---|---|---|---|---|
| backdoor | 2,250 / 45,000 | **88.3%** | 0.99% | **82.5%** |
| label-flip | 2,250 / 45,000 | 1.2% | 0.96% | 6.2% |
| gradient matching | 450 / 45,000 | 0.4% | 0.97% | 0.5% |

Per batch (128 images) and per window (10 batches), 1,000 simulated batches each (100 windows):

| Stream | Mean flags per batch | Batch alarm | Window alarm |
|---|---|---|---|
| clean | 1.21 | 0.5% | 0.0% |
| backdoor (5%, as poisoned) | 6.73 | **80.7%** | **100%** |
| label-flip (5%) | 1.27 | 1.0% | 1.0% |
| gradient matching (1%) | 1.25 | 1.6% | 1.0% |
| slow-drip, 1 per batch | 2.10 | 3.3% | **40%** |
| slow-drip, 5 per batch | 5.59 | **81.0%** | **100%** |

Finding: the false-positive rates measured on unseen training images (0.96–0.99% per image, 0.5% per batch, 0% per window) match the 1% targets set on the trusted data, so the calibration transfers. The picture check catches the **backdoor** (88% of poisoned images, 81% of batches, every 10-batch window) and **slow-drip at 5 per batch**. At **1 per batch**, single batches rarely alarm (3.3%), but adding up flags over 10 batches catches 40% of windows: cross-batch evidence matters, as expected. **Label-flip and gradient matching are invisible** to the picture check (at chance level); label-flip is the label check's job (step 3.4), and gradient matching remains the hardest case (H2).

## Experiment 09: label check detection rates (step 3.4)

`python experiments/09_label_check_detection.py` · MLflow experiment `mlecho-detection` · run `02cb63dd`

Setup: small reference classifiers (`ReferenceCNN`, width 32) trained only on the 5,000 trusted validation images with **5-fold cross-fitting** (each model trains on 4,000 and predicts the other 1,000), so the check's normal behaviour is measured out-of-fold on all 5,000 (500 per class). Out-of-fold accuracy: **64.5%**. Each training image is checked by one randomly chosen fold model. Two checks:
- **per image:** surprise = −log P(label | picture); thresholds as in experiment 08 (1% of trusted images, 1% of batches / windows);
- **label pattern:** in a window of batches, the most over-represented (given label, looks-like class) pair, as a pooled two-proportion z against the trusted data; alarm at z > 3.69 (1% split over all 90 pairs, Bonferroni).

Per image:

| Attack | Recall | False-positive rate | Precision |
|---|---|---|---|
| backdoor (picture labelled plane) | 45.8% | 0.94% | 72.0% |
| label-flip cat → dog | 4.0% | 0.95% | 18.4% |
| gradient matching | 0.4% | 0.94% | 0.5% |

Per batch / 10-batch window (flag counts): clean 0.1% / 0%; backdoor 23% / **100%**; label-flip 0.1% / 0%; gradient matching 0.2% / 1%; slow-drip 1 per batch 0.6% / 9%; 5 per batch 10% / **100%**.

Label pattern (window alarm rate; "right pair" = share of alarms naming the flipped pair):

| Stream | 10-batch window | 50-batch window |
|---|---|---|
| clean | 1.0% | 0.0% |
| backdoor | 67% | 88% |
| gradient matching | 2% | 0% |
| **label-flip cat → dog (the attack)** | 4% | 2% (right pair 100%) |
| label-flip deer → horse (unseen pair) | **93%** (right 100%) | **100%** (right 100%) |
| label-flip truck → car (unseen pair) | **100%** (right 100%) | **100%** (right 100%) |
| label-flip plane → bird (unseen pair) | **100%** (right 100%) | **100%** (right 100%) |

Finding: false alarms match the 1% targets. The label pattern check **catches label-flip between classes the reference model rarely confuses and names the flipped pair correctly** (three pairs never looked at during design: 93–100% of windows). It **misses cat → dog**: the reference model (only 64.5% accurate from 5,000 images) already calls ~20% of dogs "cat", so flipping half the cats only raises that to ~28%, below what 1,280–6,400 images can prove against a 500-image reference. A stronger reference model (more trusted data or pretrained features) is the main lever; the label check also adds evidence for the backdoor (100% of 10-batch windows). Gradient matching stays invisible (correct labels).

Development history (logged, superseded): `6d5e5e93` one reference model, per image only (label-flip recall 3.2%); `87101152` first label-pattern version with thresholds bootstrapped from the 1,000 held-out images, **miscalibrated** (84% of clean 10-batch windows alarmed, because it ignored that the trusted rates are estimates from ~100 images per class); `1572d1a7` two-proportion z with one model and per-batch patterns (27% clean batch alarms: counts too small for the normal approximation). The final design fixes both: cross-fitting (500 per class) and windows only.

## Experiment 10: detection overhead (step 3.5, NFR-P1 < 5%)

`python experiments/10_detection_overhead.py [--victim resnet18]` · MLflow experiment `mlecho-detection`

Times 200 training batches (128 images, augmentation, Adam; data loading included) with and without the checks running on the batch. Picture check = CNN-VAE worst 4×4 score; label check = one reference model.

| Setting | SmallCNN, run 1 (median of 3, quiet machine) | SmallCNN, run 3 (fastest of 5) | ResNet-18, run 1 (median of 3) | ResNet-18, run 3 (fastest of 5) |
|---|---|---|---|---|
| picture check, every batch | +3.8% | +6.0% | +2.7% | +10.8% |
| label check, every batch | +14.5% | +25.6% | +16.5% | +19.5% |
| both checks, every batch | +20.1% | +22.6% | +23.3% | +23.5% |
| both checks, every 4th batch | +6.3% | +5.2% | +5.4% | +12.5% |
| both checks, every 8th batch | +8.2% | +4.6% | +3.5% | +6.8% |
| picture every batch, label every 8th | – | +8.2% | – | +6.1% |

Runs: run 1 `faef6394` (SmallCNN) / `b58ab3fa` (ResNet-18) (an earlier SmallCNN run `ae6e9914` without the 8th-batch setting gave the same picture); run 2 (`exp10-11` log, 3 rounds) was disturbed by other load on the laptop (single runs up to 33 s vs 8.5 s) and is not used; run 3 `f7952d97` / `3df7abed` (fastest of 5, load average ≈ 3).

Finding: running **both checks on every batch costs ~20–25%**, far above the 5% target; most of it is the label check (~15–25%). The picture check alone costs ~3–11%. Checking only every 8th batch brings both checks to **~3.5–8%**. Measurements on this shared laptop vary by about ±5 points between runs, so the 5% target is met only by sampled checking and must be re-measured on an idle machine for the final report. ResNet-18 is only ~2× slower per batch than SmallCNN here because CPU data loading dominates, so a bigger victim doesn't shrink the relative overhead as much as expected. Design consequence: the guard needs a `check_every` setting (picture check frequently, label check on sampled batches, which suits its window-level pattern test).

## Experiment 11: false alarms under legitimate change (step 3.6)

`python experiments/11_false_alarms_under_shift.py` · MLflow experiment `mlecho-detection` · run `7bbc3de8`

10,000 clean training images with harmless changes applied; thresholds calibrated on the unchanged trusted data (as in experiments 08 and 09). No image is poisoned, so every alarm is false. Rates: share of images flagged / batches alarmed / 10-batch windows alarmed; label pattern = 10-batch windows.

| Change | Picture: image / batch / window | Label: image / batch / window | Label pattern |
|---|---|---|---|
| none | 1.0% / 0.8% / 0% | 1.0% / 0.9% / 1% | 4% |
| brighter ×1.3 | 5.2% / 79.6% / **100%** | 1.7% / 8.6% / 49% | 76% |
| darker ×0.7 | 0% / 0% / 0% | 1.0% / 0.9% / 0% | 83% |
| more contrast ×1.5 | 17.7% / **100%** / **100%** | 2.2% / 14% / 83% | 89% |
| less contrast ×0.6 | 0% / 0% / 0% | 1.7% / 8.2% / 46% | **100%** |
| more colour ×1.5 | 1.7% / 7.8% / 51% | 1.4% / 4.0% / 13% | 21% |
| less colour ×0.5 | 0.9% / 0.6% / 0% | 1.0% / 1.4% / 0% | 24% |
| hue shift 0.05 | 1.0% / 0.6% / 0% | 1.4% / 3.4% / 12% | 93% |
| blur (σ 0.8) | 0.1% / 0% / 0% | 4.4% / 66.9% / **100%** | **100%** |
| sensor noise 4/255 | 0.9% / 0.5% / 0% | 1.0% / 0.4% / 1% | 75% |

(`4e8a81f8` is an earlier run of the same script that stopped before logging, on an invalid metric name.)

Finding: on unchanged data the false-alarm rates match the 1% targets (label pattern 4% on these 100 windows). But **harmless changes trigger the alarms**: the picture check fires on brighter or higher-contrast images (100% of windows), the label check on blur, and the label pattern test on almost every change (75–100%), because any change shifts the reference model's confusions. The checks detect **"different from the trusted data"**, not "poisoned" specifically. Consequences: (1) the guard must be recalibrated with a fresh trusted sample when the data source changes (stated assumption: calibration data is clean and representative); (2) step 4's classifier should include a "benign shift" class so the response can escalate to a human instead of quarantining data.

## Experiment 12: statistical drift baselines (step 3.7)

`python experiments/12_drift_baselines.py` · MLflow experiment `mlecho-detection` · run `911ca15f`

What a practitioner might use instead: a **KS test** on simple picture statistics (mean and spread of R, G, B per image; Bonferroni over 6 features, alarm at p < 1%) and **PSI on the label counts** (alarm at the usual rule of thumb PSI > 0.1, and with a calibrated threshold: exceeded by 1% of bootstrapped windows of trusted labels, 0.0059 for 10 batches / 0.0012 for 50). Reference: the 5,000 trusted images. 100 windows per case.

| Case | KS (10 / 50 batches) | PSI > 0.1 | PSI calibrated (10 / 50) |
|---|---|---|---|
| clean | 0% / 0% | 0% / 0% | **8% / 35%** |
| backdoor | 1% / 0% | 0% / 0% | 89% / 100% |
| label-flip cat → dog | 0% / 0% | 0% / 0% | **100% / 100%** |
| gradient matching | 0% / 0% | 0% / 0% | 3% / 46% |
| slow-drip, 1 per batch | 1% / 0% | 0% / 0% | 3% / 23% |
| slow-drip, 5 per batch | 0% / 0% | 0% / 0% | 46% / 100% |
| harmless brighter ×1.3 | **100% / 100%** | 0% / 0% | 3% / 43% |

Finding: the **standard drift checks catch none of the attacks** and fire only on the harmless brightness change (KS 100%): they detect "different", not "poisoned". A label-count check with a **calibrated** threshold catches label-flip cat → dog (100%), which the label check missed, but it **over-alarms on clean data (8–35%)**: the trusted sample's class mix differs slightly from the training data's (e.g. planes 10.6% vs 9.93%), which large windows can detect. MLEcho's picture check catches the backdoor and slow-drip with ~1% false alarms (experiment 08), which these baselines can't. Earlier runs of this script: `b1012b1c`, `39f0e53d` (KS and rule-of-thumb PSI only, same numbers).

## Experiment 13: picture check with 5-fold cross-fitting (step 4.1)

`python experiments/13_cross_fitted_picture_check.py` · MLflow experiment `mlecho-detection` · run `78942c03`

Why: the single VAE of experiment 07 partly remembers its 4,000 training images (it flags 0.37% of them vs 1.0% of the held-out images; fit vs held-out score AUC 0.59). The attack classifier learns from windows built out of trusted images, so their scores must be out-of-fold. Same fix as the label check: 5 VAEs (same settings as experiment 07, 60 epochs), each trained on 4,000 trusted images and scoring the other 1,000 (the folds of experiment 09). Thresholds now use all 5,000 out-of-fold scores: image threshold 2.72, batch alarm at ≥ 6 flags of 128, window alarm at ≥ 23 flags in 10 batches. Each training image is scored by one randomly chosen fold VAE.

| | Single VAE (exp 08) | Cross-fitted (exp 13) |
|---|---|---|
| backdoor recall / precision per image | 88.3% / 82.5% | 88.9% / 83.7% |
| clean false-positive rate per image | 0.99% | 0.91% |
| clean batch / window alarm | 0.5% / 0% | 0.0% / 0% |
| backdoor batch / window alarm | 80.7% / 100% | 65.3% / 100% |
| slow-drip 1 per batch, window alarm | 40% | 30% |
| slow-drip 5 per batch, batch / window alarm | 81.0% / 100% | 49.0% / 100% |
| label-flip / gradient matching, window alarm | 1% / 1% | 2% / 0% |

Finding: the cross-fitted picture check behaves like the single VAE per image (recall and false-positive rate within 1 point). The batch level is stricter (≥ 6 flags instead of ≥ 5, because 5,000 reference scores estimate the clean tail differently from 1,000), so fewer batches alarm; windows are unchanged except slow-drip at 1 per batch (30% vs 40%, both on 100 windows). The step 3 conclusions stand; step 4 uses these out-of-fold scores.

## Experiment 14: attack classifier (step 4, RQ2)

`python experiments/14_attack_classifier.py [--classifier-seed S] [--model forest]` · MLflow experiment `mlecho-detection` · official run `e35fb511` (softmax, seed 42); repeats `0bdcd525` (seed 1), `7ae2e890` (seed 2); comparison `dcf6afa4` (random forest, seed 42); rerun after the step 6 refactor (made-up windows moved into the library): `32b43b58`, all 200 metrics identical

Setup: one **window** = 10 batches of 128 images. For each window, 10 numbers (`classification/features.py`), each measured against the trusted data: picture flag rate, picture steadiness (mean − variance of flags per batch), median picture score, top-label share and label surprise of flagged pictures, label flag rate, median label surprise, top-label share of surprising labels, label pattern z, label-mix PSI. Out-of-fold evidence: 5 fold VAEs (experiment 13) and 5 fold reference models (experiment 09). Classifier: softmax regression (L-BFGS, L2 1e-3) over 5 types: clean, backdoor, label-flip, accumulative, harmless shift.

**Trained only on made-up attacks** on the 5,000 trusted images (`classification/synthetic.py`): 40 random triggers (2–6 px, random place and target class; half vivid colours, half any colour) at 2–10% of the data (backdoor) or 1–6 per batch (accumulative); random flip pairs at 1–5% of the data, never the 4 test pairs (either direction); 40 random brightness/contrast/blur changes on 30–100% of a window. 600 windows per type. Made-up clean windows use a two-level bootstrap (`stream_rows`) so the label mix varies as much as real data does. Development check on new made-up settings (10 new triggers and shifts): **83.9%** (clean 92%, backdoor 74%, label-flip 87%, accumulative 70%, shift 96.5%).

Tested on the 45,000 training images (never used for training or choices), 200 windows per case. "Right" = the expected type; for harmless shifts, "clean" also counts as right.

| Test case | Seen kind? | Softmax seed 42 | Seeds 42 / 1 / 2 | Forest | Main mistake |
|---|---|---|---|---|---|
| clean | – | **97.0%** | 97.0 / 96.0 / 97.5 | 99.5% | harmless shift 2% |
| backdoor 5% (official) | yes | **86.5%** | 86.5 / 82.0 / 92.5 | 79.5% | accumulative 14% |
| label-flip cat → dog 5% (official) | yes | **98.0%** | 98.0 / 98.5 / 98.5 | 99.5% | – |
| slow-drip 1 per batch (official) | yes | **76.5%** | 76.5 / 74.5 / 82.5 | 81.5% | clean 13% |
| slow-drip 2 per batch | yes | **99.0%** | 99.0 / 99.5 / 98.5 | 99.5% | – |
| slow-drip 5 per batch (official) | yes | **98.0%** | 98.0 / 99.0 / 93.5 | 97.5% | backdoor 2% |
| backdoor, checkerboard trigger 5% | unseen trigger | 93.5% | 93.5 / 92.5 / 93.0 | 87.5% | accumulative 6% |
| backdoor yellow 2% | rate inside training range | **17.5%** | 17.5 / 14.0 / 9.5 | 31.5% | **accumulative 81%** |
| backdoor yellow 10% | edge of training range | 100% | 100 / 100 / 100 | 100% | – |
| slow-drip checkerboard 2 per batch | unseen trigger | 87.5% | 87.5 / 85.5 / 92.5 | 89.0% | harmless shift 6% |
| label-flip cat → dog 2% | weaker | **5.5%** | 5.5 / 3.0 / 5.0 | 17.0% | **clean 88%** |
| label-flip deer → horse 5% | unseen pair | 96.0% | 96.0 / 99.0 / 99.5 | 92.0% | backdoor 4% |
| label-flip truck → car 5% | unseen pair | 100% | 100 / 100 / 100 | 100% | – |
| label-flip plane → bird 5% | unseen pair | 99.0% | 99.0 / 100 / 99.5 | 96.0% | – |
| gradient matching (no class) | – | clean 95% | 95 / 96 / 94 | clean 98% | (undetectable, exp 08/09) |
| brighter / darker / more or less contrast / blur | in-family | 100% each | 100 each | 100% each | – |
| more colour ×1.5 | unseen kind | 92.0% | 92.0 / 92.5 / 88.5 | 98.0% | accumulative 5% |
| less colour ×0.5 | unseen kind | 99.0% | 99.0 / 99.5 / 99.0 | 93.5% | – |
| hue shift 0.05 | unseen kind | 79.5% | 79.5 / 78.0 / 72.5 | **40.0%** | label-flip 20% |
| sensor noise 4/255 | unseen kind | 83.0% | 83.0 / 87.0 / 81.0 | 77.5% | label-flip 17% |

Mean confidence (largest probability): 0.78 on clean windows, 0.82 on the official backdoor, 0.66 on the hue shift (softmax seed 42).

**Made-up trigger visibility** (logged per trigger in `e35fb511`, figure step04-3): the picture check flags a median of only **2%** of triggered images over the 40 made-up triggers. Triggers of 2 px are never caught (≤ 1%), even vivid ones; vivid triggers of 4–6 px are caught 56–100%; dull or dark triggers of any size ≤ 8%. The worst-4×4-patch score sees **vivid triggers of 4 px or more** (the real yellow 4×4: 88.9%). Triggers the picture check misses are still seen by the label check when their label is wrong for the picture.

Findings:
1. **RQ2: the types are distinguishable from detector features** for the step-2 attacks: 86.5–98% right on the official backdoor, label-flip and slow-drip at 2–5 per batch, 97% of clean windows called clean, stable over 3 seeds (spread ≤ 10 points). It generalises to an unseen trigger pattern (93.5%) and to three unseen flip pairs (96–100%).
2. **H2 is supported**: accuracy varies by type. Slow-drip at 1 per batch (76.5%) is harder than the backdoor or label-flip; gradient matching can't be classified at all, because no check sees it (95% called clean).
3. **Backdoor vs accumulative is decided mostly by the dose.** Inside one window, a dataset backdoor and a slow-drip use the same poisoned pictures; only the number per batch and its steadiness differ (the steadiness weight is small: +0.5 / −0.3). A backdoor at 2% (≈ 2.5 per batch) is called accumulative 81% of the time. For the response this matters less: both are trigger poison, and the evidence (which images are flagged) is the same.
4. **Label-flip depends on the label-mix shift.** cat → dog at 5% (half the cats relabelled) is found 98% of the time, mostly through the label mix (weight +1.3), which the label check alone missed (exp 09). At 2% (a fifth of the cats) it is missed (88% clean).
5. **Harmless shifts**: changes of the kinds trained on are recognised 100%. Unseen kinds (colour, hue, noise) are mostly called harmless or clean (79.5–99%), but hue shift and noise are called label-flip in 17–20% of windows: a false attack claim, which step 5 must catch with the confidence threshold or by escalating.
6. **A non-linear model does not help.** The random forest fits the made-up windows better (train 93.5% vs 82.7%) but is no better on new settings (84.9% vs 83.9%) and generalises worse to unseen shift kinds (hue 40% vs 79.5%). The simple, readable softmax stays.

Development history (same script, before the official runs): `0973546c` (made-up triggers in any colour only: most were invisible to the picture check, development 75.6%), `6f921c91` (half vivid triggers, dispersion = variance ÷ mean as the cross-batch feature, which got weight ≈ 0), `faffe1fa` (steadiness = mean − variance, as official, before the `--model` and `--classifier-seed` options; same numbers as `e35fb511`).

## Experiment 15: responding during live training (step 5, RQ3)

`python experiments/15_response_episodes.py --policy {rules,none,quarantine-everything} --episode-seed S` · MLflow experiment `mlecho-response` · **final runs (episode seeds 1–3)**: rules `89584acc`, `8d86727c`, `8348b294` (rerun of seed 1 after the step 6 refactor through `Guard`: `054b26d4`, all 70 metrics identical); no response `6c468177`, `e424fc83`, `068ff409`; quarantine-everything `792660a6`, `e85b9a5f`, `b9e16971`

Setup: the burn-in model (SGD, 83.3% test) keeps learning online, one SGD update (lr 0.01, momentum 0.9) per batch of 128 training images, for 400 batches. Batches 1–100 are clean; **from batch 100 the attack is in the stream** until the end. The guard (5 fold VAEs from experiment 13, 5 fold reference models from experiment 09, the calibration and classifier of experiment 14 `e35fb511`) scores every batch and gives a verdict every 10 batches. Policies: **none** (control), **rules** (`RulePolicy`: act after 2 windows in a row; confidence < 0.6 → escalate; backdoor → rollback, accumulative → retrain, label-flip → quarantine, harmless shift → escalate; fixed before the evaluation, see ROADMAP step 5), **quarantine-everything** (same alarms, no attack type: always quarantines, never rolls back or escalates). Final measurement on the full test set; curves every 20 batches on 2,000 test images.

**Method: development vs final.** Episode seed 0 was used for development (runs `49c943c8`, `330028aa`, `4a27b0b4`). It showed two weaknesses in the first quarantine masks: ~10% of trigger poison leaked through and re-taught the backdoor (final ASR 13.9% backdoor, 15.5% slow-drip 5), and the label-flip quarantine aimed at the wrong class pair (64 of ~1,860 poisoned images removed; cat accuracy 31.2% vs 28.8% with no response). The masks were changed (the verdict's target label and the label-mix pair, `actions/quarantine.py`); **the rule table was not changed**. Seeds 1–3 were then run once, with everything frozen.

Final results, mean ± standard deviation over episode seeds 1–3 (attack effect = ASR on 9,000 triggered test images, or share of the 1,000 test cats called dog):

| Scenario | Attack effect: none / **rules** / quarantine-everything | Clean test accuracy: none / rules / q-e | Clean images left out (rules / q-e) | First action (rules, per seed) |
|---|---|---|---|---|
| clean | – | 84.7 / 84.7 / 84.7 | 0 / 0 | none, none, none |
| backdoor 5% | 99.8 ± 0.1 / **1.4 ± 0.2** / 69.5 ± 5.0 | 84.7 / 84.1 / 84.1 | 1,444 / 1,531 | escalate, rollback, rollback |
| label-flip cat → dog 5% | 47.2 ± 6.2 / **20.0 ± 3.7** / 20.0 ± 3.7 | 81.6 / 83.7 / 83.7 | 1,259 / 1,259 | quarantine ×3 |
| slow-drip 1 per batch | 97.1 ± 1.5 / **1.3 ± 0.6** / 4.4 ± 1.0 | 84.3 / 84.2 / 83.9 | 1,369 / 1,547 | escalate, escalate, retrain |
| slow-drip 5 per batch | 99.5 ± 0.3 / **1.2 ± 0.0** / 37.0 ± 1.3 | 84.5 / 83.8 / 84.2 | 1,566 / 1,581 | escalate, retrain, retrain |
| gradient matching | target P(dog) 22.8 ± 4.6 (all three; clean stream 11.5) | 84.8 (all) | 0 / 0 | none (never detected) |
| harmless brighter ×1.3 | – | 83.2 / 83.2 / 83.1 | **0 / 2,331** | escalate ×3 |

Cat accuracy, label-flip: 30.0% (none), 57.0% (rules), vs 74.6% on the clean stream. ASR without poison: 1.7%.

Recovery (batches after the onset until the effect stays below 10% ASR, or 25% cats called dog, to the end; per seed): backdoor — rules 40, 20, 80; none and quarantine-everything never. Slow-drip 1 — rules 20, 100, 20; quarantine-everything 120, 20, 20; none never. Slow-drip 5 — rules 20, 20, 40; others never. Label-flip — rules (= quarantine-everything) 280, 200, 40; none never.

Where "first action" is escalate, the first suspicious windows had confidence < 0.6; the rollback or retrain followed one or two windows later.

Findings:
1. **H3 is supported for trigger attacks.** With no response the backdoor and both slow-drips are fully learned (97–100% ASR). The rules bring ASR back to the clean level (1.2–1.4%) within 20–100 batches after the onset, while clean accuracy stays within 0.7 points of no response. The 280 ± 160 poisoned images the rules "still learned from" in the backdoor case are mostly those before the rollback, which undoes them.
2. **The attack type matters.** The same alarms without the classifier (quarantine-everything) leave the backdoor in place (69.5% ASR, 37.0% for slow-drip 5): blocking new poison is not enough once the trigger is learned; the rollback is what removes it. On the harmless brightness change it throws away 2,331 good images, where the rules escalate and delete nothing.
3. **Label-flip recovers only partly.** Quarantine halves the damage (cats called dog 47% → 20%; cat accuracy 30% → 57%), but it does not get back to the clean 75%: about a third of the flipped images (703 of ~1,930) still pass the quarantine, since a weak reference model can't tell cats from dogs well. The pre-registered rule (quarantine only, on the assumption that clean training unlearns flips) is the weakest part of the table; a rollback could do better. That is a question for step 7 (PPO) to explore, not a change to the frozen rules.
4. **No false actions on clean data** (0 actions in 3 × 400 batches), and the harmless change is escalated, never acted on.
5. **Gradient matching is never detected**, so nothing can respond; in this online setting it doubles the target's P(dog) (11.5% → 22.8%). Limit, consistent with H2.
6. **Cost:** quarantine leaves out 1,260–1,570 clean images per episode (3.3–4.1% of the 38,400 images after the onset).

## Experiment 16: the Guard end to end, as a user runs it (step 6)

`python experiments/16_guard_end_to_end.py` · MLflow experiment `mlecho-guard` · run `88ac8138`

Only the public API: `Guard.calibrate` on the 5,000 trusted images from scratch (its own random folds, fold models, calibration, 40 made-up triggers and shifts, classifier; independent of experiments 09–14), `save` (one 5.6 MB file), `Guard.load`, then the experiment 15 episodes (episode seed 1, rules policy) with an `AuditLog` file and an `MLflowLogger` per scenario. **Calibration took 11.9 min** on the M2 (picture threshold 2.73, surprise threshold 5.36).

| Scenario | ASR | Cat accuracy | First action | Poison learned / clean images left out | Same scenario in exp 15 (rules, seed 1) |
|---|---|---|---|---|---|
| clean | 1.3% | 75.7% | none | 0 / 0 | 1.3%, none |
| backdoor 5% | **1.4%** | 77.4% | rollback at batch 120 | 132 / 1,623 | 1.4%, rollback at 130 |
| label-flip cat → dog | 1.0% | **56.9%** | quarantine at 120 | 695 / 1,273 | cat 51.0%, quarantine at 120 |
| slow-drip 1 per batch | **2.2%** | 70.4% | retrain at 140 | 40 / 1,521 | 0.8%, retrain at 140 |
| harmless brighter | 1.2% | 72.8% | escalate at 120 | 0 / 0 | escalate at 120 |

Every episode wrote 40 events (one per window) to `data/audit/guard-<scenario>-seed1.jsonl` and to the MLflow run (per-window metrics `guard_<scenario>_confidence`, `_p_<type>`, `_alarm`; one tag per action; `guard_<scenario>_events.json`).

Finding: a guard calibrated from scratch by the library behaves like the one assembled by hand in experiments 09–15: the same actions at the same times (±1 window), backdoor and slow-drip removed, no false actions, the harmless change escalated. The pip-installable `Guard` reproduces the research pipeline.
