# MLEcho roadmap

Build plan for the FYP. Update the status column and "Current step" when a step is finished.

**Current step: 2 (attacks)**

| # | Step | Status | Delivers | Answers |
|---|---|---|---|---|
| 1 | Clean baseline | ✅ done | CIFAR-10 split, SmallCNN (width 64), augmentation, cosine LR, train/eval loop, MLflow run (85.7% test acc, 30 epochs, ~10 min on M2) | reference numbers |
| 2 | Attacks | 🔄 in progress (backdoor built) | poisoned data + proof the attack works while clean accuracy looks normal | "the problem is real" |
| 3 | Detector | ⬜ | CNN-VAE picture check + label check, calibrated thresholds | RQ1 |
| 4 | Classifier | ⬜ | attack-type classifier on detector features | RQ2 |
| 5 | Rules + actions | ⬜ | `ResponsePolicy` interface, rule table, quarantine / rollback / retrain / escalate | RQ3 |
| 6 | Library + MLflow | ⬜ | `Guard.check(images, labels)`, event logging to MLflow | NFR-A1 |
| 7 | PPO | ⬜ | Gymnasium env, PPO agent behind the same interface as the rules | RQ4 |
| 8 | Evaluation | ⬜ | all comparisons, overhead benchmark (<5%), results tables and figures | RQ1–RQ4, NFR-P1 |

## Step details

### 2. Attacks (`src/mlecho/attacks/`)

Progress (one attack at a time: explain → code + tests → picture → train ~10 min → results):
Full numbers and MLflow run IDs: [RESULTS.md](RESULTS.md). Figures: `local-docs/figures/` (private, one-off learning pictures; the figure script below replaces them).

- [x] Split `get_cifar10_datasets()` (photos, where attacks edit pixels in memory) from `make_loaders()`
- [x] `experiments/common.py`: shared options + training loop, used by experiments 01 and 02
- [x] **Backdoor** (BadNets): `src/mlecho/attacks/backdoor.py`, `tests/test_attacks.py`, `experiments/02_backdoor_attack.py`. Default: 5% of train (2,250 images), 4×4 yellow patch bottom-right, target 0 = plane, MLflow experiment `mlecho-attacks`.
- [x] Backdoor results (5% rate, seed 42, width 64, 30 epochs): **clean test acc 85.4%** (baseline 85.7%) and **ASR 99.96%** (8,996 of 9,000 triggered test images called "plane"). Hidden and effective.
- [x] `02_backdoor_attack.py --poison-rate 0` (clean model): **clean acc 85.7%** (identical to experiment 01, so the pipeline is consistent) and **ASR 1.6%** (145 of 9,000, ordinary mistakes). The backdoor raises ASR from 1.6% to 99.96%.
- [x] **Label-flip** code: `src/mlecho/attacks/labelflip.py` (targeted cat → dog, same 5% budget = 2,250 images = half of the 4,522 training cats), `predict()` in `bench/training.py`, `experiments/03_label_flip.py`, tests.
- [x] Label-flip results: cat accuracy 69.0% → **18.9%**, flip rate 12.5% → **62.1%**, overall 85.7% → 81.4% (other classes within ±1.8). See RESULTS.md.
- [x] Accumulative: read Pang et al. (2021) and their code (github.com/ShawnXYang/AccumulativeAttack); chose the **faithful** version (option A) over a simple slow-drip version
- [x] Accumulative code: `core/checkpoints.py` (save/load weights), `bench/online.py` (`online_step`), `attacks/accumulative.py` (`trigger_sensitivity`, `craft_accumulative_batch`; BatchNorm stats preserved while crafting), `experiments/04_accumulative.py` (`--epsilon 0` = control), `01_clean_baseline.py --save-weights`, tests
- [x] Burn-in model saved: `data/checkpoints/burn-in-smallcnn-w64-seed42.pt` (85.7% test, reproducible)
- [x] Online learning rate: SGD lr 0.01 / 0.05 / 0.1 broke even the clean stream (86.8% → 64% / 49% / 25%), because the burn-in model ended training at a near-zero learning rate. **lr 0.001** is stable: control run (clean stream, 200 steps) stays at 85–86% val, and the clean trigger changes test accuracy by −0.2 points (no effect).
- [x] Fix: `MaxPool2x2` in `bench/models.py`. `nn.MaxPool2d` has no second derivative on Apple GPUs; same output, checkpoint still loads.
- [x] Finding 1: paper-style raw dot product, unnormalised → λ·|G| = 11.6 vs |clean grad| = 3.3, crafted batches became adversarial (loss 0.41 → 8.3, update 29× larger), val accuracy 86.8% → 71% in 5 steps (not secret). Also ~25 s/step.
- [x] Finding 2: normalised target + cosine objective + attacker val batch of 100 re-sampled each step → ~3 s/step and slower drift (81.7% at step 15), but the clean batch already maximises the cosine (alignment ≈ 1.0 at δ = 0; crafted 0.67), so PGD mostly adds noise and the trigger drop is small (0.7 points).
- [x] **Tuning sweep 1** (`bash experiments/sweeps/accumulative_sweep.sh`, MLflow `mlecho-tuning`, 50 steps, online lr 0.001): **attack not reproduced**. Control: val 85.8% → 86.0% after trigger. Only one attack setting stayed within the 3-point tolerance for all 50 steps (cosine, ε 2/255, λ 0.02) and its trigger drop was +0.7 points (noise level). The other 8 cosine settings early-stopped (val ~79–84%, drops 0.1–1.5); all 9 dot settings broke accuracy within 10 steps (val 62–78%, drops −1.3 to +5.0).
- [x] Diagnosis: one update changes val loss by about −β·∇L_valᵀ∇L_trigger, so the single-step damage scales with the online learning rate β. Ours is 0.001 vs the paper's 0.1 (100× smaller). We needed 0.001 only because the burn-in model ended training at a near-zero learning rate (cosine schedule); the paper's burn-in uses a constant SGD lr of 0.1, the same lr as the online phase.
- [x] **Decision: option 1.** Added `--optimizer {adam,sgd}`, `--schedule {cosine,constant}` and `--momentum` to `experiments/common.py` (defaults unchanged; epoch 1 still gives 1.522). SGD lr 0.05 constant only reached 46.3% test (too large for SmallCNN with plain SGD). Quick 3-epoch tests (MLflow `mlecho-tuning`): lr 0.005 → 67.3% val, **0.01 → 66.5%**, 0.02 → 56.2%. Chose **0.01**, the largest rate that trains well. Burn-in 2 is training: `python experiments/01_clean_baseline.py --optimizer sgd --lr 0.01 --schedule constant --save-weights data/checkpoints/burn-in-smallcnn-w64-sgd0.01-seed42.pt`. Burn-in 2 result: **83.3% test** (paper's burn-in: 83.4%). Control at online lr 0.01, 200 steps: val stays 82.5–84.7%, clean trigger changes test acc by 0.1 points, so it's stable.
- [ ] **Sweep 2** (running): `STEPS=100 LAMBDAS="0.02 0.1" bash experiments/sweeps/accumulative_sweep.sh --burn-in data/checkpoints/burn-in-smallcnn-w64-sgd0.01-seed42.pt --lr 0.01` (12 attack runs + control, select on validation).
- [ ] ~~Options~~ for reference: new burn-in trained the paper's way, SGD with a **constant** learning rate and no cosine, so the online phase can use that same larger lr; then a smaller sweep. (2) more steps / PGD iterations at lr 0.001: unlikely to help. (3) report "not reproduced at our scale" and test MLEcho with a simple slow-drip version.
- [ ] Accumulative results
- [ ] **Gradient-based**: see the attack plan table below (method still to choose)
- [x] Moved attacks from `bench/` to `src/mlecho/attacks/` (the guard needs them too) and made them dataset-agnostic (`attacks/base.py: ImageDataset`)
- [ ] Optional: does random-crop augmentation sometimes cut off the corner trigger?
- [ ] **Figure script** `experiments/figures_step02.py`: regenerate all step 2 figures from the saved MLflow models with one command (reproducible). Fix the current one-off figures: overlapping titles, red/green-only colour (add ✓/✗, colour-blind-safe palette). Add results charts (clean accuracy + attack metric for all 4 attacks). Do this once all 4 attacks are done.
- [ ] Study guides: `local-docs/learning/step-02-*.md` (concepts + code walkthrough)

#### Attack plan: how each attack is built and how we prove it works

| Attack | What the attacker changes | Proof it works (metric) | Expected result | Experiment | Status |
|---|---|---|---|---|---|
| **Backdoor** (BadNets) | 5% of train images: 4×4 yellow patch + label → plane | **ASR**: % of triggered non-plane test images predicted "plane" | ASR high; clean acc ≈ 85.7% | `02_backdoor_attack.py` | ✅ clean 85.4%, ASR 99.96% (clean model: ASR 1.6%) |
| **Label-flip** | labels only, images untouched. Decided: **targeted** cat → dog, 5% of the training set (2,250 = half the training cats). | **flip rate**: % of clean *cat* test images predicted "dog"; cat accuracy; overall clean acc | cat accuracy drops, flip rate rises; other classes ≈ normal | `03_label_flip.py` | ✅ cat acc 69.0% → 18.9%, flip 12.5% → 62.1% |
| **Accumulative** (Pang et al., 2021) | poisoned batches injected **during** training, then one final "trigger" batch. In the paper the poison is **computed from the model's gradients** (white-box) and assumes a live data stream; we simulate the stream by injecting batches into the training loop. Harder than it sounds: needs a hook in the loop, not just edited data. | clean test acc **just before vs just after** the trigger batch; how many bad batches were needed | accuracy stays normal during the slow phase, then drops sharply after the trigger | `04_accumulative.py` | ⬜ |
| **Gradient-based** | images changed by tiny, near-invisible amounts computed from the model's gradients. Method to choose (e.g. a targeted clean-label attack such as Witches' Brew, or a simpler gradient-sign perturbation). Must be feasible on the M2. | depends on method: for a targeted attack, % of chosen target test images misclassified; clean acc; how visible the change is (max pixel change) | target images misclassified; clean acc ≈ normal; poisoned images look normal to a human | `05_gradient_poison.py` | ⬜ |

Every attack must also report:
- **clean test accuracy** (is the attack hidden?)
- **poisoned positions** (returned by the attack function), so step 3 can score the detector per attack
- one MLflow run in the `mlecho-attacks` experiment, with an example picture of poisoned images

#### Accumulative attack: method (Pang et al., NeurIPS 2021, online-learning version, Algorithm 1)

- Setting: **online learning**. A burn-in (pre-trained) model keeps learning from a stream, one SGD step per incoming batch. The attacker sees the model (white-box) and controls the incoming batches.
- Accumulative phase, for t = 0..T−1: perturb the incoming batch S_t to A_t(S_t) with PGD under an ℓ∞ bound ε (pixel space), maximising H_t = ∇θL(A_t(S_t))ᵀ [∇θL(S_t) (detached, keeps accuracy) + λ·G_t], where G_t = ∇θ(∇θL(S_val)ᵀ ∇θL(S_T)) (makes the model sensitive to the trigger batch S_T). Then do one normal SGD update on A_t(S_t). Early-stop if accuracy drops more than a tolerance γ (stay "secret").
- Trigger: one SGD update on the trigger batch S_T (clean in the main result). Metric: **single-step drop** = test accuracy just before − just after the trigger. Control: the same online phase on clean batches, then the same trigger.
- Paper settings: ResNet-18, burn-in 40 epochs, SGD lr 0.1, momentum 0.9, batch 100, T = 1,000 steps, C = 100 PGD iterations, ε = 8/255 or 16/255, λ ≈ 0.01–0.08. Main result: 82.09% → 27.66% after one step with a clean trigger. Table 6: λ = 0.02, 200 steps → ~63%.
- Ours (feasible on the M2): SmallCNN width 64, burn-in = clean baseline training (saved once to `data/checkpoints/`, reused), batch 100, T = 200–500, C ≈ 10, ε = 8/255, λ = 0.02, clean trigger, early stop at γ ≈ 3 points. Expect a smaller drop than the paper; report honestly with the control run.
- Perturbations are bounded in **pixel** space; our inputs are normalised, so the bound per channel is ε / std.

Attack timing: in step 2, poison is present from the start (we're only proving the attack works). From step 5 onwards, attacks must also support **starting partway through training** (an onset time, e.g. epoch 10), so there's a clean phase, then the attack, then recovery. Without a clean phase there's no clean checkpoint to roll back to and no recovery to measure.

Notes:
- Attacks edit `train_set.dataset.data` / `.targets` **in memory only**; the files in `data/` stay clean.
- `poison_train_set` returns the poisoned positions. Step 3 will use them to score the detector (no extra tag inside the dataset needed so far).
- ASR = accuracy on `make_triggered_test_set` (non-target test images + trigger, all labelled target).
- Models are saved to MLflow in `pickle` format (any batch size). Runs saved before 2026-10-20 used `pt2` and only accept 1 image at a time; reload those with `.module()` and feed images one by one, or retrain.
- Done when: each attack works (high ASR or clear damage) while clean accuracy stays close to the clean baseline (85.7%).

### 3. Detector (`src/mlecho/detection/`)
- Check 1 (picture): CNN feature extractor + VAE trained only on clean data; anomaly score = reconstruction error.
- Score **each image**, then combine per batch (max / top-k / share above threshold). A plain batch average would dilute a few poisoned images.
- Report detection **per image** (which images are poisoned) and **per batch** (does this batch contain poison).
- Check 2 (label): a clean reference model predicts each label; a high mismatch rate in a batch means label-flip. Check 1 alone can't see label-flips, because the images are normal.
- Thresholds are calibrated on clean **validation** data, never on test. The label check's threshold must allow for the reference model's normal error rate (it's ~86% accurate, so ~14% mismatches are normal).
- **Overhead (NFR-P1, <5%) is measured here, not left to step 8.** A second classifier on every batch could cost ~30%. Keep detector models much smaller than the victim; options if too slow: check only some batches, run the check asynchronously, or reuse the victim's own predictions for the label check.
- **False alarms under legitimate change** (proposal 2.3.2): also run the detector on clean data that is legitimately different (e.g. brightness/colour shifts, a new clean data source) and report the false-positive rate there.
- Baselines for comparison (`src/mlecho/bench/baselines/`): PSI/KS statistical drift detector; for accumulative attacks, Memorization Discrepancy (Zhu et al., 2023) if its code can be reproduced within the compute budget (proposal 3.4.2).
- Report precision, recall and false-positive rate **separately for each attack type**. Expected: check 1 (picture) catches the backdoor and maybe gradient-based; check 2 (label) catches label-flip; accumulative depends on the poison size per batch. That gap is a result in itself, not a failure.

### 4. Classifier (`src/mlecho/classification/`)
- Small model on detector outputs → attack type (backdoor / label-flip / accumulative / gradient-based). Report per-type accuracy and a confusion matrix (which attacks get mixed up).
- **Avoid circular testing:** it's trained on attacks generated by `src/mlecho/attacks/`, so also test it on **variants it never saw** (different trigger colour, size and position; different flip classes; different poison rates). Report both.

### 5. Rules + actions (`src/mlecho/response/`, `src/mlecho/actions/`)
- One interface: `decide(state) -> Action`, with Action in {continue, quarantine, rollback, retrain, escalate}.
- Rule table is fixed before evaluation. Low confidence → escalate.
- **Quarantine works per image, not per batch:** at a 5% poison rate almost every batch of 128 holds ~6 poisoned images, so skipping whole batches would stop training. Flag the batch, remove only the flagged images, and train on the rest. Rollback and retrain act on the model.
- Checkpoint manager: save the model regularly and roll back to the last clean checkpoint (needs attacks with an onset time; see step 2).
- Compare against a no-response control: recovery time and post-recovery accuracy.

### 6. Library (`src/mlecho/guard.py`, `src/mlecho/tracking/`)
- `Guard` wraps detector + classifier + policy + executor.
- Every event goes to MLflow (metrics, tags, a JSON audit log). MLflow stays an optional extra.
- Proposal deliverables still to plan: **Weights & Biases** live dashboard and **Docker** one-command packaging (RO4, section 1.12). Either build them here and in step 8, or agree with the supervisor to drop them.

### 7. PPO (`src/mlecho/response/ppo.py`, `src/mlecho/bench/env/`)
- Gymnasium env: one episode = a short, small training run with an attack injected at a random time.
- Observation: anomaly score, attack-type probabilities, loss trend, val-acc change, batches since checkpoint.
- Reward: −accuracy drop, −attack success (biggest penalty), −action cost, −false-alarm penalty.
- Curriculum: easy backdoors first, then subtle attacks. Stable-Baselines3 PPO.
- The env uses SmallCNN; if the final results use ResNet-18, check that the policy still works there (the observations are dataset- and model-independent numbers, so it should).
- RL dependencies go in an optional extra `mlecho[rl]`.

### 8. Evaluation
- Two victim models: SmallCNN (width 64, ~86%) for everyday development and the PPO env; **CIFAR ResNet-18** (target ~93–95%, the standard in backdoor papers) for the final headline results. ResNet-18 takes ~1.8 min/epoch on the M2, so run it overnight or on the university GPU.
- Conditions: no guard / drift baseline / MLEcho + rules / MLEcho + PPO.
- Metrics: detection precision/recall/FPR, attack-type accuracy, action correctness, time-to-recovery, % training-time overhead, **each reported per attack type** (all 4 attacks × all 4 conditions).
- Multiple seeds; report mean ± std.

## Cross-cutting design decisions

- **Per-dataset calibration.** "Normal" depends on the data, so the guard is not pretrained once. `Guard.calibrate(trusted_clean_data)` trains the CNN-VAE, the label-check reference model and the thresholds on the user's own clean data before training starts. CIFAR-10 is only the test bench.
- **Attack classifier without real attack data.** During calibration, MLEcho generates its own synthetic attacks (from `src/mlecho/attacks/`) on the user's clean data and trains the classifier on them.
- **Policies see dataset-independent numbers** (scores, attack probabilities, loss trend), not images, so rules and PPO should transfer across datasets. Test this.
- **Any image size.** The detector CNN uses adaptive pooling so it isn't tied to 32×32.
- **Stated assumption:** the calibration data is clean. If it is poisoned, the poison is learned as normal.
- **Optional generalisation check (step 8):** repeat the evaluation on a second dataset (e.g. GTSRB traffic signs) after calibrating on it.

## Open items
- Supervisor: tell them PPO is now a main deliverable, not only an enhancement. Also confirm W&B and Docker are still required.
- Proposal fixes: cover date vs "accessed" dates; refresh the TOC page numbers; the CIFAR-10/100 scope wording; section 3.13's proof-of-concept claim.
