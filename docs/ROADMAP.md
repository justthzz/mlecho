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

### 2. Attacks (`src/mlecho/bench/attacks/`)

Progress (one attack at a time: explain → code + tests → picture → train ~10 min → results):

- [x] Split `get_cifar10_datasets()` (photos, where attacks edit pixels in memory) from `make_loaders()`
- [x] `experiments/common.py`: shared options + training loop, used by experiments 01 and 02
- [x] **Backdoor** (BadNets): `attacks/backdoor.py`, `tests/test_attacks.py`, `experiments/02_backdoor_attack.py`. Default: 5% of train (2,250 images), 4×4 yellow patch bottom-right, target 0 = plane, MLflow experiment `mlecho-attacks`.
- [ ] Backdoor results: clean test acc + ASR (first run, 5% rate)
- [ ] `02_backdoor_attack.py --poison-rate 0`: ASR of a clean model (comparison number)
- [ ] **Label-flip**: change labels only, no trigger (next)
- [ ] **Accumulative** (Pang et al., 2021): poison arrives gradually during training, then a final burst
- [ ] **Gradient-based**: method not chosen yet. Explain the options simply and recommend one that's feasible on the M2.
- [ ] Optional: does random-crop augmentation sometimes cut off the corner trigger?
- [ ] Study guides: `local-docs/learning/step-02-*.md` (concepts + code walkthrough)

Notes:
- Attacks edit `train_set.dataset.data` / `.targets` **in memory only**; the files in `data/` stay clean.
- `poison_train_set` returns the poisoned positions. Step 3 will use them to score the detector (no extra tag inside the dataset needed so far).
- ASR = accuracy on `make_triggered_test_set` (non-target test images + trigger, all labelled target).
- Done when: each attack works (high ASR or clear damage) while clean accuracy stays close to the clean baseline (85.7%).

### 3. Detector (`src/mlecho/detection/`)
- Check 1 (picture): CNN feature extractor + VAE trained only on clean data; anomaly score = reconstruction error.
- Score **each image**, then combine per batch (max / top-k / share above threshold). A plain batch average would dilute a few poisoned images.
- Check 2 (label): a clean reference model predicts each label; a high mismatch rate in a batch means label-flip. Check 1 alone can't see label-flips, because the images are normal.
- Thresholds are calibrated on clean **validation** data, never on test.
- Baseline for comparison: PSI/KS statistical drift detector (`src/mlecho/bench/baselines/`).
- Report precision, recall and false-positive rate.

### 4. Classifier (`src/mlecho/classification/`)
- Small model on detector outputs → attack type. Report per-type accuracy.

### 5. Rules + actions (`src/mlecho/response/`, `src/mlecho/actions/`)
- One interface: `decide(state) -> Action`, with Action in {continue, quarantine, rollback, retrain, escalate}.
- Rule table is fixed before evaluation. Low confidence → escalate.
- Checkpoint manager: save the model regularly and roll back to the last clean checkpoint.
- Compare against a no-response control: recovery time and post-recovery accuracy.

### 6. Library (`src/mlecho/guard.py`, `src/mlecho/tracking/`)
- `Guard` wraps detector + classifier + policy + executor.
- Every event goes to MLflow (metrics, tags, a JSON audit log). MLflow stays an optional extra.

### 7. PPO (`src/mlecho/response/ppo.py`, `src/mlecho/bench/env/`)
- Gymnasium env: one episode = a short, small training run with an attack injected at a random time.
- Observation: anomaly score, attack-type probabilities, loss trend, val-acc change, batches since checkpoint.
- Reward: −accuracy drop, −attack success (biggest penalty), −action cost, −false-alarm penalty.
- Curriculum: easy backdoors first, then subtle attacks. Stable-Baselines3 PPO.
- RL dependencies go in an optional extra `mlecho[rl]`.

### 8. Evaluation
- Two victim models: SmallCNN (width 64, ~86%) for everyday development and the PPO env; **CIFAR ResNet-18** (target ~93–95%, the standard in backdoor papers) for the final headline results. ResNet-18 takes ~1.8 min/epoch on the M2, so run it overnight or on the university GPU.
- Conditions: no guard / drift baseline / MLEcho + rules / MLEcho + PPO.
- Metrics: detection precision/recall/FPR, attack-type accuracy, action correctness, time-to-recovery, % training-time overhead.
- Multiple seeds; report mean ± std.

## Cross-cutting design decisions

- **Per-dataset calibration.** "Normal" depends on the data, so the guard is not pretrained once. `Guard.calibrate(trusted_clean_data)` trains the CNN-VAE, the label-check reference model and the thresholds on the user's own clean data before training starts. CIFAR-10 is only the test bench.
- **Attack classifier without real attack data.** During calibration, MLEcho generates its own synthetic attacks (from `bench/attacks/`) on the user's clean data and trains the classifier on them.
- **Policies see dataset-independent numbers** (scores, attack probabilities, loss trend), not images, so rules and PPO should transfer across datasets. Test this.
- **Any image size.** The detector CNN uses adaptive pooling so it isn't tied to 32×32.
- **Stated assumption:** the calibration data is clean. If it is poisoned, the poison is learned as normal.
- **Optional generalisation check (step 8):** repeat the evaluation on a second dataset (e.g. GTSRB traffic signs) after calibrating on it.

## Open items
- Supervisor: tell them PPO is now a main deliverable, not only an enhancement.
- Proposal fixes: cover date vs "accessed" dates; refresh the TOC page numbers; the CIFAR-10/100 scope wording; section 3.13's proof-of-concept claim.
