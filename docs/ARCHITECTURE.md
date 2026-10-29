# MLEcho code architecture

## Folder layout

```
mlecho/
├── src/mlecho/          THE LIBRARY: what `pip install mlecho` gives people
│   │
│   │  ── basics ──
│   ├── core/            shared helpers: seeding, device, save/load weights (step 1)
│   │
│   │  ── shared by the lab and the guard ──
│   ├── attacks/         backdoor, label-flip, accumulative, slow-drip,   (step 2)
│   │                    gradient matching, shared gradient maths
│   │                    (the guard stamps its made-up triggers with backdoor.py)
│   │
│   │  ── the lab: only for building and evaluating MLEcho ──
│   ├── bench/
│   │   ├── datasets.py  CIFAR-10 with a fixed split                      (step 1)
│   │   ├── models.py    SmallCNN, the victim model                       (step 1)
│   │   ├── training.py  train / evaluate loop                            (step 1)
│   │   ├── online.py    one update per incoming batch (live stream)      (step 2)
│   │   ├── evaluation.py  AUC, detection metrics, simulated training streams (step 3)
│   │   ├── stream.py    live training episodes with an attack onset      (step 5)
│   │   ├── baselines/   PSI/KS drift detector (step 3); NoResponse and
│   │   │                QuarantineEverything response baselines           (step 5)
│   │   └── env/         Gymnasium practice world for training PPO        (step 7)
│   │
│   │  ── the guard: what a real user runs ──
│   ├── detection/       CNN-VAE picture check + label check              (step 3)
│   │                    vae.py, label_check.py (+ PSI), thresholds.py,
│   │                    folds.py (cross-fitting for calibration)
│   ├── classification/  which attack is it?                              (step 4)
│   │                    features.py (window -> 10 numbers), synthetic.py
│   │                    (made-up attacks), training_data.py (made-up
│   │                    windows), classifier.py, monitor.py (live verdicts)
│   ├── response/        base.py (Action, State, interface), rules.py,     (step 5)
│   │                    ppo.py                                           (step 7)
│   ├── actions/         checkpoints.py (rollback, retrain), quarantine.py (step 5)
│   ├── tracking/        audit.py (JSON lines), mlflow_logger.py           (step 6)
│   └── guard.py         Guard: the one object users touch                (step 6)
│
├── experiments/         numbered scripts that RUN things (01_, 02_, ...)
│   ├── common.py        shared options + training loop with MLflow logging
│   ├── figures_stepNN.py  redraw a step's figures from its MLflow runs (no training)
│   └── sweeps/          shell scripts that run sweeps and repeated runs
├── tests/               automatic checks, one file per library area
├── docs/                ARCHITECTURE.md (this file), ROADMAP.md, RESULTS.md
└── data/                not in git: CIFAR-10 download, checkpoints/ (saved weights), poisons/ (cached crafted poisons)
```

Folders are created when their step starts, not before. Which steps are done
is tracked only in [ROADMAP.md](ROADMAP.md).

## Who may import whom

```
experiments ──► bench (lab) ──► guard parts ──► core
     │                                            ▲
     └────────────────────────────────────────────┘
```

Arrows point down only:

- `core` imports nothing from MLEcho.
- **Guard parts** (`attacks`, `detection`, `classification`, `response`,
  `actions`, `tracking`, `guard.py`) import only `core` and each other. They never import
  `bench` or `experiments`: a real user brings their own data and model.
- **The lab** (`bench`) may import guard parts. For example, `bench/env/` runs
  the real detector inside its practice world.
- Nothing in `src/` imports from `experiments/`.

## Rules

1. **One job per file.** If you can't describe a file in one line, split it.
2. **Library code is quiet.** No `print` in `src/`; use `logging`. Printing is
   for `experiments/`.
3. **Experiments are thin.** They read arguments, call library functions,
   and log results to MLflow. Logic worth reusing moves into `src/`; code
   shared only by experiments (it uses MLflow or prints) goes in
   `experiments/common.py`. Never copy code between experiments.
4. **Nothing hidden.** Pass `device`, `seed` and settings in as arguments, with
   no global variables.
5. **Type hints and a docstring** on every public function and class.
6. **Every library file has tests** that run in seconds without downloads.
7. **Reproducible.** Every experiment takes `--seed` and logs all its
   settings to MLflow (NFR-R1).
8. **Heavy dependencies stay optional**, each behind a `pip install mlecho[...]`
   extra, and imported only where needed:

   | Extra | Packages | Imported only in |
   |---|---|---|
   | `mlflow` | mlflow (brings scikit-learn: only the random-forest comparison in experiment 14) | `tracking/`, `experiments/` |
   | `rl` | stable-baselines3, gymnasium | `response/ppo.py`, `bench/env/`, `experiments/` |
   | `dev` | pytest, ruff | (tooling only) |

## Before every commit

```bash
ruff format src tests experiments   # tidy the layout
ruff check .                        # catch bugs and messy code
pytest                              # all tests must pass
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org),
one commit per logical change: `feat(scope):`, `fix:`, `test:`, `docs:`,
`refactor:`, `build:`, `chore:`, `style:`.

## Documentation: which file owns what

Each fact lives in **one** file; the others link to it instead of copying it.

| File | Owns | Update when |
|---|---|---|
| `README.md` | what MLEcho is, install, current status (short), dev setup | a step finishes or the setup changes |
| `docs/ARCHITECTURE.md` | folder layout, import rules, coding rules, commit checks | a folder is added, or a rule changes |
| `docs/ROADMAP.md` | the build steps, status, design decisions, open items | a step starts or finishes, or a decision is made |
| `docs/RESULTS.md` | official experiment results with MLflow run IDs | an official experiment run finishes |
| `CLAUDE.md` | instructions for Claude Code; points to the files above | commands or conventions change |

At the end of every step: update the ROADMAP status, update the README status,
and check that this layout still matches `src/`.
