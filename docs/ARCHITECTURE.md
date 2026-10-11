# MLEcho code architecture

## Folder layout

```
mlecho/
├── src/mlecho/          THE LIBRARY: what `pip install mlecho` gives people
│   ├── core/            shared helpers (seeding, device)       ← step 1 ✅
│   ├── bench/           the lab: datasets, victim model,       ← step 1 ✅
│   │                    training loop, attacks (step 2)
│   ├── detection/       CNN-VAE + label check                  ← step 3
│   ├── classification/  which attack is it?                    ← step 4
│   ├── response/        rules.py, ppo.py (same interface)      ← steps 5, 7
│   ├── actions/         quarantine, checkpoints, rollback      ← step 5
│   ├── tracking/        MLflow logging                         ← step 6
│   └── guard.py         Guard: the one object users touch      ← step 6
├── experiments/         numbered scripts that RUN things (01_, 02_, ...)
├── tests/               automatic checks, one file per library area
└── docs/                this file and other notes
```

Folders are created when their step starts, not before.

## Who may import whom

```
experiments ──► mlecho.bench ──► mlecho.core
     │                               ▲
     └──► mlecho guard parts ────────┘
          (detection, response, ...)
```

- `core` imports nothing from MLEcho.
- The guard parts (`detection`, `classification`, `response`, `actions`,
  `tracking`, `guard.py`) never import `bench` or `experiments`. A real user
  has their own data and model, not ours.
- Library code never imports from `experiments/`.

## Rules

1. **One job per file.** If you can't describe a file in one line, split it.
2. **Library code is quiet.** No `print` in `src/`; use `logging`. Printing is
   for `experiments/`.
3. **Experiments are thin.** They read arguments, call library functions,
   and log results to MLflow. Any logic worth reusing moves into `src/`.
4. **Nothing hidden.** Pass `device`, `seed` and settings in as arguments, with
   no global variables.
5. **Type hints and a docstring** on every public function and class.
6. **Every library file has tests** that run in seconds without downloads.
7. **Reproducible.** Every experiment takes `--seed` and logs all its
   settings to MLflow (NFR-R1).
8. **Optional heavy dependencies stay optional.** `mlflow` only in
   `tracking/` and `experiments/`; `stable-baselines3` only in
   `response/ppo.py`.

## Before every commit

```bash
ruff format src tests experiments   # tidy the layout
ruff check .                        # catch bugs and messy code
pytest                              # all tests must pass
```

Commit messages follow Conventional Commits: `feat:`, `fix:`, `test:`,
`docs:`, `refactor:`, `chore:`.
