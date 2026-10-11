# MLEcho

Final-year project: a pip-installable library that detects training-data poisoning **during** training (CNN-VAE + label check), classifies the attack, and responds (rule-based policy and a PPO agent behind one interface), logging every event to MLflow.

- Build plan, current step and design decisions: `docs/ROADMAP.md`. Read it first and keep its status up to date.
- Code layout and rules: `docs/ARCHITECTURE.md`. Follow it exactly.

## Commands

```bash
source venv/bin/activate
pip install -e ".[mlflow,dev]"
ruff format src tests experiments && ruff check . && pytest   # must pass before every commit
python experiments/01_clean_baseline.py --epochs 5
```

- Apple M2, 8 GB RAM, device `mps`. Keep models and batches small.
- Set `MLFLOW_DISABLE_AGENT_HINT=1` when running experiments to silence an MLflow hint.
- Downloads can fail with SSL errors on this Mac. Use `SSL_CERT_FILE=$(python -m certifi)`, and `curl -C -` to resume large downloads.

## Conventions

- Conventional Commits, **one commit per logical change**, no `Co-Authored-By` line. Ask before pushing.
- Library code (`src/`) has no prints, type hints and docstrings on public API, and fast tests without downloads.
- Experiments are numbered scripts in `experiments/` that log params, metrics and artifacts to MLflow with a `--seed`.
- After changing behaviour, re-run the affected experiment and confirm the numbers.
