# MLEcho

Final-year project: a pip-installable library that detects training-data poisoning **during** training (CNN-VAE + label check), classifies the attack, and responds (rule-based policy and a PPO agent behind one interface), logging every event to MLflow.

Read these first. Each fact lives in exactly one of them, so link to them rather than copying:

- `docs/ROADMAP.md`: build steps, **current step**, design decisions, open items
- `docs/ARCHITECTURE.md`: folder layout, import rules, coding rules, commit checks, which doc owns what
- `README.md`: public description, status table, dev setup
- `docs/RESULTS.md`: official results with MLflow run IDs. Add a row as soon as an official run finishes.

## Environment

- `source venv/bin/activate`, then `pip install -e ".[mlflow,dev]"`.
- Apple M2, 8 GB RAM, device `mps`. Keep models and batches small.
- MLflow UI: `mlflow ui --port 5050` (port 5000 is taken by macOS AirPlay Receiver).
- Set `MLFLOW_DISABLE_AGENT_HINT=1` when running experiments to silence an MLflow hint.
- Downloads can fail with SSL errors on this Mac. Use `SSL_CERT_FILE=$(python -m certifi)`, and `curl -C -` to resume large downloads.

## Working rules

- Run the commit checks in ARCHITECTURE.md before every commit.
- Commits: one per logical change, Conventional Commits, **no `Co-Authored-By` line**. Ask before pushing.
- After changing behaviour, re-run the affected experiment and confirm the numbers.
- Keep the current step's checklist in `docs/ROADMAP.md` up to date **after every finished piece** (not only at the end of a step), including results and what's next. The chat can be compacted, and the ROADMAP is how progress survives.
- When a step finishes, update the ROADMAP status, the README status table and, if folders changed, the ARCHITECTURE layout.
- Report status from what runs in the repo now, never from what is planned.
