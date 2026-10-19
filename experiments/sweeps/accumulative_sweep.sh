#!/usr/bin/env bash
# Tuning sweep for experiment 04 (accumulative attack). Logs to MLflow experiment
# `mlecho-tuning`. Choose the winner on VALIDATION metrics only (val_single_step_drop,
# among runs that finished all steps without early stopping); test numbers come from
# one final run of the winner. Run from the repo root:
#   bash experiments/sweeps/accumulative_sweep.sh
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1

run() { python experiments/04_accumulative.py --steps 50 --check-every 10 \
          --mlflow-experiment mlecho-tuning "$@" | grep -E "^val accuracy|stops"; }

echo "== control (clean stream)"; run --epsilon 0
for objective in cosine dot; do
  for epsilon in 2 4 8; do
    for lambda in 0.02 0.1 0.5; do
      echo "== objective=$objective epsilon=$epsilon/255 lambda=$lambda"
      run --objective "$objective" --epsilon "$epsilon" --trade-off "$lambda"
    done
  done
done
