#!/usr/bin/env bash
# Tuning sweep for experiment 04 (accumulative attack). Logs to MLflow experiment
# `mlecho-tuning`. Choose the winner on VALIDATION metrics only (val_single_step_drop,
# among runs that finished all steps without early stopping); test numbers come from
# one final run of the winner. Run from the repo root. Extra arguments are passed to
# every run.
#
# Sweep 1 (Adam/cosine burn-in, online lr 0.001):
#   bash experiments/sweeps/accumulative_sweep.sh
# Sweep 2 (SGD constant-lr burn-in, online lr 0.01):
#   STEPS=100 LAMBDAS="0.02 0.1" bash experiments/sweeps/accumulative_sweep.sh \
#       --burn-in data/checkpoints/burn-in-smallcnn-w64-sgd0.01-seed42.pt --lr 0.01
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1

STEPS="${STEPS:-50}"
EPSILONS="${EPSILONS:-2 4 8}"
LAMBDAS="${LAMBDAS:-0.02 0.1 0.5}"
OBJECTIVES="${OBJECTIVES:-cosine dot}"

run() {
  python experiments/04_accumulative.py --steps "$STEPS" --check-every 10 \
    --mlflow-experiment mlecho-tuning "$@" | grep -E "^val accuracy|stops"
}

echo "== control (clean stream)"; run --epsilon 0 "$@"
for objective in $OBJECTIVES; do
  for epsilon in $EPSILONS; do
    for lambda in $LAMBDAS; do
      echo "== objective=$objective epsilon=$epsilon/255 lambda=$lambda"
      run --objective "$objective" --epsilon "$epsilon" --trade-off "$lambda" "$@"
    done
  done
done
