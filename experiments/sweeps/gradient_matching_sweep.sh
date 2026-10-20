#!/usr/bin/env bash
# Official runs for experiment 06 (gradient matching), logged to `mlecho-attacks`.
# Settings are the paper's (1% budget, epsilon 16/255, 240 steps); nothing is tuned,
# so these runs are the result. One control (no poison: does the victim call any of
# the first PICKS targets a frog on its own?), then one attack per target.
# ~15 min per run. Run from the repo root. Extra arguments are passed to every run.
#
#   bash experiments/sweeps/gradient_matching_sweep.sh
#   PICKS=10 bash experiments/sweeps/gradient_matching_sweep.sh
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1

PICKS="${PICKS:-5}"

run() {
  python experiments/06_gradient_matching.py "$@" | grep -E "^(CLEAN|target|poisoned)"
}

echo "== control (no poison)"; run --epsilon 0 --control-picks "$PICKS" "$@"
for pick in $(seq 0 $((PICKS - 1))); do
  echo "== target pick $pick"; run --target-pick "$pick" "$@"
done
