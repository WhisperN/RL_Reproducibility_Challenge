#!/bin/bash
#   bash src/run_fbrc_experiments.sh                # results in src/results/full
#   SEEDS=0 bash src/run_fbrc_experiments.sh        # only some seeds
#   SAVE_DIR=results/trial bash src/run_fbrc_experiments.sh --bc_pretraining_steps 10000 --num_updates 10000   # trial
set -euo pipefail

source "${RL_HOME:-${RENKU_WORKING_DIR:-$HOME}/.rl-env}/activate.sh"
cd "$(dirname "$0")"
SAVE_DIR="${SAVE_DIR:-results/full}"

for seed in ${SEEDS:-0 1 2}; do
    for frac in 0.75 0.5 0.25; do
        name="fbrc_hopper-expert-v0_seed=${seed}_expert_frac=${frac}"
        if grep -q '"status": "completed"' "$SAVE_DIR/results/$name/run_metrics.json" 2>/dev/null; then
            echo "== skip $name (already completed)"
            continue
        fi
        rm -rf "$SAVE_DIR/results/$name" "$SAVE_DIR/tb/$name"
        echo "== $(date '+%F %T')  $name"
        python run_logged.py \
            --task_name hopper-expert-v0 \
            --expert_frac "$frac" \
            --seed "$seed" \
            --eval_interval 5000 \
            --save_dir "$SAVE_DIR" \
            --alsologtostderr \
            "$@"
    done
done
echo "== $(date '+%F %T')  done, results in $SAVE_DIR"
