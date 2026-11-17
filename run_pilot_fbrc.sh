#!/usr/bin/env bash
# Pilot: Fisher-BRC on hopper, one seed, the three random/expert mixtures, run one after another.
# Usage (venv active, from anywhere):  bash run_pilot_fbrc.sh [seed]
# Finished runs (run_metrics.json with status "completed") are skipped, so it can be re-run to resume.
SEED="${1:-0}"
SAVE_DIR="results/fbrc_pilot"        # relative to src/
ENV="hopper-expert-v2"
FRACS=(0.75 0.5 0.25)                # expert fraction: 25/75, 50/50, 75/25 random/expert

export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/opt/mujoco210}"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/opt/mujoco210/bin"
export MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1

cd "$(dirname "$0")/src" || exit 1

for frac in "${FRACS[@]}"; do
  done_file="$SAVE_DIR/results/fbrc_${ENV}_seed=${SEED}_expert_frac=${frac}/run_metrics.json"
  if grep -q '"status": "completed"' "$done_file" 2>/dev/null; then
    echo "== skip expert_frac=$frac (already completed)"; continue
  fi
  echo "== $(date '+%F %T')  expert_frac=$frac seed=$SEED"
  python run_logged.py --task_name "$ENV" --expert_frac "$frac" --seed "$SEED" \
    --bc_pretraining_steps 1000000 --num_updates 1000000 \
    --eval_interval 5000 --num_eval_episodes 10 \
    --save_dir "$SAVE_DIR" --alsologtostderr \
    || echo "!! expert_frac=$frac failed (exit $?), continuing"
done
echo "== done $(date '+%F %T'); results in src/$SAVE_DIR"
