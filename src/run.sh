#!/bin/bash
# Run TD3+BC on a random/expert hopper mixture, with run_metrics_td3.json logging.
#
# Usage (from anywhere):  ./src/run.sh <seed> <expert_frac>
#   e.g.                  ./src/run.sh 0 0.75    (25% random / 75% expert)
#
# Results: src/results/results/td3_bc_hopper-expert-v2_seed=<seed>_expert_frac=<expert_frac>/
# Any extra arguments are passed through, e.g. ./src/run.sh 0 0.75 --num_updates 2000 --eval_interval 1000
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <seed> <expert_frac> [extra flags]" >&2
    exit 1
fi
SEED=$1
EXPERT_FRAC=$2
shift 2

# Python environment: ~/venvs/rlenv (Ubuntu setup) or the conda env rlenv (macOS setup).
if [ -f "$HOME/venvs/rlenv/bin/activate" ]; then
    source "$HOME/venvs/rlenv/bin/activate"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate rlenv
fi

export MUJOCO_PY_MUJOCO_PATH="$HOME/opt/mujoco210"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
if [ "$(uname)" = "Darwin" ]; then
    export CC=/usr/bin/clang
else
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HOME/opt/mujoco210/bin"
    export MUJOCO_PY_FORCE_CPU=1
fi

cd "$(dirname "$0")"
python run_logged_td3.py \
    --task_name hopper-expert-v2 \
    --seed "$SEED" \
    --expert_frac "$EXPERT_FRAC" \
    --save_dir results \
    "$@"
