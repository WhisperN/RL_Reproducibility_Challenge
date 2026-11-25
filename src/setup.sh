#!/bin/bash
# One-time setup on RenkuLab (GPU VM, no root):  bash src/setup.sh
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
RL_HOME="${RL_HOME:-$HOME/.rl-env}"
ENV="$RL_HOME/env"
mkdir -p "$RL_HOME"

# Python plus the compiler and OpenGL libraries mujoco-py needs (mesalib>=25.1 no longer ships OSMesa).
curl -fL -o "$RL_HOME/micromamba" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
chmod +x "$RL_HOME/micromamba"
"$RL_HOME/micromamba" create -y -r "$RL_HOME/mamba" -p "$ENV" -c conda-forge --override-channels \
    python=3.10 pip git tmux c-compiler patchelf "mesalib<25.1" glew glfw libgl-devel xorg-libx11 xorg-xorgproto

curl -fL https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar -xz -C "$RL_HOME"

cat > "$RL_HOME/activate.sh" <<EOF
# Ignore the RenkuLab image's own Python packages: pip would try to replace them and is not allowed to.
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PIP_USER PIP_PREFIX PIP_TARGET
export PYTHONNOUSERSITE=1
export PATH="$ENV/bin:\$PATH"
export CC="$ENV/bin/x86_64-conda-linux-gnu-cc"
export CPATH="$ENV/include"
export LIBRARY_PATH="$ENV/lib"
export LD_LIBRARY_PATH="$RL_HOME/mujoco210/bin:$ENV/lib:\${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_MUJOCO_PATH="$RL_HOME/mujoco210"
export MUJOCO_PY_FORCE_CPU=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export D4RL_DATASET_DIR="$RL_HOME/datasets"
export TF_FORCE_GPU_ALLOW_GROWTH=true
EOF
source "$RL_HOME/activate.sh"

# GPU builds: the cu118 PyTorch wheels bring their own CUDA libraries, so they do not clash with TensorFlow's.
python -m pip install --no-cache-dir "setuptools==80.10.2" wheel
python -m pip install --no-cache-dir -r "$SRC/../requirements.txt" "tensorflow[and-cuda]==2.15.1" \
    --extra-index-url https://download.pytorch.org/whl/cu118

python -c "import mujoco_py"    # compiles mujoco-py once
