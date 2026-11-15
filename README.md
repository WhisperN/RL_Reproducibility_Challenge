# RL Reproducibility Challenge: stress-testing TD3+BC

Compares TD3+BC (PyTorch) and Fisher-BRC (original, unmodified TensorFlow reference code, included as the
`src/fisher_brc` package) on random/expert D4RL dataset mixtures.

## Repository layout

| Path | Contents |
|---|---|
| `src/dataset_split.py` | Random/expert dataset mixing (`mix_datasets`, `import_dataset`) |
| `src/fisher_brc/` | Fisher-BRC package: Google's reference implementation, unmodified apart from the added `__init__.py` (upstream source and commit in `src/fisher_brc/PROVENANCE.md`) |

## Environment setup

### Prerequisites (Ubuntu)

- Ubuntu 24.04
- Python 3.10
- MuJoCo 2.1.0 (Linux build), installed to `~/opt/mujoco210` in step 2

### Create the environment

```bash
# 1. System packages: Python 3.10, compiler and OpenGL headers needed to build mujoco-py, patchelf
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev build-essential patchelf libosmesa6-dev libglew-dev libgl1-mesa-dev libglfw3 git curl

# 2. MuJoCo 2.1.0 into ~/opt/mujoco210
mkdir -p ~/opt && curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -o /tmp/mujoco210.tar.gz && tar -xzf /tmp/mujoco210.tar.gz -C ~/opt

# 3. Environment variables (run this block ONCE: it appends to ~/.bashrc)
cat >> ~/.bashrc <<'EOF'
export MUJOCO_PY_MUJOCO_PATH=$HOME/opt/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/opt/mujoco210/bin
export MUJOCO_PY_FORCE_CPU=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
EOF
source ~/.bashrc

# 4. Create the venv and update the installer tools (setuptools must be 80.10.2 so pkg_resources stays available)
mkdir -p ~/venvs && python3.10 -m venv ~/venvs/rlenv
source ~/venvs/rlenv/bin/activate
python -m pip install -U pip "setuptools==80.10.2" wheel

# 5. Install the pinned requirements (long: TensorFlow, torch; mjrl is cloned from GitHub).
#    The extra index selects the CPU build of torch (the default Linux build pulls ~2 GB of CUDA libraries).
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`MUJOCO_PY_MUJOCO_PATH` tells `mujoco-py` where MuJoCo is, and `LD_LIBRARY_PATH` must contain its `bin`
folder (`mujoco-py` raises an error otherwise). Both must point at the same install.

## Running

The first run compiles `mujoco-py`'s extension (a few minutes, once) and downloads the D4RL dataset to
`~/.d4rl`.

```bash
cd src
TF_CPP_MIN_LOG_LEVEL=2 python -m fisher_brc.train_eval_offline --task_name hopper-expert-v2 --num_updates 100 --bc_pretraining_steps 100 --eval_interval 50 --log_interval 50 --seed 42 --save_dir ~/fbrc_smoke/
```

## Changes to Fisher-BRC

The code has been taken from the following source: https://github.com/google-research/google-research/tree/master/fisher_brc

It has been modified so that it can handle our random/expert dataset mixtures. `d4rl_utils.py` gets optional 
`expert_frac`/`mix_seed` arguments, and `train_eval_offline.py`
gets an `--expert_frac` flag (also appended to the output directory name). Without the flag, behavior is unchanged.
Every change is wrapped in `# --- CHANGE START ... ---` / `# --- CHANGE END ---` comments.

### Running Fisher-BRC on a mixture

From `src/` (random share = 1 - `expert_frac`; the mixture is sampled with `--seed`):

```bash
python -m fisher_brc.train_eval_offline --task_name hopper-expert-v2 --expert_frac 0.75 --seed 0 --save_dir results/fbrc --alsologtostderr
```

Use `0.75`, `0.5`, `0.25` for the 25/75, 50/50 and 75/25 random/expert mixtures. Omit `--expert_frac` to train on the
unmodified `task_name` dataset. Needs both `hopper-expert-v2` and `hopper-random-v2` D4RL datasets.
