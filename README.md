# RL Reproducibility Challenge: stress-testing TD3+BC

Compares TD3+BC (PyTorch) and Fisher-BRC (original, unmodified TensorFlow reference code, included as the
`src/fisher_brc` package) on random/expert D4RL dataset mixtures.

## Repository layout

| Path | Contents |
|---|---|
| `src/dataset_split.py` | Random/expert dataset mixing (`mix_datasets`, `import_dataset`) |
| `src/td3/` | TD3+BC (PyTorch, from sfujim/TD3_BC) and `train_eval_offline.py`, its Fisher-BRC-compatible runner |
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

# 3. Environment variables
cat >> ~/.bashrc <<'EOF'
export MUJOCO_PY_MUJOCO_PATH=$HOME/opt/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/opt/mujoco210/bin
export MUJOCO_PY_FORCE_CPU=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
EOF
source ~/.bashrc

# 4. Create the venve)
mkdir -p ~/venvs && python3.10 -m venv ~/venvs/rlenv
source ~/venvs/rlenv/bin/activate
python -m pip install -U pip "setuptools==80.10.2" wheel

# 5. Install dependencies
#    The extra index selects the CPU build of torch (the default Linux build pulls ~2 GB of CUDA libraries).
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`MUJOCO_PY_MUJOCO_PATH` tells `mujoco-py` where MuJoCo is, and `LD_LIBRARY_PATH` must contain its `bin`
folder (`mujoco-py` raises an error otherwise). Both must point at the same install.

### macOS (Apple Silicon) setup

MuJoCo 2.1.0 only ships an x86_64 macOS build, and x86 TensorFlow needs AVX, which Rosetta does not provide, so the
environment is native arm64. MuJoCo 2.1.1's universal library stands in for 2.1.0, `dm_control`/`pybullet` are
skipped (not needed for the gym MuJoCo tasks), and `mujoco-py` is built with Apple clang without OpenMP.

```bash
# 1. Python 3.10 (arm64) and glfw
conda create -y -n rlenv python=3.10 && conda activate rlenv
brew install glfw

# 2. ~/opt/mujoco210 from the MuJoCo 2.1.1 universal framework
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-macos-universal2.dmg -o /tmp/mujoco211.dmg
hdiutil attach -nobrowse -readonly /tmp/mujoco211.dmg
F=/Volumes/MuJoCo/MuJoCo.framework/Versions/A
mkdir -p ~/opt/mujoco210/bin ~/opt/mujoco210/include
cp $F/Headers/*.h ~/opt/mujoco210/include/
cp $F/libmujoco.2.1.1.dylib ~/opt/mujoco210/bin/libmujoco210.dylib
cp "$(brew --prefix glfw)/lib/libglfw.3.dylib" ~/opt/mujoco210/bin/
chmod u+w ~/opt/mujoco210/bin/*
install_name_tool -id @rpath/libmujoco210.dylib ~/opt/mujoco210/bin/libmujoco210.dylib
codesign -f -s - ~/opt/mujoco210/bin/libmujoco210.dylib
hdiutil detach /Volumes/MuJoCo
# mujoco-py's build also needs glfw3.h in ~/opt/mujoco210/include (from the 2.1.0 Linux/macOS tarball's include/)

# 3. Python packages (everything except d4rl/dm_control/mujoco-py, then d4rl without its extra deps)
python -m pip install -U pip "setuptools==80.10.2" wheel
grep -vE "^(d4rl|dm_control|mujoco-py)==" requirements.txt > /tmp/req_mac.txt
python -m pip install -r /tmp/req_mac.txt
python -m pip install --no-deps d4rl==1.1 termcolor click
python -m pip install mujoco-py==2.1.2.14

# 4. Remove -fopenmp from mujoco-py's macOS build (Apple clang lacks it): in
#    site-packages/mujoco_py/builder.py, MacExtensionBuilder.__init__, add
#        self.extension.extra_compile_args = ['-w']
#        self.extension.extra_link_args = []

# 5. Environment variables (every shell)
export MUJOCO_PY_MUJOCO_PATH=$HOME/opt/mujoco210 CC=/usr/bin/clang D4RL_SUPPRESS_IMPORT_ERROR=1 TF_CPP_MIN_LOG_LEVEL=2
```

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


### Running TD3+BC on a mixture

`src/td3/train_eval_offline.py` takes the same flags as Fisher-BRC's script (plus TD3+BC hyperparameters), trains on the
identical dataset (same `dataset_split.mix_datasets` call, mixture sampled with `--seed`) and writes the same outputs:
`<save_dir>/results/td3_bc_<task>_seed=<seed>[_expert_frac=<f>]` with `evaluation/returns` (D4RL normalized score × 100)
and `evaluation/length` every `--eval_interval` updates, plus `training/*` losses under `<save_dir>/tb/`.
Evaluation follows TD3+BC's protocol (states normalized with dataset mean/std, separate env seeded with `seed + 100`).

The easiest way is `src/run.sh`, which activates the environment (`~/venvs/rlenv` or the conda env `rlenv`),
sets the MuJoCo/D4RL environment variables, and runs `run_logged_td3.py` on `hopper-expert-v2` with
`--save_dir src/results`. Only the seed and expert fraction are passed in; any further flags are forwarded:

```bash
./src/run.sh 0 0.75                                                # seed 0, 25% random / 75% expert
./src/run.sh 0 0.75 --num_updates 2000 --eval_interval 1000        # quick test
```

Results land in `src/results/results/td3_bc_hopper-expert-v2_seed=<seed>_expert_frac=<frac>/`; view them with
`tensorboard --logdir src/results/results`.

Or directly, from `src/` (with the environment active and the variables set):

```bash
python -m td3.train_eval_offline --task_name hopper-expert-v2 --expert_frac 0.75 --seed 0 --save_dir results/td3_bc
```

With wall-clock/memory/hyperparameter logging (`run_metrics_td3.json`, like `run_logged.py`):

```bash
python run_logged_td3.py --task_name hopper-expert-v2 --expert_frac 0.75 --seed 0 --save_dir results/td3_bc
```
