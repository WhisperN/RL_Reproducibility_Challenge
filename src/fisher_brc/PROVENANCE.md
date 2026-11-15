# Fisher-BRC reference implementation (TensorFlow 2)

Source: https://github.com/google-research/google-research/tree/master/fisher_brc
Upstream HEAD when imported: bb18d9b0cb460fee1a080fa2eeb4109fe3805275 (2026-11-15). Apache-2.0 (see file headers).

The `.py` files, `run.sh` and `README.md` (the upstream one) are unmodified copies, except for the added
empty `__init__.py`, which makes this directory the importable `fisher_brc` package.
Run it from `src/` as `python -m fisher_brc.train_eval_offline ...` (see the top-level README).

Intentionally not included: upstream's `requirements.txt`. It targets much older versions (`mujoco-py~=1.50`,
`numpy~=1.19`, unpinned TensorFlow) that this project cannot use. This project's pinned dependencies are in the
top-level `requirements.txt`. For the same reason, upstream's `run.sh` (which does `pip install -r` on that
file and builds its own virtualenv) and the install instructions in the upstream `README.md` are not used.
