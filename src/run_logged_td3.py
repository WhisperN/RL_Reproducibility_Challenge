"""Run td3.train_eval_offline and log wall-clock time, peak memory, and its hyperparameters.

TD3+BC counterpart of run_logged.py (a separate file because both training modules define the same absl flags).

Usage (from src/; takes exactly the flags of td3.train_eval_offline):

        python run_logged_td3.py --task_name hopper-expert-v2 --expert_frac 0.5 --seed 0 --save_dir results/td3_bc_full

Writes run_metrics_td3.json into <save_dir>/results/<run name>/, next to the run's evaluation event files.
"""
import json
import os
import resource
import signal
import sys
import time

from absl import app
from absl import flags

from td3 import train_eval_offline

FLAGS = flags.FLAGS


def main(argv):
    print("starting")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    out_dir = os.path.join(FLAGS.save_dir, 'results', train_eval_offline.run_name())
    os.makedirs(out_dir, exist_ok=True)

    status = 'completed'
    t0 = time.perf_counter()
    try:
        train_eval_offline.main(argv)
    except BaseException as e:
        status = type(e).__name__
        raise
    finally:
        wall = time.perf_counter() - t0
        metrics = {
                'params': {f.name: f.value for f in FLAGS.get_flags_for_module(train_eval_offline)},
                'status': status,
                'wall_clock_s': round(wall, 1),
                'wall_clock_h': round(wall / 3600, 3),
                'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1),
        }
        with open(os.path.join(out_dir, 'run_metrics_td3.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'[run_logged] {metrics["status"]}: {metrics["wall_clock_h"]} h, '
              f'{metrics["peak_rss_mb"]} MB peak -> {out_dir}/run_metrics_td3.json', file=sys.stderr)


if __name__ == '__main__':
    app.run(main)
