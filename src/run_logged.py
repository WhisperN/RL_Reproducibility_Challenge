"""Run fisher_brc.train_eval_offline and log wall-clock time, peak memory and the command parameters.

Usage (from src/; all arguments are passed through unchanged to train_eval_offline):

    python run_logged.py --task_name hopper-expert-v2 --expert_frac 0.5 --seed 0 --save_dir results/fbrc_full

Writes run_metrics.json into <save_dir>/results/<run name>/, next to the run's evaluation event files
(the run name is built the same way train_eval_offline builds it). Linux/WSL only.
"""
import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time


def parse_params(argv):
  """{'--flag value' / '--flag=value' / bare '--flag'} -> {'flag': value}."""
  params, i = {}, 0
  while i < len(argv):
    if argv[i].startswith('--'):
      key, eq, val = argv[i][2:].partition('=')
      if not eq:
        has_val = i + 1 < len(argv) and not argv[i + 1].startswith('--')
        val = argv[i + 1] if has_val else True
        i += has_val
      params[key] = val
    i += 1
  return params


def run_dir(p):
  """<save_dir>/results/<run name>, mirroring train_eval_offline (including its defaults)."""
  name = (f"{p.get('algo_name', 'fbrc')}_{p.get('task_name', 'halfcheetah-expert-v0')}"
          f"_seed={p.get('seed', 42)}")
  if 'expert_frac' in p:
    name += f"_expert_frac={float(p['expert_frac'])}"
  return os.path.join(p.get('save_dir', '/tmp/save/'), 'results', name)


def main():
  argv = sys.argv[1:]
  params = parse_params(argv)
  out_dir = run_dir(params)
  os.makedirs(out_dir, exist_ok=True)

  t0 = time.perf_counter()
  proc = subprocess.Popen([sys.executable, '-m', 'fisher_brc.train_eval_offline'] + argv)
  for sig in (signal.SIGINT, signal.SIGTERM):  # e.g. the IDE's stop button
    signal.signal(sig, lambda s, _: proc.send_signal(s))
  code = proc.wait()
  wall = time.perf_counter() - t0

  metrics = {
      'params': params,
      'exit_code': code,
      'wall_clock_s': round(wall, 1),
      'wall_clock_h': round(wall / 3600, 3),
      # Largest resident memory of the training process (ru_maxrss is in KB on Linux).
      'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
  }
  with open(os.path.join(out_dir, 'run_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
  print(f'[run_logged] {metrics}', file=sys.stderr)
  sys.exit(code)


if __name__ == '__main__':
  main()
