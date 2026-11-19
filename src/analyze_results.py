"""Aggregate TD3+BC and Fisher-BRC evaluation results (protocol of Table 2/7 in Fujimoto & Gu, 2021).

Score of a run = mean normalized score over its final 10 evaluations; reported as mean ± std across seeds.

Usage (from src/):  python analyze_results.py results --out results/analysis

Writes per_run.csv, summary.md, learning_curves.png and score_vs_mixture.png to --out.
"""
import argparse
import collections
import csv
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, TENSORS
from tensorboard.util.tensor_util import make_ndarray

RUN_NAME = re.compile(r'(?P<algo>fbrc|td3_bc)_(?P<task>.+)_seed=(?P<seed>\d+)_expert_frac=(?P<frac>[0-9.]+)')
LABELS = {'td3_bc': 'TD3+BC', 'fbrc': 'Fisher-BRC'}
COLORS = {'td3_bc': 'C0', 'fbrc': 'C1'}  # same color per algorithm in every plot


def mixture(frac):
    return f'{round((1 - frac) * 100)}/{round(frac * 100)}'  # random/expert, e.g. 0.75 -> '25/75'


def read_curve(run_dir):
    """(steps, scores) of evaluation/returns. tf.summary.scalar in TF2 stores scalars as tensors."""
    curve = {}
    for path in sorted(glob.glob(os.path.join(run_dir, 'events.out.tfevents.*'))):
        acc = EventAccumulator(path, size_guidance={TENSORS: 0})
        acc.Reload()
        if 'evaluation/returns' in acc.Tags()['tensors']:
            curve.update({e.step: float(make_ndarray(e.tensor_proto)) for e in acc.Tensors('evaluation/returns')})
    steps = sorted(curve)
    return np.array(steps), np.array([curve[s] for s in steps])


def load_runs(roots, last_n):
    run_dirs = {os.path.dirname(os.path.abspath(p))
                for root in roots for p in glob.glob(f'{root}/**/results/*/events.out.tfevents.*', recursive=True)}
    runs = []
    for run_dir in sorted(run_dirs):
        m = RUN_NAME.fullmatch(os.path.basename(run_dir))
        if not m:
            continue
        steps, scores = read_curve(run_dir)
        if len(scores) < last_n:
            print(f'warning: {run_dir} has only {len(scores)} evaluations')
        metrics_files = glob.glob(os.path.join(run_dir, 'run_metrics*.json'))
        metrics = json.load(open(metrics_files[0])) if metrics_files else {}
        runs.append(dict(algo=m['algo'], task=m['task'], frac=float(m['frac']), seed=int(m['seed']),
                         steps=steps, scores=scores, final=scores[-last_n:].mean(),
                         wall_clock_h=metrics.get('wall_clock_h'), peak_rss_mb=metrics.get('peak_rss_mb')))
    if len({r['steps'][1] - r['steps'][0] for r in runs if len(r['steps']) > 1}) > 1:
        print('warning: runs use different eval intervals, so their final 10 evaluations are not comparable')
    return runs


def group(runs):
    """{(task, frac): {algo: [runs]}}, mixtures ordered from most to least expert data."""
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sorted(runs, key=lambda r: (r['task'], -r['frac'])):
        groups[r['task'], r['frac']][r['algo']].append(r)
    return groups


def write_per_run_csv(path, runs):
    fields = ['algo', 'task', 'frac', 'seed', 'final', 'wall_clock_h', 'peak_rss_mb']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(runs)


def write_summary(path, groups, last_n):
    lines = [f'Normalized score, mean of the final {last_n} evaluations; mean ± std across seeds (n).\n',
             '| Task | Random/expert | TD3+BC | Fisher-BRC | TD3+BC − Fisher-BRC |', '|---|---|---|---|---|']
    for (task, frac), by_algo in groups.items():
        finals = {a: [r['final'] for r in by_algo.get(a, [])] for a in LABELS}
        cells = [f'{np.mean(f):.1f} ± {np.std(f):.1f} ({len(f)})' if f else '-' for f in finals.values()]
        diff = f'{np.mean(finals["td3_bc"]) - np.mean(finals["fbrc"]):+.1f}' if all(finals.values()) else '-'
        lines.append(f'| {task} | {mixture(frac)} | {" | ".join(cells)} | {diff} |')
    text = '\n'.join(lines) + '\n'
    with open(path, 'w') as f:
        f.write(text)
    print(text)


def plot_learning_curves(path, groups):
    fig, axes = plt.subplots(1, len(groups), figsize=(4.5 * len(groups), 3.5), sharey=True, squeeze=False)
    for ax, ((task, frac), by_algo) in zip(axes[0], groups.items()):
        for algo, runs in sorted(by_algo.items(), key=lambda kv: list(LABELS).index(kv[0])):
            n = min(len(r['scores']) for r in runs)  # cut all seeds to the shortest run
            scores = np.array([r['scores'][:n] for r in runs])
            mean, std = scores.mean(0), scores.std(0)
            ax.plot(runs[0]['steps'][:n], mean, color=COLORS[algo], label=f'{LABELS[algo]} (n={len(runs)})')
            ax.fill_between(runs[0]['steps'][:n], mean - std, mean + std, color=COLORS[algo], alpha=0.2)
        ax.set(title=f'{task}, {mixture(frac)} random/expert', xlabel='Time steps')
        ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter())  # 250000 -> 250k
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel('Normalized score')
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_score_vs_mixture(path, groups):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for algo, label in LABELS.items():
        points = [(100 * (1 - frac), [r['final'] for r in by_algo[algo]])
                  for (_, frac), by_algo in groups.items() if algo in by_algo]
        x = [p[0] for p in points]
        ax.errorbar(x, [np.mean(f) for _, f in points], yerr=[np.std(f) for _, f in points],
                    marker='o', capsize=4, color=COLORS[algo], label=label)
        ax.set_xticks(x)
    ax.set(xlabel='Random transitions in dataset (%)', ylabel='Normalized score (final evaluations)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description='Aggregate TD3+BC and Fisher-BRC evaluation results.')
    parser.add_argument('roots', nargs='*', default=['results'], help='directories containing <save_dir>/results/')
    parser.add_argument('--out', default='results/analysis')
    parser.add_argument('--last_n', type=int, default=10, help='final evaluations averaged per run')
    args = parser.parse_args()

    runs = load_runs(args.roots, args.last_n)
    if not runs:
        raise SystemExit(f'No runs found under {args.roots}.')
    os.makedirs(args.out, exist_ok=True)
    groups = group(runs)
    write_per_run_csv(os.path.join(args.out, 'per_run.csv'), runs)
    write_summary(os.path.join(args.out, 'summary.md'), groups, args.last_n)
    plot_learning_curves(os.path.join(args.out, 'learning_curves.png'), groups)
    plot_score_vs_mixture(os.path.join(args.out, 'score_vs_mixture.png'), groups)
    print(f'{len(runs)} runs -> {args.out}')


if __name__ == '__main__':
    main()
