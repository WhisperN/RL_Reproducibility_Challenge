"""Train and evaluate TD3+BC offline, with the same data, flags and outputs as fisher_brc.train_eval_offline.

Run from src/:

        python -m td3.train_eval_offline --task_name hopper-expert-v2 --expert_frac 0.75 --seed 0 --save_dir results/td3_bc

What matches Fisher-BRC:
    * Data: same loading path. Without --expert_frac, d4rl.qlearning_dataset(task_name); with it,
        dataset_split.mix_datasets(expert, random, expert_frac, seed=--seed), i.e. the identical mixture Fisher-BRC trains
        on for the same --seed.
    * Outputs: TF summary event files under <save_dir>/tb/<run name> (training/*) and <save_dir>/results/<run name>
        (evaluation/returns = D4RL normalized score * 100, evaluation/length), logged at step i+1 every --eval_interval
        updates, averaged over --num_eval_episodes episodes. Run name: td3_bc_<task_name>_seed=<seed>[_expert_frac=<f>].
What follows TD3+BC (Fujimoto & Gu, 2021) instead: hyperparameters, state normalization, and evaluation on a separate
env seeded with seed + 100 (the original main.py).
"""
import os

from absl import app
from absl import flags
import d4rl  # noqa: F401  (registers the D4RL envs)
import gym
import numpy as np
import tensorflow as tf
import torch
import tqdm

import dataset_split
from td3 import td3
from td3 import utils

FLAGS = flags.FLAGS

# Hide gym 0.23's repeated 'Box bound precision lowered' warning (harmless, printed on every gym.make).
gym.logger.set_level(gym.logger.ERROR)

# Same names and defaults as fisher_brc.train_eval_offline.
flags.DEFINE_string('task_name', 'halfcheetah-expert-v0', 'Env name.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('num_updates', 1_000_000, 'Num updates.')
flags.DEFINE_integer('num_eval_episodes', 10, 'Num eval episodes.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10_000, 'Evaluation interval.')
flags.DEFINE_string('save_dir', '/tmp/save/', 'Saving directory.')
flags.DEFINE_float(
        'expert_frac', None,
        'If set, train on a random/expert mixture with this fraction of expert '
        'transitions (0.75 = 25% random / 75% expert). task_name must then be an '
        "'-expert-' task, e.g. hopper-expert-v2. The mixture is sampled with "
        '--seed. If unset, task_name\'s dataset is used unchanged.')
# TD3+BC hyperparameters (defaults from the original main.py).
flags.DEFINE_float('discount', 0.99, 'Discount factor.')
flags.DEFINE_float('tau', 0.005, 'Target network update rate.')
flags.DEFINE_float('policy_noise', 0.2, 'Target policy smoothing noise (scaled by max_action).')
flags.DEFINE_float('noise_clip', 0.5, 'Target noise clip (scaled by max_action).')
flags.DEFINE_integer('policy_freq', 2, 'Delayed actor update frequency.')
flags.DEFINE_float('alpha', 2.5, 'TD3+BC alpha (weight of the Q term vs. BC).')
flags.DEFINE_boolean('normalize', True, 'Normalize states with dataset mean/std.')

ALGO_NAME = 'td3_bc'


def run_name():
    name = f'{ALGO_NAME}_{FLAGS.task_name}_seed={FLAGS.seed}'
    if FLAGS.expert_frac is not None:
        name += f'_expert_frac={FLAGS.expert_frac}'
    return name


def load_dataset(task_name, env, expert_frac, mix_seed):
    """Same dataset selection as fisher_brc.d4rl_utils.create_d4rl_env_and_dataset, returned as numpy arrays."""
    if expert_frac is None:
        return d4rl.qlearning_dataset(env)
    if '-expert-' not in task_name:
        raise ValueError(f"expert_frac requires an '-expert-' task_name, got {task_name!r}.")
    if not 0.0 <= expert_frac <= 1.0:
        raise ValueError(f'expert_frac must be in [0, 1], got {expert_frac}.')
    return dataset_split.mix_datasets(
            dataset_split.import_dataset(task_name),
            dataset_split.import_dataset(task_name.replace('-expert-', '-random-')),
            expert_frac=expert_frac,
            seed=mix_seed)


def evaluate(policy, task_name, seed, mean, std, num_episodes):
    """TD3+BC's eval_policy: fresh env seeded with seed + 100. Returns raw average return and average length."""
    eval_env = gym.make(task_name)
    eval_env.seed(seed + 100)
    total_returns, total_timesteps = 0.0, 0
    for _ in range(num_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            total_returns += reward
            total_timesteps += 1
    return total_returns / num_episodes, total_timesteps / num_episodes, eval_env


def main(_):
    env = gym.make(FLAGS.task_name)
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = load_dataset(FLAGS.task_name, env, FLAGS.expert_frac, FLAGS.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = utils.ReplayBuffer(dataset, td3.device, seed=FLAGS.seed)
    if FLAGS.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0.0, 1.0

    policy = td3.TD3_BC(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            discount=FLAGS.discount,
            tau=FLAGS.tau,
            policy_noise=FLAGS.policy_noise * max_action,
            noise_clip=FLAGS.noise_clip * max_action,
            policy_freq=FLAGS.policy_freq,
            alpha=FLAGS.alpha)

    hparam_str = run_name()
    summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.save_dir, 'tb', hparam_str))
    result_writer = tf.summary.create_file_writer(os.path.join(FLAGS.save_dir, 'results', hparam_str))

    for i in tqdm.tqdm(range(FLAGS.num_updates)):
        info_dict = policy.train(replay_buffer, FLAGS.batch_size)

        if i % FLAGS.log_interval == 0:
            with summary_writer.as_default():
                for k, v in info_dict.items():
                    tf.summary.scalar(f'training/{k}', v, step=i)

        if (i + 1) % FLAGS.eval_interval == 0:
            average_returns, average_length, eval_env = evaluate(
                    policy, FLAGS.task_name, FLAGS.seed, mean, std, FLAGS.num_eval_episodes)
            average_returns = eval_env.get_normalized_score(average_returns) * 100.0

            with result_writer.as_default():
                tf.summary.scalar('evaluation/returns', average_returns, step=i + 1)
                tf.summary.scalar('evaluation/length', average_length, step=i + 1)


if __name__ == '__main__':
    app.run(main)
