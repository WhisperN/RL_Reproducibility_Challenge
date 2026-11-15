# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loads D4RL dataset from pickle files."""
import typing

import d4rl
import gym
import numpy as np
import tensorflow as tf
# --- CHANGE START: import for random/expert mixtures ---
import dataset_split
# --- CHANGE END ---


def create_d4rl_env_and_dataset(
    task_name,
    batch_size,
    # --- CHANGE START: new optional arguments ---
    expert_frac=None,
    mix_seed=0,
    # --- CHANGE END ---
):
  """Create gym environment and dataset for d4rl.

  Args:
    task_name: Name of d4rl task.
    batch_size: Mini batch size.
    expert_frac: If None, the dataset of `task_name` is used unchanged. Otherwise
      `task_name` must be an '-expert-' task, and the training data is a
      random/expert mixture with this fraction of expert transitions.
    mix_seed: Seed for the mixture sampling.
  Returns:
    Gym env and dataset.
  """
  env = gym.make(task_name)
  # --- CHANGE START: optional random/expert mixture (src/dataset_split.py) ---
  if expert_frac is None:
    dataset = d4rl.qlearning_dataset(env)
  else:
    if '-expert-' not in task_name:
      raise ValueError(
          f"expert_frac requires an '-expert-' task_name, got {task_name!r}.")
    if not 0.0 <= expert_frac <= 1.0:
      raise ValueError(f'expert_frac must be in [0, 1], got {expert_frac}.')
    random_task_name = task_name.replace('-expert-', '-random-')
    dataset = dataset_split.mix_datasets(
        dataset_split.import_dataset(task_name),
        dataset_split.import_dataset(random_task_name),
        expert_frac=expert_frac,
        seed=mix_seed)
  # --- CHANGE END ---

  states = np.array(dataset['observations'], dtype=np.float32)
  actions = np.array(dataset['actions'], dtype=np.float32)
  rewards = np.array(dataset['rewards'], dtype=np.float32)
  discounts = np.array(np.logical_not(dataset['terminals']), dtype=np.float32)
  next_states = np.array(dataset['next_observations'], dtype=np.float32)

  dataset = tf.data.Dataset.from_tensor_slices(
      (states, actions, rewards, discounts, next_states)).cache().shuffle(  # pyrefly: ignore[bad-argument-type]
          states.shape[0], reshuffle_each_iteration=True).repeat().batch(
              batch_size,
              drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  return env, dataset
