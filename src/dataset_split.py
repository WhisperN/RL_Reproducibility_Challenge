import os
import typing

import d4rl
import d4rl.gym_mujoco
import gym
import numpy as np

def mix_datasets(
        expert_dataset: typing.Dict[str, np.ndarray],
        random_dataset: typing.Dict[str, np.ndarray],
        expert_frac: float,
        seed: int,
) -> typing.Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = min(len(expert_dataset["observations"]), len(random_dataset["observations"]))
    n_expert = int(total * expert_frac)

    expert_indices = rng.choice(len(expert_dataset["observations"]), n_expert, replace=False)
    random_indices = rng.choice(len(random_dataset["observations"]), total - n_expert, replace=False)
    return {k: np.concatenate((expert_dataset[k][expert_indices], random_dataset[k][random_indices]), axis=0) for k in expert_dataset}

def import_dataset(env_name, h5path=None):
    """Loads a D4RL dataset through its env, e.g. 'hopper-expert-v2'.

    Without h5path, d4rl downloads the file to ~/.d4rl/datasets on first use.
    """
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env, dataset=env.get_dataset(h5path=h5path))
    return {k: v.astype(np.float32) for k, v in dataset.items()}

if __name__ == "__main__":
    expert = import_dataset("hopper-expert-v2")
    random_ds = import_dataset("hopper-random-v2")
    combined = mix_datasets(expert, random_ds, expert_frac=0.5, seed=0)
    print("Expert dataset size:", len(expert["observations"]))
