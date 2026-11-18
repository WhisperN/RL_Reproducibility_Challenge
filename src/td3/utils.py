"""Replay buffer for TD3+BC, adapted from https://github.com/sfujim/TD3_BC/blob/main/utils.py.

Differences to the original: the buffer is filled directly from a D4RL-style transition dict (as returned by
d4rl.qlearning_dataset or dataset_split.mix_datasets) instead of through convert_D4RL, and sampling uses a seeded
numpy Generator.
"""
import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, dataset, device, seed=0):
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.state = np.asarray(dataset["observations"], dtype=np.float32)
        self.action = np.asarray(dataset["actions"], dtype=np.float32)
        self.next_state = np.asarray(dataset["next_observations"], dtype=np.float32)
        self.reward = np.asarray(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        self.not_done = 1.0 - np.asarray(dataset["terminals"], dtype=np.float32).reshape(-1, 1)
        self.size = self.state.shape[0]

    def sample(self, batch_size):
        ind = self.rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[ind], device=self.device),
            torch.as_tensor(self.action[ind], device=self.device),
            torch.as_tensor(self.next_state[ind], device=self.device),
            torch.as_tensor(self.reward[ind], device=self.device),
            torch.as_tensor(self.not_done[ind], device=self.device),
        )

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
