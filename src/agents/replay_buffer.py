"""Experience replay buffer for off-policy deep RL agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    """Fixed-size replay buffer backed by a deque."""

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size > len(self._buffer):
            raise ValueError("batch_size cannot exceed current buffer size")

        indices = self.rng.choice(len(self._buffer), size=batch_size, replace=False)
        transitions = [self._buffer[int(i)] for i in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)

        return ReplayBatch(
            states=np.stack(states).astype(np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.stack(next_states).astype(np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)
