"""Tabular Q-learning agent for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base_agent import BaseAgent


@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995


class QLearningAgent(BaseAgent):
    """Q-table based off-policy learner for deterministic grid mazes."""

    def __init__(
        self,
        n_states: int,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    @classmethod
    def from_config(cls, n_states: int, n_actions: int, config: QLearningConfig, seed: int | None = None):
        return cls(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            seed=seed,
        )

    def select_action(self, state: int, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        bootstrap = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * bootstrap
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)
        return path

    def load(self, path: str | Path) -> None:
        q_table = np.load(path)
        if q_table.shape != self.q_table.shape:
            raise ValueError(f"Expected Q-table shape {self.q_table.shape}, got {q_table.shape}.")
        self.q_table = q_table.astype(np.float64, copy=True)
