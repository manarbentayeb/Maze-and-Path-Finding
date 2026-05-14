"""Deep Q-Network agent for the maze environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base_agent import BaseAgent
from .networks import MLPQNetwork
from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    buffer_size: int = 50_000
    batch_size: int = 64
    target_update_every: int = 250
    warmup_steps: int = 500
    hidden_dims: tuple[int, ...] = (64, 64)
    grad_clip: float | None = 10.0
    device: str = "cpu"


class DQNAgent(BaseAgent):
    """DQN with replay memory and a periodically synced target network."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int = 4,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_every: int = 250,
        warmup_steps: int = 500,
        hidden_dims: tuple[int, ...] = (128, 128),
        grad_clip: float | None = 10.0,
        seed: int | None = None,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.warmup_steps = warmup_steps
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.train_steps = 0

        if seed is not None:
            torch.manual_seed(seed)

        self.q_network = MLPQNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.target_network = MLPQNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        self.sync_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(buffer_size, seed=seed)

    @classmethod
    def from_config(cls, obs_dim: int, n_actions: int, config: DQNConfig, seed: int | None = None):
        return cls(
            obs_dim=obs_dim,
            n_actions=n_actions,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_every=config.target_update_every,
            warmup_steps=config.warmup_steps,
            hidden_dims=config.hidden_dims,
            grad_clip=config.grad_clip,
            seed=seed,
            device=config.device,
        )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        q_values = self.q_values(np.asarray(state, dtype=np.float32)[None, :])
        return int(np.argmax(q_values[0]))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        q_selected = self.q_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_selected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_every == 0:
            self.sync_target_network()

        return float(loss.item())

    def q_values(self, states: np.ndarray) -> np.ndarray:
        self.q_network.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            values = self.q_network(tensor).cpu().numpy()
        self.q_network.train()
        return values

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_network.state_dict(), path)
        return path

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.sync_target_network()
        