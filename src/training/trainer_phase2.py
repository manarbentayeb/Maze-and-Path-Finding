"""Training helpers for Phase 2 door/key experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from src.env.door_key_maze_env import DoorKeyMazeEnv
from src.utils.metrics_phase2 import DoorKeyMetricsTracker
from src.utils.seed import set_global_seed


DQN_CFG = {
    "episodes": 10_000,
    "lr": 1e-3,
    "gamma": 0.99,
    "buffer_size": 50_000,
    "batch": 64,
    "target_update": 500,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay": 0.995,
    "warmup": 1_000,
    "hidden": [256, 128],
}

PPO_CFG = {
    "total_timesteps": 1_000_000,
    "lr": 3e-4,
    "n_steps": 4096,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}


@dataclass
class Phase2TrainingResult:
    model: Any
    tracker: DoorKeyMetricsTracker
    paths: dict[str, list[tuple[int, int]]]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition) -> None:
        self.buf.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.buf, batch_size)

    def __len__(self) -> int:
        return len(self.buf)


def build_q_network(obs_dim: int, hidden: list[int], n_actions: int):
    """Create the Phase 2 MLP Q-network."""
    import torch.nn as nn

    layers = []
    in_dim = obs_dim
    for width in hidden:
        layers.extend([nn.Linear(in_dim, width), nn.ReLU()])
        in_dim = width
    layers.append(nn.Linear(in_dim, n_actions))
    return nn.Sequential(*layers)


def train_dqn_phase2(
    env: DoorKeyMazeEnv,
    cfg: dict | None = None,
    seed: int = 0,
    double: bool = False,
    logger=None,
    capture_paths: bool = True,
) -> Phase2TrainingResult:
    """Train DQN or DDQN on ``DoorKeyMazeEnv``.

    Set ``double=True`` for DDQN. All other hyperparameters and architecture are
    intentionally shared with DQN for the Phase 2 ablation.
    """
    import torch
    import torch.nn.functional as F

    cfg = {**DQN_CFG, **(cfg or {})}
    set_global_seed(seed)
    random.seed(seed)
    env.action_space.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    q_net = build_q_network(obs_dim, cfg["hidden"], env.action_space.n).to(device)
    target_net = build_q_network(obs_dim, cfg["hidden"], env.action_space.n).to(device)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=cfg["lr"])
    replay = ReplayBuffer(cfg["buffer_size"])

    eps = cfg["eps_start"]
    total_steps = 0
    tracker = DoorKeyMetricsTracker(window=100, optimal_path_len=env._optimal)
    paths: dict[str, list[tuple[int, int]]] = {}

    for episode in range(1, cfg["episodes"] + 1):
        obs, _ = env.reset()
        state = torch.tensor(obs, dtype=torch.float32, device=device)
        total_reward = 0.0
        episode_path = [env.agent_pos]

        while True:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(q_net(state.unsqueeze(0)).argmax(1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.push(obs, action, reward, next_obs, done)
            obs = next_obs
            state = torch.tensor(obs, dtype=torch.float32, device=device)
            total_reward += reward
            total_steps += 1
            episode_path.append(env.agent_pos)

            if len(replay) >= max(cfg["warmup"], cfg["batch"]):
                batch = replay.sample(cfg["batch"])
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)
                states = torch.tensor(np.array(s_b), dtype=torch.float32, device=device)
                actions = torch.tensor(a_b, dtype=torch.long, device=device)
                rewards = torch.tensor(r_b, dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array(ns_b), dtype=torch.float32, device=device)
                dones = torch.tensor(d_b, dtype=torch.float32, device=device)

                q_vals = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    if double:
                        best_actions = q_net(next_states).argmax(1, keepdim=True)
                        next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
                    else:
                        next_q = target_net(next_states).max(1).values
                    target = rewards + cfg["gamma"] * next_q * (1.0 - dones)

                loss = F.mse_loss(q_vals, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if total_steps % cfg["target_update"] == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        eps = max(cfg["eps_end"], eps * cfg["eps_decay"])
        tracker.record(
            episode=episode,
            total_reward=total_reward,
            steps=env.steps_taken,
            success=bool(terminated),
            picked_up_key=bool(info["picked_up_key"]),
            opened_door=bool(info["opened_door"]),
            steps_to_key=info["steps_to_key"],
            steps_to_door=info["steps_to_door"],
            optimal_path_len=info["optimal_path_len"],
        )

        if capture_paths and episode == 1:
            paths["early"] = episode_path
        if capture_paths and episode == cfg["episodes"]:
            paths["late_training"] = episode_path

        if logger is not None:
            logger.log_episode(
                episode=episode,
                total_reward=total_reward,
                steps=env.steps_taken,
                success=bool(terminated),
                epsilon=eps,
                key=info["picked_up_key"],
                door=info["opened_door"],
                rolling_success=tracker.rolling_success_rate() or 0.0,
            )

    if capture_paths:
        paths["greedy"] = rollout_dqn_greedy_path(env, q_net)

    return Phase2TrainingResult(model=q_net, tracker=tracker, paths=paths)


def rollout_dqn_greedy_path(env: DoorKeyMazeEnv, q_net) -> list[tuple[int, int]]:
    import torch

    device = next(q_net.parameters()).device
    obs, _ = env.reset()
    path = [env.agent_pos]
    for _ in range(env.max_steps):
        state = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action = int(q_net(state.unsqueeze(0)).argmax(1).item())
        obs, _, terminated, truncated, _ = env.step(action)
        path.append(env.agent_pos)
        if terminated or truncated:
            break
    return path


class SubgoalLoggerCallback:
    """Stable-Baselines3 callback that stores per-episode subgoal flags."""

    def __init__(self):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.ep_key = []
                self.ep_door = []

            def _on_step(self):
                for info in self.locals.get("infos", []):
                    if "picked_up_key" in info:
                        self.ep_key.append(info["picked_up_key"])
                        self.ep_door.append(info["opened_door"])
                return True

        self.callback = _Callback()


def train_ppo_phase2(env: DoorKeyMazeEnv, cfg: dict | None = None, seed: int = 0):
    """Train PPO with Stable-Baselines3 on the Phase 2 environment."""
    from stable_baselines3 import PPO

    cfg = {**PPO_CFG, **(cfg or {})}
    set_global_seed(seed)
    callback = SubgoalLoggerCallback().callback
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=cfg["lr"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        seed=seed,
    )
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callback)
    return model, callback.ep_key, callback.ep_door
