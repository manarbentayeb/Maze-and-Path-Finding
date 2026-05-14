"""Training loop for the DQN maze agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.env.maze_env import MazeEnv
from src.utils.metrics import MetricsTracker


@dataclass
class DQNTrainingResult:
    agent: DQNAgent
    tracker: MetricsTracker
    paths: dict[str, list[tuple[int, int]]]
    losses: list[float]


def train_dqn(
    env: MazeEnv,
    agent: DQNAgent,
    episodes: int = 1000,
    log_every: int = 100,
    logger=None,
    capture_paths: bool = True,
) -> DQNTrainingResult:
    """Train a DQN agent on ``env`` and return metrics, paths, and losses."""
    tracker = MetricsTracker(window=100, optimal_path_len=env._optimal)
    paths: dict[str, list[tuple[int, int]]] = {}
    losses: list[float] = []

    for episode in range(1, episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        episode_path = [env.agent_pos]

        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.update(obs, action, reward, next_obs, done)
            if loss > 0.0:
                losses.append(loss)

            obs = next_obs
            total_reward += reward
            episode_path.append(env.agent_pos)

        success = bool(terminated)
        tracker.record(episode=episode, total_reward=total_reward, steps=env.steps_taken, success=success)

        if capture_paths and episode == 1:
            paths["early"] = episode_path
        if capture_paths and episode == episodes:
            paths["late_training"] = episode_path

        if logger is not None:
            logger.log_episode(
                episode=episode,
                total_reward=total_reward,
                steps=env.steps_taken,
                success=success,
                epsilon=agent.epsilon,
                rolling_success=tracker.rolling_success_rate() or 0.0,
            )
        elif log_every and episode % log_every == 0:
            rolling = tracker.rolling_success_rate()
            rolling_text = "n/a" if rolling is None else f"{rolling:.2%}"
            recent_loss = float(np.mean(losses[-100:])) if losses else 0.0
            print(
                f"Episode {episode:>5}/{episodes} | "
                f"return={total_reward:>7.3f} | steps={env.steps_taken:>4} | "
                f"success={int(success)} | eps={agent.epsilon:.3f} | "
                f"loss={recent_loss:.4f} | rolling_success={rolling_text}"
            )

        agent.decay_epsilon()

    if capture_paths:
        paths["greedy"] = rollout_greedy_path(env, agent)

    return DQNTrainingResult(agent=agent, tracker=tracker, paths=paths, losses=losses)


def rollout_greedy_path(env: MazeEnv, agent: DQNAgent) -> list[tuple[int, int]]:
    """Run one greedy evaluation episode and return visited positions."""
    obs, _ = env.reset()
    path = [env.agent_pos]

    for _ in range(env.max_steps):
        action = agent.select_action(obs, training=False)
        obs, _, terminated, truncated, _ = env.step(action)
        path.append(env.agent_pos)
        if terminated or truncated:
            break

    return path


def q_table_from_dqn(env: MazeEnv, agent: DQNAgent) -> np.ndarray:
    """Evaluate the network at every free-cell agent position for plotting."""
    base_grid = (1.0 - env.maze).astype(np.float32)
    gr, gc = env.goal_pos
    observations = []

    for r in range(env.size):
        for c in range(env.size):
            grid = base_grid.copy()
            grid[gr, gc] = 3.0
            if env.maze[r, c] == 0:
                grid[r, c] = 2.0
            observations.append(grid.flatten())

    return agent.q_values(np.stack(observations).astype(np.float32))
