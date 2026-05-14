"""Training loop for the Phase 1 tabular Q-learning baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.agents.qlearning_agent import QLearningAgent
from src.env.maze_env import MazeEnv
from src.utils.metrics import MetricsTracker


@dataclass
class QLearningTrainingResult:
    agent: QLearningAgent
    tracker: MetricsTracker
    paths: dict[str, list[tuple[int, int]]]


def train_qlearning(
    env: MazeEnv,
    agent: QLearningAgent,
    episodes: int = 5000,
    log_every: int = 100,
    logger=None,
    capture_paths: bool = True,
) -> QLearningTrainingResult:
    """Train a Q-learning agent on ``env`` and return metrics plus paths."""
    tracker = MetricsTracker(window=100, optimal_path_len=env._optimal)
    paths: dict[str, list[tuple[int, int]]] = {}

    for episode in range(1, episodes + 1):
        env.reset()
        state = env.pos_to_state_idx()
        total_reward = 0.0
        done = False
        episode_path = [env.agent_pos]

        while not done:
            action = agent.select_action(state, training=True)
            _, reward, terminated, truncated, _ = env.step(action)
            next_state = env.pos_to_state_idx()
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)
            state = next_state
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
            print(
                f"Episode {episode:>5}/{episodes} | "
                f"return={total_reward:>7.3f} | steps={env.steps_taken:>4} | "
                f"success={int(success)} | eps={agent.epsilon:.3f} | rolling_success={rolling_text}"
            )

        agent.decay_epsilon()

    if capture_paths:
        paths["greedy"] = rollout_greedy_path(env, agent)

    return QLearningTrainingResult(agent=agent, tracker=tracker, paths=paths)


def rollout_greedy_path(env: MazeEnv, agent: QLearningAgent) -> list[tuple[int, int]]:
    """Run one greedy evaluation episode and return visited positions."""
    env.reset()
    path = [env.agent_pos]

    for _ in range(env.max_steps):
        state = env.pos_to_state_idx()
        action = agent.select_action(state, training=False)
        _, _, terminated, truncated, _ = env.step(action)
        path.append(env.agent_pos)
        if terminated or truncated:
            break

    return path


def q_table_for_env(env: MazeEnv) -> np.ndarray:
    return np.zeros((env.size * env.size, env.action_space.n), dtype=np.float64)
