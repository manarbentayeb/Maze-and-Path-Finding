"""Evaluation helpers for trained maze agents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.agents.qlearning_agent import QLearningAgent
from src.env.maze_env import MazeEnv


@dataclass
class EvaluationResult:
    success_rate: float
    mean_return: float
    mean_steps: float
    paths: list[list[tuple[int, int]]]


def evaluate_qlearning(
    env: MazeEnv,
    agent: QLearningAgent,
    episodes: int = 100,
) -> EvaluationResult:
    """Evaluate a Q-learning agent greedily."""
    rewards: list[float] = []
    steps: list[int] = []
    successes: list[bool] = []
    paths: list[list[tuple[int, int]]] = []

    for _ in range(episodes):
        env.reset()
        total_reward = 0.0
        path = [env.agent_pos]

        for _ in range(env.max_steps):
            state = env.pos_to_state_idx()
            action = agent.select_action(state, training=False)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            path.append(env.agent_pos)
            if terminated or truncated:
                break

        rewards.append(total_reward)
        steps.append(env.steps_taken)
        successes.append(env.agent_pos == env.goal_pos)
        paths.append(path)

    return EvaluationResult(
        success_rate=float(np.mean(successes)),
        mean_return=float(np.mean(rewards)),
        mean_steps=float(np.mean(steps)),
        paths=paths,
    )
