"""Agent package exports."""

from .base_agent import BaseAgent
from .qlearning_agent import QLearningAgent, QLearningConfig

__all__ = ["BaseAgent", "DQNAgent", "DQNConfig", "QLearningAgent", "QLearningConfig"]


def __getattr__(name: str):
    if name in {"DQNAgent", "DQNConfig"}:
        from .dqn_agent import DQNAgent, DQNConfig

        return {"DQNAgent": DQNAgent, "DQNConfig": DQNConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
