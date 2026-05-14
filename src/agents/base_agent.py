"""Common interface for maze agents."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Minimal agent contract used by trainers and evaluators."""

    @abstractmethod
    def select_action(self, state: int, training: bool = True) -> int:
        """Return an action for a scalar state index."""

    @abstractmethod
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Update the agent and return the TD error or loss-like value."""
