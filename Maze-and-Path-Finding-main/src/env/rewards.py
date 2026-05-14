"""
rewards.py
----------
All reward logic lives here — isolated from the environment so you can
swap reward functions easily in Phase 2 (reward shaping experiments).

Design philosophy
-----------------
- The environment calls `RewardFunction.compute(...)` after every step.
- Each reward function is a small class with a single `compute` method.
- Adding a new reward scheme = adding a new class below.

Phase 1 uses `SparseWithPenalty` (the standard baseline).
"""

from dataclasses import dataclass
from typing import Tuple


# ──────────────────────── data container ───────────────────────────────────

@dataclass(frozen=True)
class StepInfo:
    """
    All context the reward function might need for a single step.

    Attributes
    ----------
    prev_pos    : (row, col) before the action
    next_pos    : (row, col) after the action  (same as prev if wall hit)
    goal_pos    : (row, col) of the goal cell
    hit_wall    : True if the action tried to walk into a wall / boundary
    reached_goal: True if next_pos == goal_pos
    steps_taken : total steps taken so far in this episode
    max_steps   : episode step budget
    optimal_dist: BFS optimal distance (used by shaping rewards)
    """
    prev_pos    : Tuple[int, int]
    next_pos    : Tuple[int, int]
    goal_pos    : Tuple[int, int]
    hit_wall    : bool
    reached_goal: bool
    steps_taken : int
    max_steps   : int
    optimal_dist: int = 0   # only used by shaping variants


# ──────────────────────── base class ───────────────────────────────────────

class RewardFunction:
    """Abstract base — all reward functions inherit from this."""

    def compute(self, info: StepInfo) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


# ──────────────────────── Phase 1 baseline ─────────────────────────────────

class SparseWithPenalty(RewardFunction):
    """
    Standard reward for Phase 1.

    +goal_reward   : reached the goal
    -wall_penalty  : tried to walk into a wall (action rejected, no movement)
    -step_penalty  : every other step (encourages shorter paths)

    Why separate the penalties?
      The wall penalty discourages aimless wall-bashing.
      The step penalty pushes the agent toward efficient paths.
      Without the step penalty the agent may take any path and still succeed.
    """

    def __init__(
        self,
        goal_reward  : float = 1.0,
        wall_penalty : float = 0.5,   # will be negated inside compute()
        step_penalty : float = 0.01,  # will be negated inside compute()
    ):
        self.goal_reward  = goal_reward
        self.wall_penalty = wall_penalty
        self.step_penalty = step_penalty

    def compute(self, info: StepInfo) -> float:
        if info.reached_goal:
            return self.goal_reward
        if info.hit_wall:
            return -self.wall_penalty
        return -self.step_penalty

    def __repr__(self) -> str:
        return (
            f"SparseWithPenalty("
            f"goal={self.goal_reward}, "
            f"wall=-{self.wall_penalty}, "
            f"step=-{self.step_penalty})"
        )


# ──────────────────────── Phase 2 stubs (do not use yet) ───────────────────
# These are here so the structure is ready for Phase 2 reward experiments.
# They raise NotImplementedError to prevent accidental use.

class PotentialShaping(RewardFunction):
    """
    Reward shaping using BFS distance as potential.
    F(s,s') = γ·Φ(s') − Φ(s)   where Φ(s) = -BFS_dist(s, goal)

    Guaranteed not to change the optimal policy (Ng et al. 1999).
    FOR PHASE 2 ONLY.
    """

    def __init__(self, gamma: float = 0.99):
        self.gamma   = gamma
        self._base   = SparseWithPenalty()

    def compute(self, info: StepInfo) -> float:
        raise NotImplementedError("PotentialShaping is for Phase 2 only.")


class DenseDistance(RewardFunction):
    """
    Dense reward: negative Manhattan distance to goal.
    Simple but can cause sub-optimal shortcut behaviour.
    FOR PHASE 2 ONLY.
    """

    def compute(self, info: StepInfo) -> float:
        raise NotImplementedError("DenseDistance is for Phase 2 only.")


# ──────────────────────── factory helper ───────────────────────────────────

def get_reward_function(name: str, **kwargs) -> RewardFunction:
    """
    Convenience factory so configs can reference reward functions by name.

    Usage:
        rf = get_reward_function("sparse")
        rf = get_reward_function("sparse", goal_reward=2.0)
    """
    registry = {
        "sparse"    : SparseWithPenalty,
        "potential" : PotentialShaping,
        "dense"     : DenseDistance,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[name](**kwargs)