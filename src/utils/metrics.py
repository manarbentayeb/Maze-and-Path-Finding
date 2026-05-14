"""
metrics.py
----------
Tracks all metrics defined in Section 06 of the research spec.

Design
------
- `EpisodeMetrics`   : dataclass for one episode's raw data
- `MetricsTracker`   : accumulates episodes, computes rolling stats
- `compute_summary`  : final summary dict for the comparison table

Usage
-----
    tracker = MetricsTracker(window=100)

    # inside training loop:
    tracker.record(episode=ep, total_reward=r, steps=s, success=done)

    # check live:
    print(tracker.rolling_success_rate())

    # at end:
    summary = tracker.compute_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing      import List, Optional
import numpy as np


# ──────────────────────── per-episode record ───────────────────────────────

@dataclass
class EpisodeMetrics:
    episode      : int
    total_reward : float
    steps        : int
    success      : bool    # True = goal reached within max_steps


# ──────────────────────── tracker ──────────────────────────────────────────

class MetricsTracker:
    """
    Accumulates episode records and exposes rolling and aggregate statistics.

    Parameters
    ----------
    window : int
        Window size for rolling averages (default 100 episodes).
    optimal_path_len : int
        BFS optimal path length for this maze. Used to compute path
        optimality ratio. Set to 0 to skip optimality tracking.
    """

    def __init__(self, window: int = 100, optimal_path_len: int = 0):
        self.window           = window
        self.optimal_path_len = optimal_path_len

        self._records: List[EpisodeMetrics] = []

        # Convergence episode: first ep where rolling success ≥ 0.90
        self._convergence_episode: Optional[int] = None

    # ── recording ──────────────────────────────────────────────────────

    def record(
        self,
        episode     : int,
        total_reward: float,
        steps       : int,
        success     : bool,
    ) -> None:
        """Add one episode's data."""
        rec = EpisodeMetrics(episode, total_reward, steps, success)
        self._records.append(rec)

        # Check convergence on every record (only sets once)
        if self._convergence_episode is None:
            rate = self.rolling_success_rate()
            if rate is not None and rate >= 0.90:
                self._convergence_episode = episode

    # ── live rolling stats ──────────────────────────────────────────────

    def rolling_success_rate(self) -> Optional[float]:
        """Rolling mean success over last `window` episodes, or None if < window recorded."""
        if len(self._records) < self.window:
            return None
        recent = self._records[-self.window:]
        return float(np.mean([r.success for r in recent]))

    def rolling_mean_reward(self) -> Optional[float]:
        """Rolling mean total reward over last `window` episodes."""
        if len(self._records) < self.window:
            return None
        recent = self._records[-self.window:]
        return float(np.mean([r.total_reward for r in recent]))

    def rolling_mean_steps(self) -> Optional[float]:
        """Rolling mean steps over last `window` episodes."""
        if len(self._records) < self.window:
            return None
        recent = self._records[-self.window:]
        return float(np.mean([r.steps for r in recent]))

    # ── full arrays (for plotting) ──────────────────────────────────────

    @property
    def rewards(self) -> np.ndarray:
        return np.array([r.total_reward for r in self._records])

    @property
    def steps_arr(self) -> np.ndarray:
        return np.array([r.steps for r in self._records])

    @property
    def success_arr(self) -> np.ndarray:
        return np.array([r.success for r in self._records], dtype=float)

    @property
    def episodes_arr(self) -> np.ndarray:
        return np.array([r.episode for r in self._records])

    def rolling_rewards(self) -> np.ndarray:
        """Compute the rolling mean reward for every episode (returns NaN for first window-1)."""
        return _rolling_mean(self.rewards, self.window)

    def rolling_success(self) -> np.ndarray:
        """Compute the rolling success rate for every episode."""
        return _rolling_mean(self.success_arr, self.window)

    # ── final summary ───────────────────────────────────────────────────

    def compute_summary(self) -> dict:
        """
        Compute all paper metrics.

        Returns a dict with keys matching the comparison table:
            success_rate, mean_return, mean_steps_to_goal,
            convergence_episode, sample_efficiency, path_optimality,
            std_return, std_steps
        """
        n = len(self._records)
        if n == 0:
            return {}

        # Use last `window` episodes for final metrics
        last = self._records[-min(self.window, n):]

        successes     = [r for r in last if r.success]
        success_rate  = len(successes) / len(last)
        mean_return   = float(np.mean([r.total_reward for r in last]))
        std_return    = float(np.std([r.total_reward  for r in last]))

        if successes:
            steps_success  = [r.steps for r in successes]
            mean_steps     = float(np.mean(steps_success))
            std_steps      = float(np.std(steps_success))
        else:
            mean_steps = std_steps = float("nan")

        # Path optimality: optimal / actual   (1.0 = perfect)
        if self.optimal_path_len > 0 and not np.isnan(mean_steps):
            path_optimality = self.optimal_path_len / mean_steps
        else:
            path_optimality = float("nan")

        # Sample efficiency = convergence_episode × avg_steps_per_episode
        if self._convergence_episode is not None:
            # Use all records up to convergence
            conv_idx = self._convergence_episode
            conv_records = [r for r in self._records if r.episode <= conv_idx]
            avg_steps_to_conv = float(np.mean([r.steps for r in conv_records]))
            sample_efficiency = conv_idx * avg_steps_to_conv
        else:
            sample_efficiency = float("nan")

        return {
            "success_rate"        : round(success_rate,   4),
            "mean_return"         : round(mean_return,    4),
            "std_return"          : round(std_return,     4),
            "mean_steps_to_goal"  : round(mean_steps,     2),
            "std_steps_to_goal"   : round(std_steps,      2),
            "convergence_episode" : self._convergence_episode,
            "sample_efficiency"   : (round(sample_efficiency, 0)
                                     if not np.isnan(sample_efficiency) else None),
            "path_optimality"     : (round(path_optimality, 4)
                                     if not np.isnan(path_optimality) else None),
            "optimal_path_len"    : self.optimal_path_len,
            "total_episodes"      : n,
        }

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return (
            f"MetricsTracker("
            f"episodes={len(self)}, "
            f"success={self.rolling_success_rate()}, "
            f"converged_at={self._convergence_episode})"
        )


# ──────────────────────── utility ──────────────────────────────────────────

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling mean using np.convolve. Returns NaN for first window-1 values."""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < window:
        return result
    kernel = np.ones(window) / window
    valid  = np.convolve(arr, kernel, mode="valid")
    result[window - 1:] = valid
    return result