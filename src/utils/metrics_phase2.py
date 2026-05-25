"""Phase 2 metric helpers for door/key experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metrics import EpisodeMetrics, MetricsTracker


@dataclass
class DoorKeyEpisodeMetrics(EpisodeMetrics):
    picked_up_key: bool
    opened_door: bool
    steps_to_key: Optional[int] = None
    steps_to_door: Optional[int] = None
    optimal_path_len: Optional[int] = None


class DoorKeyMetricsTracker(MetricsTracker):
    """Metrics tracker extended with Phase 2 subgoal completion rates."""

    def __init__(self, window: int = 100, optimal_path_len: int = 0):
        super().__init__(window=window, optimal_path_len=optimal_path_len)
        self._records: list[DoorKeyEpisodeMetrics] = []

    def record(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        success: bool,
        picked_up_key: bool,
        opened_door: bool,
        steps_to_key: int | None = None,
        steps_to_door: int | None = None,
        optimal_path_len: int | None = None,
    ) -> None:
        rec = DoorKeyEpisodeMetrics(
            episode=episode,
            total_reward=total_reward,
            steps=steps,
            success=success,
            picked_up_key=picked_up_key,
            opened_door=opened_door,
            steps_to_key=steps_to_key,
            steps_to_door=steps_to_door,
            optimal_path_len=optimal_path_len,
        )
        self._records.append(rec)

        if self._convergence_episode is None:
            rate = self.rolling_success_rate()
            if rate is not None and rate >= 0.90:
                self._convergence_episode = episode

    @property
    def key_arr(self) -> np.ndarray:
        return np.array([r.picked_up_key for r in self._records], dtype=float)

    @property
    def door_arr(self) -> np.ndarray:
        return np.array([r.opened_door for r in self._records], dtype=float)

    @property
    def steps_to_key_arr(self) -> np.ndarray:
        return np.array(
            [np.nan if r.steps_to_key is None else r.steps_to_key for r in self._records],
            dtype=float,
        )

    @property
    def steps_to_door_arr(self) -> np.ndarray:
        return np.array(
            [np.nan if r.steps_to_door is None else r.steps_to_door for r in self._records],
            dtype=float,
        )

    @property
    def optimal_path_arr(self) -> np.ndarray:
        return np.array(
            [np.nan if r.optimal_path_len is None else r.optimal_path_len for r in self._records],
            dtype=float,
        )

    def rolling_key_rate(self) -> np.ndarray:
        from .metrics import _rolling_mean

        return _rolling_mean(self.key_arr, self.window)

    def rolling_door_rate(self) -> np.ndarray:
        from .metrics import _rolling_mean

        return _rolling_mean(self.door_arr, self.window)

    def compute_summary(self) -> dict:
        summary = super().compute_summary()
        if not self._records:
            return summary

        last = self._records[-min(self.window, len(self._records)) :]
        successes = [r for r in last if r.success]
        key_steps = [r.steps_to_key for r in last if r.steps_to_key is not None]
        door_steps = [
            r.steps_to_door - r.steps_to_key
            for r in last
            if r.steps_to_key is not None and r.steps_to_door is not None
        ]
        optimal_lengths = [r.optimal_path_len for r in last if r.optimal_path_len is not None]
        successful_optimality = [
            r.optimal_path_len / r.steps
            for r in successes
            if r.optimal_path_len is not None and r.steps > 0
        ]

        summary.update(
            {
                "key_pickup_rate": round(float(np.mean([r.picked_up_key for r in last])), 4),
                "door_opening_rate": round(float(np.mean([r.opened_door for r in last])), 4),
                "subgoal_order_rate": round(float(np.mean([r.picked_up_key and r.opened_door for r in last])), 4),
                "mean_steps_to_key": round(float(np.mean(key_steps)), 2) if key_steps else None,
                "mean_key_to_door_steps": round(float(np.mean(door_steps)), 2) if door_steps else None,
                "mean_optimal_path_len": round(float(np.mean(optimal_lengths)), 2) if optimal_lengths else None,
                "path_optimality": round(float(np.mean(successful_optimality)), 4) if successful_optimality else None,
            }
        )
        return summary


def bootstrap_ci(values, statistic=np.mean, n_resamples: int = 1000, ci: float = 0.95, seed: int = 0):
    """Return a percentile bootstrap confidence interval."""
    arr = np.asarray(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats.append(statistic(sample))

    alpha = (1.0 - ci) / 2.0
    return tuple(np.quantile(stats, [alpha, 1.0 - alpha]).tolist())


def cohens_d(a, b) -> float:
    """Cohen's d effect size between two samples."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = np.sqrt(((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2))
    return float((np.mean(x) - np.mean(y)) / pooled) if pooled > 0 else float("nan")
