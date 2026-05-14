"""Learning-curve plots for Phase 1 Q-learning runs."""

from __future__ import annotations

import numpy as np


def rolling_mean(values, window: int = 100) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    if len(arr) < window:
        return out
    out[window - 1 :] = np.convolve(arr, np.ones(window) / window, mode="valid")
    return out


def plot_learning_curve(rewards, window: int = 100, title: str = "Q-learning return"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rolling_mean(rewards, window), label=f"rolling {window}-episode mean")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average return")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_success_curve(success, window: int = 100, title: str = "Q-learning success rate"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rolling_mean(success, window) * 100.0, label=f"rolling {window}-episode success")
    ax.axhline(90, color="tab:red", linestyle="--", linewidth=1, label="90% solved")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig
