"""Q-table heatmap visualization."""

from __future__ import annotations

import numpy as np


def plot_q_heatmap(q_table: np.ndarray, maze: np.ndarray, title: str = "Q-value heatmap"):
    import matplotlib.pyplot as plt

    size = maze.shape[0]
    max_q = q_table.max(axis=1).reshape(size, size)
    max_q = max_q.astype(float)
    max_q[maze == 1] = np.nan

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(max_q, cmap="RdYlGn", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Max Q-value")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig
