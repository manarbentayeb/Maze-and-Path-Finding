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

def plot_q_heatmap_weighted(q_table: np.ndarray, maze: np.ndarray, title: str = "Q-value heatmap"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    fig = plot_q_heatmap(q_table, maze, title)
    ax = fig.axes[0]
    
    # Overlay hatch pattern on mud cells (value 2 in weighted maze)
    size = maze.shape[0]
    for r in range(size):
        for c in range(size):
            if maze[r, c] == 2:
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5), 1.0, 1.0,
                        fill=False, hatch='//', edgecolor='black', alpha=0.3, zorder=10
                    )
                )
    return fig
