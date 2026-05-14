"""Path animation helpers for trained Q-learning agents."""

from __future__ import annotations

import numpy as np


def animate_path(maze: np.ndarray, path: list[tuple[int, int]], interval: int = 120):
    """Create a matplotlib animation for a path through the maze."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(maze, cmap="binary")
    ax.set_xticks([])
    ax.set_yticks([])
    point, = ax.plot([], [], "o", color="tab:blue", markersize=10)
    line, = ax.plot([], [], color="tab:blue", linewidth=2, alpha=0.6)

    xs: list[int] = []
    ys: list[int] = []

    def update(frame: int):
        r, c = path[frame]
        xs.append(c)
        ys.append(r)
        point.set_data([c], [r])
        line.set_data(xs, ys)
        return point, line

    anim = FuncAnimation(fig, update, frames=len(path), interval=interval, blit=True)
    fig.tight_layout()
    return anim
