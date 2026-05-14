"""Policy arrow map visualization for Q-learning."""

from __future__ import annotations

import numpy as np


def plot_policy_arrows(q_table: np.ndarray, maze: np.ndarray, title: str = "Q-learning policy"):
    import matplotlib.pyplot as plt

    size = maze.shape[0]
    dy = [-0.35, 0.35, 0.0, 0.0]
    dx = [0.0, 0.0, -0.35, 0.35]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(maze, cmap="binary")

    for r in range(size):
        for c in range(size):
            if maze[r, c] == 0:
                state = r * size + c
                action = int(np.argmax(q_table[state]))
                ax.arrow(
                    c,
                    r,
                    dx[action],
                    dy[action],
                    head_width=0.18,
                    head_length=0.12,
                    color="tab:red",
                    length_includes_head=True,
                )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig
