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

def plot_policy_arrows_weighted(q_table: np.ndarray, maze: np.ndarray, title: str = "Q-learning policy"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # We treat mud (2) just like free space (0) for plotting arrows
    maze_binary = maze.copy()
    maze_binary[maze_binary == 2] = 0
    
    fig = plot_policy_arrows(q_table, maze_binary, title)
    ax = fig.axes[0]
    
    # Overlay color for mud cells
    size = maze.shape[0]
    for r in range(size):
        for c in range(size):
            if maze[r, c] == 2:
                ax.add_patch(
                    Rectangle(
                        (c - 0.5, r - 0.5), 1.0, 1.0,
                        facecolor="#8B4513", alpha=0.3, zorder=1
                    )
                )
    return fig
