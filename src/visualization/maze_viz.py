"""Matplotlib maze visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.env.maze_generator import shortest_path

def plot_weighted_maze(
    env: Any,
    path: Iterable[GridPos] | None = None,
    *,
    show_solution: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Draw a weighted maze showing mud cells."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
    fig, ax = plot_maze(
        env, path=path, show_solution=show_solution, title=title, show=False
    )
    
    # Draw mud cells
    if hasattr(env, "mud_cells"):
        for r, c in env.mud_cells:
            _draw_cell(ax, (r, c), "#8B4513", zorder=2) # SaddleBrown for mud
            
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(2, Patch(facecolor="#8B4513", label="Mud"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.035), ncol=3, frameon=False)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
    if show:
        plt.show()
        
    return fig, ax

def plot_dynamic_maze(
    env: Any,
    path: Iterable[GridPos] | None = None,
    *,
    show_solution: bool = True,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Draw a dynamic maze snapshot."""
    return plot_maze(env, path=path, show_solution=show_solution, title=title, save_path=save_path, show=show)



GridPos = tuple[int, int]


def plot_maze(
    maze_or_env: Any,
    path: Iterable[GridPos] | None = None,
    *,
    show_solution: bool = True,
    start_pos: GridPos | None = None,
    goal_pos: GridPos | None = None,
    title: str | None = None,
    ax=None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """
    Draw a maze with walls, rooms, start/goal cells, and an optional path.

    ``maze_or_env`` can be either a raw maze array from ``generate_maze`` or a
    ``MazeEnv`` instance. In the maze array, ``1`` means wall and ``0`` means
    free cell.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    maze = _maze_array(maze_or_env)
    size = maze.shape[0]
    start = start_pos or getattr(maze_or_env, "start_pos", (0, 0))
    goal = goal_pos or getattr(maze_or_env, "goal_pos", (size - 1, size - 1))

    if path is None and show_solution:
        path = shortest_path(maze)
    path_points = list(path or [])

    if ax is None:
        fig, ax = plt.subplots(figsize=_figure_size(size))
    else:
        fig = ax.figure

    display_grid = np.where(maze == 1, 0, 1)
    cmap = ListedColormap(["#3b3b3b", "#ffffff"])
    ax.imshow(display_grid, cmap=cmap, interpolation="none", vmin=0, vmax=1)

    # Thin cell borders make rectangular rooms and corridor boundaries easy to read.
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linewidth=0.55)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")

    if path_points:
        ys = [r for r, _ in path_points]
        xs = [c for _, c in path_points]
        line_width = max(2.0, min(5.0, 80.0 / size))
        marker_size = max(18.0, min(70.0, 900.0 / size))
        ax.plot(
            xs,
            ys,
            color="#f28c28",
            linewidth=line_width,
            solid_capstyle="round",
            zorder=3,
            label="Solution path",
        )
        ax.scatter(xs, ys, s=marker_size, color="#f28c28", edgecolors="none", zorder=3)

    _draw_cell(ax, start, "#1f77ff", zorder=4)
    _draw_cell(ax, goal, "#2ca02c", zorder=4)

    for label, pos in [("S", start), ("G", goal)]:
        ax.text(
            pos[1],
            pos[0],
            label,
            color="#ffffff",
            ha="center",
            va="center",
            fontsize=max(8, min(14, 140 // size)),
            fontweight="bold",
            zorder=5,
        )

    ax.legend(
        handles=[
            Patch(facecolor="#3b3b3b", label="Wall"),
            Patch(facecolor="#ffffff", edgecolor="#bdbdbd", label="Free / room"),
            Patch(facecolor="#1f77ff", label="Start"),
            Patch(facecolor="#2ca02c", label="Goal"),
            Patch(facecolor="#f28c28", label="Solution path"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=3,
        frameon=False,
    )

    if title:
        ax.set_title(title, pad=10)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def show_maze(
    maze_or_env: Any,
    path: Iterable[GridPos] | None = None,
    **kwargs,
):
    """Display a maze immediately using matplotlib."""
    kwargs["show"] = True
    return plot_maze(maze_or_env, path=path, **kwargs)


def _maze_array(maze_or_env: Any) -> np.ndarray:
    maze = getattr(maze_or_env, "maze", maze_or_env)
    arr = np.asarray(maze)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Expected a square 2-D maze array or a MazeEnv instance.")
    return arr.astype(np.int32, copy=False)


def _draw_cell(ax, pos: GridPos, color: str, zorder: int) -> None:
    from matplotlib.patches import Rectangle

    r, c = pos
    ax.add_patch(
        Rectangle(
            (c - 0.5, r - 0.5),
            1.0,
            1.0,
            facecolor=color,
            edgecolor="#ffffff",
            linewidth=1.2,
            zorder=zorder,
        )
    )


def _figure_size(size: int) -> tuple[float, float]:
    side = max(5.0, min(10.0, size * 0.45))
    return side, side
