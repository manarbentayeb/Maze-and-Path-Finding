"""Maze generation utilities for the Phase 1 direct-MDP environment.

The first version used DFS carving plus a border repair path, which could make
some mazes too easy. This generator creates exact-size obstacle mazes closer to
the grid in the research sketches: free cells mixed with black obstacle blocks,
while still guaranteeing that start and goal are connected.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


LEVEL_SIZES = {
    "tiny": 8,
    "small": 10,
    "medium": 20,
    "large": 50,
}


@dataclass(frozen=True)
class DifficultySpec:
    wall_density: float
    min_path_factor: float
    block_attempt_factor: float


DIFFICULTY_SPECS = {
    "easy": DifficultySpec(wall_density=0.18, min_path_factor=1.20, block_attempt_factor=0.40),
    "medium": DifficultySpec(wall_density=0.28, min_path_factor=1.55, block_attempt_factor=0.70),
    "hard": DifficultySpec(wall_density=0.36, min_path_factor=1.90, block_attempt_factor=1.00),
}


def generate_level_maze(
    level: str = "small",
    seed: int | None = None,
    difficulty: str = "medium",
) -> np.ndarray:
    """Generate a maze by named level: tiny, small, medium, or large."""
    if level not in LEVEL_SIZES:
        raise ValueError(f"Unknown level '{level}'. Available: {list(LEVEL_SIZES)}")
    return generate_maze(size=LEVEL_SIZES[level], seed=seed, difficulty=difficulty)


def generate_maze(
    size: int,
    seed: int | None = None,
    difficulty: str = "medium",
    max_attempts: int = 300,
) -> np.ndarray:
    """
    Generate an exact ``size x size`` binary obstacle maze.

    The generator samples wall blocks and individual obstacles, then accepts a
    candidate only when:

    - start and goal are free
    - a path exists from start to goal
    - the shortest path is not too close to the trivial Manhattan route

    If no candidate reaches the target complexity, the longest solvable
    candidate found is returned.
    """
    if size < 5:
        raise ValueError("Maze size must be at least 5.")
    if difficulty not in DIFFICULTY_SPECS:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Available: {list(DIFFICULTY_SPECS)}")

    rng = np.random.default_rng(seed)
    spec = DIFFICULTY_SPECS[difficulty]
    min_path = int((2 * (size - 1)) * spec.min_path_factor)

    best_maze: np.ndarray | None = None
    best_path_len = -1

    for _ in range(max_attempts):
        maze = _sample_obstacle_maze(size=size, rng=rng, spec=spec)
        path_len = optimal_path_length(maze)

        if path_len > best_path_len:
            best_maze = maze
            best_path_len = path_len

        if path_len >= min_path:
            return maze

    if best_maze is not None and best_path_len > 0:
        return best_maze

    return _snake_fallback(size)


def get_free_cells(maze: np.ndarray) -> List[Tuple[int, int]]:
    """Return all free ``(row, col)`` cells."""
    rows, cols = np.where(maze == 0)
    return list(zip(rows.tolist(), cols.tolist()))


def optimal_path_length(maze: np.ndarray) -> int:
    """Return shortest start-to-goal path length in steps, or -1."""
    path = shortest_path(maze)
    return len(path) - 1 if path else -1


def shortest_path(maze: np.ndarray) -> list[tuple[int, int]]:
    """Return one shortest path from start to goal using BFS."""
    size = maze.shape[0]
    start = (0, 0)
    goal = (size - 1, size - 1)

    if maze[start] == 1 or maze[goal] == 1:
        return []

    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return _reconstruct_path(parent, goal)

        for nr, nc in _neighbors(r, c, size):
            if maze[nr, nc] == 0 and (nr, nc) not in parent:
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return []


def _sample_obstacle_maze(
    size: int,
    rng: np.random.Generator,
    spec: DifficultySpec,
) -> np.ndarray:
    maze = np.zeros((size, size), dtype=np.int32)
    protected = _protected_cells(size)

    block_attempts = max(1, int(size * size * spec.block_attempt_factor / 10))
    for _ in range(block_attempts):
        _place_random_block(maze, rng, protected)

    current_density = float(np.mean(maze))
    if current_density < spec.wall_density:
        missing = spec.wall_density - current_density
        random_walls = rng.random((size, size)) < missing
        for r, c in protected:
            random_walls[r, c] = False
        maze[random_walls] = 1

    _open_protected_cells(maze, protected)
    return maze


def _place_random_block(
    maze: np.ndarray,
    rng: np.random.Generator,
    protected: set[tuple[int, int]],
) -> None:
    size = maze.shape[0]
    max_side = max(2, min(5, size // 4))
    height = int(rng.integers(1, max_side + 1))
    width = int(rng.integers(1, max_side + 1))
    row = int(rng.integers(0, size - height + 1))
    col = int(rng.integers(0, size - width + 1))

    for r in range(row, row + height):
        for c in range(col, col + width):
            if (r, c) not in protected:
                maze[r, c] = 1


def _protected_cells(size: int) -> set[tuple[int, int]]:
    cells = {
        (0, 0),
        (size - 1, size - 1),
        (0, 1),
        (1, 0),
        (size - 2, size - 1),
        (size - 1, size - 2),
    }
    return {(r, c) for r, c in cells if 0 <= r < size and 0 <= c < size}


def _open_protected_cells(maze: np.ndarray, protected: set[tuple[int, int]]) -> None:
    for r, c in protected:
        maze[r, c] = 0


def _neighbors(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def _snake_fallback(size: int) -> np.ndarray:
    """Deterministic nontrivial fallback that always reaches the goal."""
    maze = np.ones((size, size), dtype=np.int32)

    for r in range(size):
        if r % 2 == 0:
            maze[r, :] = 0
        else:
            gap = size - 1 if (r // 2) % 2 == 0 else 0
            maze[r, gap] = 0

    maze[0, 0] = 0
    maze[size - 1, size - 1] = 0
    return maze