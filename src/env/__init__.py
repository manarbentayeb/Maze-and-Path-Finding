"""src/env — maze environment package."""

from .maze_env import MazeEnv, ACTIONS, ACTION_NAMES

from .maze_generator import (
    LEVEL_SIZES,
    DIFFICULTY_SPECS,
    generate_maze,
    generate_level_maze,
    optimal_path_length,
    shortest_path,
    get_free_cells,
)

from .rewards import (
    SparseWithPenalty,
    RewardFunction,
    StepInfo,
    get_reward_function,
)

__all__ = [
    "MazeEnv",
    "ACTIONS",
    "ACTION_NAMES",
    "LEVEL_SIZES",
    "DIFFICULTY_SPECS",
    "generate_maze",
    "generate_level_maze",
    "optimal_path_length",
    "shortest_path",
    "get_free_cells",
    "SparseWithPenalty",
    "RewardFunction",
    "StepInfo",
    "get_reward_function",
]