"""src/env — maze environment package."""

from .maze_env import MazeEnv, ACTIONS, ACTION_NAMES
from .door_key_maze_env import (
    AGENT,
    DOOR_LOCKED,
    DOOR_OPEN,
    FREE,
    GOAL,
    KEY,
    REWARD_TYPES,
    WALL,
    DoorKeyMazeEnv,
)

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
    "DoorKeyMazeEnv",
    "ACTIONS",
    "ACTION_NAMES",
    "REWARD_TYPES",
    "FREE",
    "WALL",
    "AGENT",
    "GOAL",
    "KEY",
    "DOOR_LOCKED",
    "DOOR_OPEN",
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
