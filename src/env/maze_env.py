"""
maze_env.py
-----------
Custom Gymnasium environment for the RL maze research project.

Conforms fully to the Gymnasium API (gymnasium >= 0.26):
  - observation_space and action_space are declared in __init__
  - reset() returns (obs, info)
  - step() returns (obs, reward, terminated, truncated, info)

State representation
--------------------
A flat float32 vector of length N*N where each cell encodes:
    0.0  →  wall
    1.0  →  free path
    2.0  →  agent's current position
    3.0  →  goal position

This single vector is the input for Q-learning (index only),
DQN (neural net input), and PPO/SAC (MLP policy input).

Actions
-------
    0  Up     (-1,  0)
    1  Down   (+1,  0)
    2  Left   ( 0, -1)
    3  Right  ( 0, +1)
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only when gymnasium is absent
    class _Env:
        metadata = {}

        def reset(self, seed=None):
            return None

    class _Discrete:
        def __init__(self, n):
            self.n = n
            self._rng = np.random.default_rng()

        def contains(self, x):
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

        def sample(self):
            return int(self._rng.integers(self.n))

        def seed(self, seed=None):
            self._rng = np.random.default_rng(seed)

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Box = _Box
        Discrete = _Discrete

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()
from typing import Any

from .maze_generator import LEVEL_SIZES, generate_maze, optimal_path_length
from .rewards        import SparseWithPenalty, RewardFunction, StepInfo


# ──────────────────────── action definitions ───────────────────────────────

ACTIONS = {
    0: (-1,  0),   # Up
    1: ( 1,  0),   # Down
    2: ( 0, -1),   # Left
    3: ( 0,  1),   # Right
}
ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}


# ──────────────────────── environment ──────────────────────────────────────

class MazeEnv(gym.Env):
    """
    2-D discrete grid maze — Phase 1 (fully observable, static walls).

    Parameters
    ----------
    size           : int, side length of the maze
    level          : str | None, named size: tiny, small, medium, large
    difficulty     : str, obstacle density/complexity: easy, medium, hard
    seed           : int | None, seed for maze generation
    reward_fn      : RewardFunction, defaults to SparseWithPenalty()
    render_mode    : "human" | "rgb_array" | None
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        size       : int              = 10,
        seed       : int | None       = None,
        level      : str | None       = None,
        difficulty : str              = "medium",
        reward_fn  : RewardFunction   = None,
        render_mode: str | None       = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.reward_fn   = reward_fn or SparseWithPenalty()

        # ── Build maze ────────────────────────────────────────────────────
        if level is not None:
            if level not in LEVEL_SIZES:
                raise ValueError(f"Unknown level '{level}'. Available: {list(LEVEL_SIZES)}")
            size = LEVEL_SIZES[level]

        self.level = level
        self.difficulty = difficulty
        self.maze = generate_maze(size, seed=seed, difficulty=difficulty)
        self.size = self.maze.shape[0]

        # ── Fixed positions ───────────────────────────────────────────────
        self.start_pos = (0, 0)
        self.goal_pos  = (self.size - 1, self.size - 1)
        self.max_steps = 4 * self.size ** 2

        # ── BFS optimal path (used for metrics, not learning) ─────────────
        self._optimal = optimal_path_length(self.maze)

        # ── Gymnasium spaces ──────────────────────────────────────────────
        # Observation: flat vector, values in {0,1,2,3}
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 3.0,
            shape = (self.size * self.size,),
            dtype = np.float32,
        )
        # Action: 4 discrete directions
        self.action_space = spaces.Discrete(4)

        # ── Episode state (set properly in reset()) ───────────────────────
        self.agent_pos  = self.start_pos
        self.steps_taken = 0

        # ── Pygame surface (lazy-init on first render call) ───────────────
        self._screen = None
        self._clock  = None

    # ──────────────────────── Gymnasium API ──────────────────────────────

    def reset(
        self,
        *,
        seed     : int | None = None,
        options  : dict | None = None,
    ):
        """
        Reset the environment to the start state.

        Returns
        -------
        obs  : np.ndarray  flat grid observation
        info : dict        auxiliary info (optimal path, maze size, etc.)
        """
        super().reset(seed=seed)

        self.agent_pos   = self.start_pos
        self.steps_taken = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        """
        Take one step in the environment.

        Parameters
        ----------
        action : int  in {0,1,2,3}

        Returns
        -------
        obs        : np.ndarray
        reward     : float
        terminated : bool  — True if goal reached
        truncated  : bool  — True if max_steps exceeded
        info       : dict
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dr, dc = ACTIONS[action]
        nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        hit_wall = not self._is_free(nr, nc)

        prev_pos = self.agent_pos
        if not hit_wall:
            self.agent_pos = (nr, nc)

        self.steps_taken += 1
        reached_goal = self.agent_pos == self.goal_pos

        # ── Reward ────────────────────────────────────────────────────────
        step_info = StepInfo(
            prev_pos     = prev_pos,
            next_pos     = self.agent_pos,
            goal_pos     = self.goal_pos,
            hit_wall     = hit_wall,
            reached_goal = reached_goal,
            steps_taken  = self.steps_taken,
            max_steps    = self.max_steps,
            optimal_dist = self._optimal,
        )
        reward = self.reward_fn.compute(step_info)

        # ── Termination ───────────────────────────────────────────────────
        terminated = reached_goal
        truncated  = self.steps_taken >= self.max_steps

        obs  = self._get_obs()
        info = self._get_info(hit_wall=hit_wall, action=action)

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Render the maze.
        - "human"     → opens a pygame window
        - "rgb_array" → returns an H×W×3 uint8 numpy array
        """
        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    # ──────────────────────── helpers ────────────────────────────────────

    def _is_free(self, r: int, c: int) -> bool:
        """True if (r,c) is inside the grid and not a wall."""
        return (
            0 <= r < self.size
            and 0 <= c < self.size
            and self.maze[r, c] == 0
        )

    def _get_obs(self) -> np.ndarray:
        """
        Build the flat observation vector.

        Encoding:
            1.0 = free cell  (wall=0 in maze array → 1.0 here, inverted)
            0.0 = wall       (maze array == 1)
            2.0 = agent
            3.0 = goal
        """
        # Free cells = 1.0, walls = 0.0
        grid = (1.0 - self.maze).astype(np.float32)
        # Overlay agent and goal
        r, c = self.agent_pos
        grid[r, c] = 2.0
        gr, gc = self.goal_pos
        grid[gr, gc] = 3.0
        return grid.flatten()

    def _get_info(self, hit_wall: bool = False, action: int | None = None) -> dict:
        """Auxiliary info dict — never used for learning, useful for debugging."""
        return {
            "agent_pos"       : self.agent_pos,
            "goal_pos"        : self.goal_pos,
            "steps_taken"     : self.steps_taken,
            "optimal_path_len": self._optimal,
            "hit_wall"        : hit_wall,
            "action_name"     : ACTION_NAMES.get(action, "N/A"),
            "maze_size"       : self.size,
            "level"           : self.level,
            "difficulty"      : self.difficulty,
        }

    def pos_to_state_idx(self, pos: tuple | None = None) -> int:
        """
        Convert a (row, col) position to a scalar state index.
        Used by Q-learning for table lookup.

        If pos is None, uses self.agent_pos.
        """
        r, c = pos if pos is not None else self.agent_pos
        return r * self.size + c

    # ──────────────────────── rendering internals ─────────────────────────

    _COLORS = {
        "wall"  : (30,  30,  46),   # dark charcoal
        "free"  : (205, 214, 244),  # light lavender
        "agent" : (137, 220, 235),  # teal-blue
        "goal"  : (166, 227, 161),  # mint green
        "start" : (243, 139, 168),  # soft pink
    }
    _CELL_PX = 32  # pixels per cell for human rendering

    def _render_rgb(self) -> np.ndarray:
        """Return an H×W×3 uint8 array — no window required."""
        px  = self._CELL_PX
        H   = W = self.size * px
        img = np.zeros((H, W, 3), dtype=np.uint8)

        for r in range(self.size):
            for c in range(self.size):
                y, x = r * px, c * px
                if (r, c) == self.agent_pos:
                    color = self._COLORS["agent"]
                elif (r, c) == self.goal_pos:
                    color = self._COLORS["goal"]
                elif (r, c) == self.start_pos:
                    color = self._COLORS["start"]
                elif self.maze[r, c] == 1:
                    color = self._COLORS["wall"]
                else:
                    color = self._COLORS["free"]
                img[y:y+px, x:x+px] = color

        return img

    def _render_pygame(self) -> None:
        """Open (or update) a pygame window."""
        import pygame

        px  = self._CELL_PX
        H   = W = self.size * px

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("RL Maze")
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        frame = self._render_rgb()
        # numpy H×W×3 → pygame Surface
        surf  = pygame.surfarray.make_surface(
            np.transpose(frame, (1, 0, 2))
        )
        self._screen.blit(surf, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
