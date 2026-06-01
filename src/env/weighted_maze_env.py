import numpy as np

try:
    import gymnasium as gym
except ImportError:
    pass

from .maze_env import MazeEnv


class WeightedMazeEnv(MazeEnv):
    """
    Weighted-Cost Maze Environment.
    
    In this maze, certain free cells are designated as 'mud' (costly terrain).
    Stepping on these cells incurs an additional penalty on top of the standard step penalty.
    
    In `self.maze`:
      0 = free path
      1 = wall
      2 = mud
      
    Observation space encoding:
      0.0 → wall
      1.0 → free path
      2.0 → agent
      3.0 → goal
      4.0 → mud (weighted cell)
    """

    def __init__(
        self,
        size: int = 10,
        seed: int | None = None,
        level: str | None = None,
        difficulty: str = "medium",
        reward_fn = None,
        render_mode: str | None = None,
        mud_density: float = 0.2,
        mud_penalty: float = 0.05,
    ):
        super().__init__(
            size=size,
            seed=seed,
            level=level,
            difficulty=difficulty,
            reward_fn=reward_fn,
            render_mode=render_mode,
        )

        try:
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=4.0,
                shape=(self.size * self.size,),
                dtype=np.float32,
            )
        except NameError:
            pass  # Fallback for systems without gymnasium

        self.mud_penalty = mud_penalty

        # Initialize mud cells
        rng = np.random.default_rng(seed)
        free_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.maze[r, c] == 0 and (r, c) != self.start_pos and (r, c) != self.goal_pos:
                    free_cells.append((r, c))

        # Randomly choose free cells to become mud
        num_mud = int(len(free_cells) * mud_density)
        if num_mud > 0:
            mud_choices = rng.choice(free_cells, size=num_mud, replace=False)
            for r, c in mud_choices:
                self.maze[r, c] = 2

    def _is_free(self, r: int, c: int) -> bool:
        """True if (r,c) is inside the grid and not a wall. Mud (2) is traversable."""
        return (
            0 <= r < self.size
            and 0 <= c < self.size
            and self.maze[r, c] != 1
        )

    def _get_obs(self) -> np.ndarray:
        """
        Build the flat observation vector.
        0.0=wall, 1.0=free, 2.0=agent, 3.0=goal, 4.0=mud
        """
        grid = np.zeros_like(self.maze, dtype=np.float32)
        grid[self.maze == 0] = 1.0
        grid[self.maze == 1] = 0.0
        grid[self.maze == 2] = 4.0

        r, c = self.agent_pos
        grid[r, c] = 2.0
        gr, gc = self.goal_pos
        grid[gr, gc] = 3.0
        return grid.flatten()

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        # Apply additional penalty if the agent stepped onto a mud cell
        if not info["hit_wall"] and self.maze[self.agent_pos[0], self.agent_pos[1]] == 2:
            reward -= self.mud_penalty

        return obs, reward, terminated, truncated, info

    @property
    def mud_cells(self):
        """Return a list of mud cell positions for visualization."""
        cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.maze[r, c] == 2:
                    cells.append((r, c))
        return cells

    def _render_rgb(self) -> np.ndarray:
        px  = self._CELL_PX
        H   = W = self.size * px
        img = np.zeros((H, W, 3), dtype=np.uint8)
        
        colors = dict(self._COLORS)
        colors["mud"] = (139, 69, 19)  # saddlebrown
        
        for r in range(self.size):
            for c in range(self.size):
                cell_val = self.maze[r, c]
                color = colors["free"]
                if cell_val == 1:
                    color = colors["wall"]
                elif cell_val == 2:
                    color = colors["mud"]
                    
                if (r, c) == self.start_pos:
                    color = colors["start"]
                if (r, c) == self.goal_pos:
                    color = colors["goal"]
                if (r, c) == self.agent_pos:
                    color = colors["agent"]
                    
                img[r*px:(r+1)*px, c*px:(c+1)*px] = color
                
        return img
