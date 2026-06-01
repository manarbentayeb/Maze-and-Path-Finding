import numpy as np
from .maze_env import MazeEnv

class DynamicMazeEnv(MazeEnv):
    """
    Dynamic Obstacle Maze Environment.
    
    In this maze, walls can dynamically appear or disappear probabilistically
    at each step, simulating a changing environment. We protect the start,
    goal, and agent's current position from being overwritten.
    """

    def __init__(
        self,
        size: int = 10,
        seed: int | None = None,
        level: str | None = None,
        difficulty: str = "medium",
        reward_fn = None,
        render_mode: str | None = None,
        dynamic_prob: float = 0.05,
    ):
        super().__init__(
            size=size,
            seed=seed,
            level=level,
            difficulty=difficulty,
            reward_fn=reward_fn,
            render_mode=render_mode,
        )
        self.dynamic_prob = dynamic_prob
        self.rng = np.random.default_rng(seed)
        self._base_maze = self.maze.copy()
        self.dynamic_changes_total = 0
        self.dynamic_changes_this_episode = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        obs, info = super().reset(seed=seed, options=options)
        self.maze = self._base_maze.copy()
        self.dynamic_changes_this_episode = 0
        return obs, info

    def step(self, action: int):
        # Regular step transition
        obs, reward, terminated, truncated, info = super().step(action)
        
        if self.rng.random() < self.dynamic_prob:
            for _ in range(5):
                r, c = self.rng.integers(0, self.size, size=2)
                if (r, c) not in [self.start_pos, self.goal_pos, self.agent_pos]:
                    original_val = self.maze[r, c]
                    self.maze[r, c] = 1 - original_val
                    
                    if self._is_solvable():
                        self.dynamic_changes_total += 1
                        self.dynamic_changes_this_episode += 1
                        break
                    else:
                        self.maze[r, c] = original_val

        obs = self._get_obs()
        info["dynamic_changes_total"] = self.dynamic_changes_total
        info["dynamic_changes_this_episode"] = self.dynamic_changes_this_episode
        return obs, reward, terminated, truncated, info

    def _is_solvable(self) -> bool:
        """Check if there is still a path from start to goal."""
        from collections import deque
        if self.maze[self.start_pos] == 1 or self.maze[self.goal_pos] == 1:
            return False
            
        q = deque([self.start_pos])
        seen = {self.start_pos}
        
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal_pos:
                return True
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.maze[nr, nc] == 0 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
        return False
