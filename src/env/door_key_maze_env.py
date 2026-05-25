"""Door/key maze environment for Phase 2.

This module mirrors the Phase 1 ``MazeEnv`` API while adding two state flags:
``has_key`` and ``door_open``. Maze size and difficulty remain parameterized by
the same ``level`` and ``difficulty`` options used in Phase 1.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    from .maze_env import gym, spaces

from .maze_generator import LEVEL_SIZES, generate_maze
from .maze_env import ACTIONS, ACTION_NAMES


FREE = 0
WALL = 1
AGENT = 2
GOAL = 3
KEY = 4
DOOR_LOCKED = 5
DOOR_OPEN = 6

REWARD_TYPES = ("sparse", "subgoal", "potential")


class DoorKeyMazeEnv(gym.Env):
    """Phase 2 discrete maze with automatic key pickup and door opening.

    Observation is a flat ``float32`` vector:
    ``size * size`` grid cells plus ``[has_key, door_open]``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        size: int = 10,
        seed: int | None = None,
        level: str | None = None,
        difficulty: str = "medium",
        reward_type: str = "sparse",
        gamma: float = 0.99,
        render_mode: str | None = None,
        max_generation_attempts: int = 500,
        dynamic_objects: bool = True,
        n_dynamic_layouts: int = 4,
    ):
        super().__init__()

        if level is not None:
            if level not in LEVEL_SIZES:
                raise ValueError(f"Unknown level '{level}'. Available: {list(LEVEL_SIZES)}")
            size = LEVEL_SIZES[level]
        if reward_type not in REWARD_TYPES:
            raise ValueError(f"Unknown reward_type '{reward_type}'. Available: {list(REWARD_TYPES)}")

        self.size = size
        self.level = level
        self.difficulty = difficulty
        self.reward_type = reward_type
        self.gamma = gamma
        self.render_mode = render_mode
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.max_steps = 4 * size**2
        self._seed = seed
        self.dynamic_objects = dynamic_objects
        self.n_dynamic_layouts = max(1, n_dynamic_layouts)
        self._layout_rng = np.random.default_rng(seed)
        self._layout_id = 0

        self._layouts = self._build_validated_layouts(
            size=size,
            seed=seed,
            difficulty=difficulty,
            max_attempts=max_generation_attempts,
            n_layouts=self.n_dynamic_layouts if dynamic_objects else 1,
        )
        self._activate_layout(0)

        obs_size = size * size + 2
        self.observation_space = spaces.Box(low=0.0, high=6.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.has_key = 0
        self.door_open = 0
        self.picked_up_key = 0
        self.opened_door = 0
        self.wall_collisions = 0
        self.steps_to_key: int | None = None
        self.steps_to_door: int | None = None
        self._screen = None
        self._clock = None
        self._phi_prev = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._layout_rng = np.random.default_rng(seed)
        if self.dynamic_objects and len(self._layouts) > 1:
            layout_id = int(self._layout_rng.integers(len(self._layouts)))
            self._activate_layout(layout_id)
        self.maze = self.base_maze.copy()
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self.has_key = 0
        self.door_open = 0
        self.picked_up_key = 0
        self.opened_door = 0
        self.wall_collisions = 0
        self.steps_to_key = None
        self.steps_to_door = None
        self._phi_prev = self._phi(self.agent_pos, self.has_key, self.door_open)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_pos = self.agent_pos
        prev_phi = self._phi(prev_pos, self.has_key, self.door_open)
        dr, dc = ACTIONS[action]
        nr, nc = prev_pos[0] + dr, prev_pos[1] + dc
        self.steps_taken += 1

        rejected = False
        picked_key_now = False
        opened_door_now = False

        target = self.maze[nr, nc] if self._in_bounds(nr, nc) else WALL
        if not self._in_bounds(nr, nc) or target == WALL:
            rejected = True
            self.wall_collisions += 1
        elif target == DOOR_LOCKED:
            if self.has_key:
                self.maze[nr, nc] = DOOR_OPEN
                self.door_open = 1
                self.agent_pos = (nr, nc)
                self.opened_door = 1
                self.steps_to_door = self.steps_taken
                opened_door_now = True
            else:
                rejected = True
                self.wall_collisions += 1
        elif target == KEY:
            self.maze[nr, nc] = FREE
            self.has_key = 1
            self.agent_pos = (nr, nc)
            self.picked_up_key = 1
            self.steps_to_key = self.steps_taken
            picked_key_now = True
        else:
            self.agent_pos = (nr, nc)

        reached_goal = self.agent_pos == self.goal_pos
        reward = self._reward(
            rejected=rejected,
            picked_key=picked_key_now,
            opened_door=opened_door_now,
            reached_goal=reached_goal,
            prev_phi=prev_phi,
        )
        terminated = reached_goal
        truncated = self.steps_taken >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info(
            hit_wall=rejected,
            action=action,
            picked_key_now=picked_key_now,
            opened_door_now=opened_door_now,
        )

    def optimal_path_length(self) -> int:
        """Shortest valid route length for start -> key -> door -> goal."""
        parts = [
            self._bfs_dist_between(self.start_pos, self.key_pos, door_passable=True),
            self._bfs_dist_between(self.key_pos, self.door_pos, door_passable=True),
            self._bfs_dist_between(self.door_pos, self.goal_pos, door_passable=True),
        ]
        return int(sum(parts)) if all(p >= 0 for p in parts) else -1

    def pos_to_state_idx(self, pos: tuple[int, int] | None = None) -> int:
        """Encode ``(row, col, has_key, door_open)`` as one integer state."""
        r, c = pos if pos is not None else self.agent_pos
        flag_idx = self.has_key * 2 + self.door_open
        return (r * self.size + c) * 4 + flag_idx

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def close(self):
        if self._screen is not None:
            import pygame

            pygame.quit()
            self._screen = None

    def _reward(
        self,
        rejected: bool,
        picked_key: bool,
        opened_door: bool,
        reached_goal: bool,
        prev_phi: float,
    ) -> float:
        if self.reward_type == "sparse":
            if reached_goal:
                return 1.0
            if rejected:
                return -0.5
            return -0.01

        if self.reward_type == "subgoal":
            if reached_goal:
                return 1.0
            if picked_key:
                return 0.5
            if opened_door:
                return 0.3
            if rejected:
                return -0.5
            return -0.01

        base = 1.0 if reached_goal else (-0.5 if rejected else -0.01)
        phi_next = self._phi(self.agent_pos, self.has_key, self.door_open)
        self._phi_prev = phi_next
        return float(base + self.gamma * phi_next - prev_phi)

    def _phi(self, pos: tuple[int, int], has_key: int, door_open: int) -> float:
        r, c = pos
        if has_key == 0:
            dist = self._dist_key[r, c]
        elif door_open == 0:
            dist = self._dist_door[r, c]
        else:
            dist = self._dist_goal[r, c]
        if np.isinf(dist):
            return -float(self.size * self.size)
        return -float(dist)

    def _get_obs(self) -> np.ndarray:
        grid = self.maze.copy().astype(np.float32)
        grid[self.goal_pos] = GOAL
        grid[self.agent_pos] = AGENT
        flags = np.array([self.has_key, self.door_open], dtype=np.float32)
        return np.concatenate([grid.flatten(), flags])

    def _get_info(
        self,
        hit_wall: bool = False,
        action: int | None = None,
        picked_key_now: bool = False,
        opened_door_now: bool = False,
    ) -> dict:
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "key_pos": self.key_pos,
            "door_pos": self.door_pos,
            "has_key": self.has_key,
            "door_open": self.door_open,
            "picked_up_key": self.picked_up_key,
            "opened_door": self.opened_door,
            "picked_key_now": picked_key_now,
            "opened_door_now": opened_door_now,
            "wall_collisions": self.wall_collisions,
            "steps_to_key": self.steps_to_key,
            "steps_to_door": self.steps_to_door,
            "steps_taken": self.steps_taken,
            "optimal_path_len": self._optimal,
            "hit_wall": hit_wall,
            "action_name": ACTION_NAMES.get(action, "N/A"),
            "maze_size": self.size,
            "level": self.level,
            "difficulty": self.difficulty,
            "reward_type": self.reward_type,
            "dynamic_objects": self.dynamic_objects,
            "layout_id": self._layout_id,
        }

    def _activate_layout(self, layout_id: int) -> None:
        self._layout_id = layout_id
        self.base_maze, self.key_pos, self.door_pos = self._layouts[layout_id]
        self._dist_key = self._bfs_distances(self.key_pos)
        self._dist_door = self._bfs_distances(self.door_pos)
        self._dist_goal = self._bfs_distances(self.goal_pos)
        self._optimal = self.optimal_path_length()

    def _build_validated_layouts(
        self,
        size: int,
        seed: int | None,
        difficulty: str,
        max_attempts: int,
        n_layouts: int,
    ) -> list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]]:
        layouts = []
        seen: set[tuple[bytes, tuple[int, int], tuple[int, int]]] = set()
        base_seed = 0 if seed is None else seed
        attempts_per_layout = max(50, max_attempts)

        offset = 0
        while len(layouts) < n_layouts and offset < attempts_per_layout * n_layouts:
            try:
                layout = self._build_validated_maze(
                    size=size,
                    seed=base_seed + offset,
                    difficulty=difficulty,
                    max_attempts=attempts_per_layout,
                )
            except RuntimeError:
                offset += attempts_per_layout
                continue

            maze, key_pos, door_pos = layout
            signature = (maze.tobytes(), key_pos, door_pos)
            if signature not in seen:
                layouts.append(layout)
                seen.add(signature)
            offset += attempts_per_layout

        if not layouts:
            raise RuntimeError("Could not generate any valid dynamic door/key layouts.")
        return layouts

    def _build_validated_maze(
        self,
        size: int,
        seed: int | None,
        difficulty: str,
        max_attempts: int,
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        base_seed = 0 if seed is None else seed
        for attempt in range(max_attempts):
            attempt_seed = base_seed + attempt
            rng = np.random.default_rng(attempt_seed)
            maze = generate_maze(size=size, seed=attempt_seed, difficulty=difficulty)
            door_pos = self._choose_door(maze)
            if door_pos is None:
                continue

            blocked = maze.copy()
            blocked[door_pos] = WALL
            if self._bfs_dist_on(blocked, self.start_pos, self.goal_pos) != -1:
                continue

            start_side = self._bfs_reachable(blocked, self.start_pos)
            key_pos = self._choose_key(maze, start_side, door_pos, rng)
            if key_pos is None:
                continue
            if not self._meets_difficulty_requirements(maze, key_pos, door_pos, difficulty):
                continue

            phase2 = maze.copy()
            phase2[door_pos] = DOOR_LOCKED
            phase2[key_pos] = KEY
            return phase2, key_pos, door_pos

        raise RuntimeError(
            "Could not generate a valid door/key maze. Try another seed, "
            "larger size, or easier difficulty."
        )

    def _meets_difficulty_requirements(
        self,
        maze: np.ndarray,
        key_pos: tuple[int, int],
        door_pos: tuple[int, int],
        difficulty: str,
    ) -> bool:
        """Reject layouts where the required subgoal chain is too short.

        Phase 2 is meant to test long-horizon credit assignment. Without these
        checks, a valid chokepoint maze can still place the key and door close
        enough that random exploration solves the task too often.
        """
        door_free = maze.copy()
        door_free[door_pos] = FREE
        d_start_key = self._bfs_dist_on(door_free, self.start_pos, key_pos)
        d_key_door = self._bfs_dist_on(door_free, key_pos, door_pos)
        d_door_goal = self._bfs_dist_on(door_free, door_pos, self.goal_pos)
        if min(d_start_key, d_key_door, d_door_goal) < 0:
            return False

        segment_min = {
            "easy": max(4, int(0.5 * self.size)),
            "medium": max(8, int(0.8 * self.size)),
            "hard": max(10, int(1.0 * self.size)),
        }[difficulty]
        total_min = {
            "easy": max(20, int(2.4 * self.size)),
            "medium": max(34, int(3.4 * self.size)),
            "hard": max(38, int(3.8 * self.size)),
        }[difficulty]
        total = d_start_key + d_key_door + d_door_goal
        return (
            d_start_key >= segment_min
            and d_key_door >= segment_min
            and d_door_goal >= segment_min
            and total >= total_min
        )

    def _choose_door(self, maze: np.ndarray) -> tuple[int, int] | None:
        path = self._shortest_path_on(maze, self.start_pos, self.goal_pos)
        if len(path) < 5:
            return None

        best_cell = None
        best_score = -1
        for cell in path[1:-1]:
            test = maze.copy()
            test[cell] = WALL
            dist = self._bfs_dist_on(test, self.start_pos, self.goal_pos)
            score = self.size * self.size * 2 if dist == -1 else dist
            if score > best_score:
                best_cell = cell
                best_score = score
        return best_cell

    def _choose_key(
        self,
        maze: np.ndarray,
        reachable: Iterable[tuple[int, int]],
        door_pos: tuple[int, int],
        rng: np.random.Generator,
    ) -> tuple[int, int] | None:
        candidates = []
        for cell in reachable:
            if cell in (self.start_pos, self.goal_pos, door_pos):
                continue
            if self._degree(maze, cell) == 1:
                dist = self._bfs_dist_on(maze, self.start_pos, cell)
                if dist > 0:
                    candidates.append((dist, cell))

        if not candidates:
            return None

        max_dist = max(dist for dist, _ in candidates)
        far_dead_ends = [cell for dist, cell in candidates if dist >= max_dist * 0.6]
        return far_dead_ends[int(rng.integers(len(far_dead_ends)))]

    def _degree(self, maze: np.ndarray, cell: tuple[int, int]) -> int:
        r, c = cell
        return sum(
            1
            for nr, nc in self._neighbors(r, c)
            if maze[nr, nc] != WALL
        )

    def _bfs_distances(self, target: tuple[int, int]) -> np.ndarray:
        dist = np.full((self.size, self.size), np.inf)
        dist[target] = 0
        q = deque([target])
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors(r, c):
                if self.base_maze[nr, nc] != WALL and np.isinf(dist[nr, nc]):
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

    def _bfs_dist_between(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        door_passable: bool = True,
    ) -> int:
        maze = self.base_maze.copy()
        if door_passable:
            maze[self.door_pos] = FREE
        return self._bfs_dist_on(maze, start, goal)

    def _bfs_dist_on(self, maze: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
        path = self._shortest_path_on(maze, start, goal)
        return len(path) - 1 if path else -1

    def _shortest_path_on(
        self,
        maze: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        if maze[start] == WALL or maze[goal] == WALL:
            return []
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        q = deque([start])
        while q:
            cell = q.popleft()
            if cell == goal:
                return self._reconstruct_path(parent, goal)
            r, c = cell
            for nr, nc in self._neighbors(r, c):
                if maze[nr, nc] != WALL and (nr, nc) not in parent:
                    parent[(nr, nc)] = cell
                    q.append((nr, nc))
        return []

    def _bfs_reachable(self, maze: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
        if maze[start] == WALL:
            return []
        seen = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors(r, c):
                if maze[nr, nc] != WALL and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return list(seen)

    def _neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in ACTIONS.values():
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc):
                out.append((nr, nc))
        return out

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    @staticmethod
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

    _COLORS = {
        FREE: (205, 214, 244),
        WALL: (30, 30, 46),
        AGENT: (137, 220, 235),
        GOAL: (166, 227, 161),
        KEY: (249, 226, 175),
        DOOR_LOCKED: (245, 169, 127),
        DOOR_OPEN: (148, 226, 213),
    }
    _CELL_PX = 32

    def _render_rgb(self) -> np.ndarray:
        px = self._CELL_PX
        img = np.zeros((self.size * px, self.size * px, 3), dtype=np.uint8)
        view = self._get_obs()[:-2].reshape(self.size, self.size).astype(int)
        for r in range(self.size):
            for c in range(self.size):
                img[r * px : (r + 1) * px, c * px : (c + 1) * px] = self._COLORS[view[r, c]]
        return img

    def _render_pygame(self) -> None:
        import pygame

        px = self._CELL_PX
        side = self.size * px
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((side, side))
            pygame.display.set_caption("Door + Key Maze")
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        frame = self._render_rgb()
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._screen.blit(surf, (0, 0))
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
