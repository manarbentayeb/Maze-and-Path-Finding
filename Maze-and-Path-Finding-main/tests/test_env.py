import numpy as np

from src.env import LEVEL_SIZES, MazeEnv, optimal_path_length, shortest_path


def test_maze_env_uses_exact_requested_size():
    env = MazeEnv(size=10, seed=7)

    assert env.size == 10
    assert env.maze.shape == (10, 10)
    assert env.observation_space.shape == (100,)
    assert env.max_steps == 400


def test_generated_maze_is_solvable():
    env = MazeEnv(size=10, seed=123)

    assert env.maze[0, 0] == 0
    assert env.maze[-1, -1] == 0
    assert optimal_path_length(env.maze) > 0


def test_generated_maze_is_not_trivial_border_repair():
    env = MazeEnv(size=10, seed=42, difficulty="medium")
    path = shortest_path(env.maze)
    wall_ratio = float(np.mean(env.maze))

    assert wall_ratio >= 0.20
    assert len(path) - 1 > 2 * (env.size - 1)
    assert not all(r == 0 or c == env.size - 1 for r, c in path)


def test_named_levels_set_grid_size():
    for level, size in LEVEL_SIZES.items():
        env = MazeEnv(level=level, seed=1, difficulty="easy")

        assert env.size == size
        assert env.maze.shape == (size, size)
        assert optimal_path_length(env.maze) > 0


def test_step_rewards_and_wall_rejection():
    env = MazeEnv(size=10, seed=1)
    env.reset()

    _, reward, terminated, truncated, info = env.step(0)

    assert reward == -0.5
    assert env.agent_pos == (0, 0)
    assert info["hit_wall"] is True
    assert terminated is False
    assert truncated is False


def test_observation_marks_agent_and_goal():
    env = MazeEnv(size=10, seed=1)
    obs, _ = env.reset()
    grid = obs.reshape(env.size, env.size)

    assert grid[0, 0] == 2.0
    assert grid[-1, -1] == 3.0
    assert np.isin(grid, [0.0, 1.0, 2.0, 3.0]).all()
