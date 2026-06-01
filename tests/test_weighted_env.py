import numpy as np

from src.env import WeightedMazeEnv
from src.agents import QLearningAgent, DQNAgent
from src.training.trainer_qlearning import train_qlearning

def test_weighted_maze_has_correct_size():
    env = WeightedMazeEnv(size=10, seed=42)
    assert env.size == 10
    assert env.maze.shape == (10, 10)
    assert env.observation_space.shape == (100,)
    assert env.max_steps == 400

def test_weighted_maze_has_mud_cells():
    env = WeightedMazeEnv(size=10, seed=42, mud_density=0.2)
    mud_count = np.sum(env.maze == 2)
    assert mud_count > 0
    assert len(env.mud_cells) == mud_count

def test_weighted_maze_is_solvable():
    from src.env.maze_generator import shortest_path
    env = WeightedMazeEnv(size=10, seed=42)
    # Treat mud as passable (0) for BFS
    maze_for_bfs = env.maze.copy()
    maze_for_bfs[maze_for_bfs == 2] = 0
    path = shortest_path(maze_for_bfs)
    assert len(path) > 0

def test_mud_step_applies_extra_penalty():
    env = WeightedMazeEnv(size=10, seed=42, mud_penalty=0.05)
    env.reset()
    
    # Manually place agent next to a mud cell and step onto it
    if len(env.mud_cells) > 0:
        mud_r, mud_c = env.mud_cells[0]
        # Put agent above the mud cell if possible
        if mud_r > 0 and env.maze[mud_r-1, mud_c] != 1:
            env.agent_pos = (mud_r-1, mud_c)
            action = 1 # Down
        # Otherwise put agent left of the mud cell
        elif mud_c > 0 and env.maze[mud_r, mud_c-1] != 1:
            env.agent_pos = (mud_r, mud_c-1)
            action = 3 # Right
        else:
            return # Skip if we can't easily test this cell
            
        _, reward, _, _, info = env.step(action)
        assert info["hit_wall"] is False
        # Normal penalty is -0.001, mud penalty is -0.05, total should be -0.051
        assert np.isclose(reward, -0.051)

def test_normal_step_no_extra_penalty():
    env = WeightedMazeEnv(size=10, seed=42)
    env.reset()
    # Find a free cell next to the start pos that is NOT mud
    start_r, start_c = env.start_pos
    action = None
    for a, (dr, dc) in {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}.items():
        nr, nc = start_r + dr, start_c + dc
        if 0 <= nr < env.size and 0 <= nc < env.size and env.maze[nr, nc] == 0:
            action = a
            break
            
    if action is not None:
        _, reward, _, _, _ = env.step(action)
        assert np.isclose(reward, -0.001)

def test_weighted_obs_encodes_mud_as_4():
    env = WeightedMazeEnv(size=10, seed=42)
    obs, _ = env.reset()
    grid = obs.reshape((env.size, env.size))
    for r, c in env.mud_cells:
        assert grid[r, c] == 4.0

def test_weighted_wall_rejection_unchanged():
    env = WeightedMazeEnv(size=10, seed=42)
    env.reset()
    # Start pos is (0,0), going Up or Left is a guaranteed wall hit
    _, reward, _, _, info = env.step(0)
    assert info["hit_wall"] is True
    assert np.isclose(reward, -0.3)

def test_weighted_training_loop_runs():
    env = WeightedMazeEnv(size=5, seed=42)
    agent = QLearningAgent(n_states=25, n_actions=4, seed=42)
    result = train_qlearning(env, agent, episodes=2, log_every=0)
    assert len(result.tracker) == 2
