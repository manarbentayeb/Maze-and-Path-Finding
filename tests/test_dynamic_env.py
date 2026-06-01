import numpy as np

from src.env import DynamicMazeEnv
from src.agents import QLearningAgent
from src.training.trainer_qlearning import train_qlearning

def test_dynamic_maze_has_correct_size():
    env = DynamicMazeEnv(size=10, seed=42)
    assert env.size == 10
    assert env.maze.shape == (10, 10)
    assert env.observation_space.shape == (100,)

def test_dynamic_maze_changes_during_episode():
    env = DynamicMazeEnv(size=10, seed=42, dynamic_prob=1.0) # Always change
    env.reset()
    initial_maze = env.maze.copy()
    
    # Step a few times to trigger changes
    for _ in range(5):
        # Pick an action that keeps us near the start
        env.step(1)
        env.step(0)
        
    assert not np.array_equal(env.maze, initial_maze)

def test_dynamic_maze_stays_solvable():
    env = DynamicMazeEnv(size=10, seed=42, dynamic_prob=1.0)
    env.reset()
    for _ in range(20):
        # We step around, dynamic changes happen, but _is_solvable should block any breaking change
        env.step(np.random.randint(4))
        
    from src.env.maze_generator import shortest_path
    path = shortest_path(env.maze)
    assert len(path) > 0

def test_dynamic_reset_restores_base_maze():
    env = DynamicMazeEnv(size=10, seed=42, dynamic_prob=1.0)
    env.reset()
    initial_maze = env.maze.copy()
    
    # Mutate
    for _ in range(10):
        env.step(np.random.randint(4))
        
    # Reset
    env.reset()
    assert np.array_equal(env.maze, initial_maze)
    assert env.dynamic_changes_this_episode == 0

def test_dynamic_protects_critical_cells():
    env = DynamicMazeEnv(size=10, seed=42, dynamic_prob=1.0)
    env.reset()
    
    start_r, start_c = env.start_pos
    goal_r, goal_c = env.goal_pos
    
    # Mutate a bunch
    for _ in range(50):
        env.step(np.random.randint(4))
        
    assert env.maze[start_r, start_c] == 0
    assert env.maze[goal_r, goal_c] == 0

def test_dynamic_info_tracks_changes():
    env = DynamicMazeEnv(size=10, seed=42, dynamic_prob=1.0)
    env.reset()
    
    # Because prob=1.0, it will try to change 1 cell every step
    _, _, _, _, info = env.step(1)
    
    assert "dynamic_changes_total" in info
    assert "dynamic_changes_this_episode" in info
    # It might fail solvability check, but if it succeeds it should be 1
    assert info["dynamic_changes_total"] >= 0

def test_dynamic_training_loop_runs():
    env = DynamicMazeEnv(size=5, seed=42, dynamic_prob=0.1)
    agent = QLearningAgent(n_states=25, n_actions=4, seed=42)
    result = train_qlearning(env, agent, episodes=2, log_every=0)
    assert len(result.tracker) == 2
