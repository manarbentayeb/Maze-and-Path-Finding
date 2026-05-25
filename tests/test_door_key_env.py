import numpy as np

from src.env import DOOR_OPEN, DoorKeyMazeEnv
from src.utils.metrics_phase2 import DoorKeyMetricsTracker


def _actions_for_path(path):
    action_by_delta = {
        (-1, 0): 0,
        (1, 0): 1,
        (0, -1): 2,
        (0, 1): 3,
    }
    return [
        action_by_delta[(b[0] - a[0], b[1] - a[1])]
        for a, b in zip(path[:-1], path[1:])
    ]


def test_door_key_env_uses_phase1_size_params():
    env = DoorKeyMazeEnv(level="small", difficulty="medium", seed=42)
    obs, info = env.reset()

    assert env.size == 10
    assert obs.shape == (102,)
    assert env.max_steps == 400
    assert info["key_pos"] != info["door_pos"]
    assert np.isin(obs[:-2], [0, 1, 2, 3, 4, 5, 6]).all()


def test_door_key_env_supports_easy_medium_hard_params():
    for difficulty in ["easy", "medium", "hard"]:
        env = DoorKeyMazeEnv(level="small", difficulty=difficulty, seed=42)

        assert env.difficulty == difficulty
        assert env.optimal_path_length() > 0


def test_medium_layout_has_nontrivial_subgoal_chain():
    env = DoorKeyMazeEnv(level="small", difficulty="medium", seed=42)
    d_start_key = env._bfs_dist_between(env.start_pos, env.key_pos, door_passable=True)
    d_key_door = env._bfs_dist_between(env.key_pos, env.door_pos, door_passable=True)
    d_door_goal = env._bfs_dist_between(env.door_pos, env.goal_pos, door_passable=True)

    assert d_start_key >= 8
    assert d_key_door >= 8
    assert d_door_goal >= 8
    assert env.optimal_path_length() >= 34


def test_dynamic_env_changes_key_or_door_across_resets():
    env = DoorKeyMazeEnv(level="small", difficulty="medium", seed=42, n_dynamic_layouts=4)
    layouts = set()

    for _ in range(12):
        _, info = env.reset()
        layouts.add((info["key_pos"], info["door_pos"], info["layout_id"]))

    assert len(layouts) > 1


def test_door_blocks_all_start_goal_bypasses():
    env = DoorKeyMazeEnv(size=10, difficulty="medium", seed=42)
    blocked = env.base_maze.copy()
    blocked[env.door_pos] = 1

    assert env._bfs_dist_on(blocked, env.start_pos, env.goal_pos) == -1


def test_locked_door_rejects_agent_without_key():
    env = DoorKeyMazeEnv(size=10, difficulty="medium", seed=42, reward_type="sparse")
    env.reset()
    maze_without_objects = env.base_maze.copy()
    maze_without_objects[env.door_pos] = 0
    path = env._shortest_path_on(maze_without_objects, env.start_pos, env.door_pos)

    for action in _actions_for_path(path[:-1]):
        env.step(action)
    before = env.agent_pos
    _, reward, terminated, truncated, info = env.step(_actions_for_path(path[-2:])[0])

    assert env.agent_pos == before
    assert reward == -0.5
    assert info["hit_wall"] is True
    assert terminated is False
    assert truncated is False


def test_key_then_door_then_goal_sequence_opens_door():
    env = DoorKeyMazeEnv(size=10, difficulty="medium", seed=42, reward_type="subgoal")
    env.reset()
    maze_without_objects = env.base_maze.copy()
    maze_without_objects[env.door_pos] = 0

    key_path = env._shortest_path_on(maze_without_objects, env.start_pos, env.key_pos)
    rewards = []
    for action in _actions_for_path(key_path):
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)
    assert env.has_key == 1
    assert rewards[-1] == 0.5

    door_path = env._shortest_path_on(maze_without_objects, env.key_pos, env.door_pos)
    for action in _actions_for_path(door_path):
        _, reward, _, _, _ = env.step(action)
    assert env.door_open == 1
    assert env.maze[env.door_pos] == DOOR_OPEN
    assert reward == 0.3

    goal_path = env._shortest_path_on(maze_without_objects, env.door_pos, env.goal_pos)
    terminated = False
    for action in _actions_for_path(goal_path):
        _, reward, terminated, _, _ = env.step(action)
    assert terminated is True
    assert reward == 1.0


def test_potential_reward_returns_float():
    env = DoorKeyMazeEnv(size=10, difficulty="medium", seed=42, reward_type="potential")
    env.reset()
    _, reward, _, _, _ = env.step(3)

    assert isinstance(reward, float)


def test_phase2_summary_reports_partial_progress_without_success():
    tracker = DoorKeyMetricsTracker(window=100, optimal_path_len=52)
    tracker.record(
        episode=1,
        total_reward=-10.0,
        steps=400,
        success=False,
        picked_up_key=True,
        opened_door=True,
        steps_to_key=50,
        steps_to_door=120,
        optimal_path_len=52,
    )

    summary = tracker.compute_summary()

    assert summary["success_rate"] == 0.0
    assert summary["mean_steps_to_goal"] is None
    assert summary["path_optimality"] is None
    assert summary["key_pickup_rate"] == 1.0
    assert summary["door_opening_rate"] == 1.0
    assert summary["mean_steps_to_key"] == 50.0
    assert summary["mean_key_to_door_steps"] == 70.0
