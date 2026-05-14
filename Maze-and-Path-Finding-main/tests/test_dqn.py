import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.agents.dqn_agent import DQNAgent
from src.env import MazeEnv
from src.training.evaluator import evaluate_dqn
from src.training.trainer_dqn import q_table_from_dqn, train_dqn


def test_dqn_selects_valid_action():
    env = MazeEnv(size=8, seed=0, difficulty="easy")
    obs, _ = env.reset()
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        batch_size=2,
        warmup_steps=2,
        hidden_dims=(16,),
        seed=0,
    )

    action = agent.select_action(obs, training=False)

    assert env.action_space.contains(action)


def test_replay_update_returns_loss_after_warmup():
    env = MazeEnv(size=8, seed=1, difficulty="easy")
    obs, _ = env.reset()
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        batch_size=2,
        warmup_steps=2,
        hidden_dims=(16,),
        seed=1,
    )

    losses = []
    for _ in range(4):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        losses.append(agent.update(obs, action, reward, next_obs, terminated or truncated))
        obs = next_obs

    assert any(loss > 0.0 for loss in losses)


def test_short_dqn_training_loop_records_metrics():
    env = MazeEnv(size=8, seed=2, difficulty="easy")
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        batch_size=4,
        warmup_steps=4,
        hidden_dims=(16,),
        seed=2,
    )

    result = train_dqn(env, agent, episodes=2, log_every=0)

    assert len(result.tracker) == 2
    assert np.isfinite(result.tracker.rewards).all()
    assert "greedy" in result.paths


def test_dqn_can_be_evaluated_and_plotted_as_q_table():
    env = MazeEnv(size=8, seed=3, difficulty="easy")
    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        batch_size=4,
        warmup_steps=4,
        hidden_dims=(16,),
        seed=3,
    )
    train_dqn(env, agent, episodes=1, log_every=0)

    result = evaluate_dqn(env, agent, episodes=2)
    q_table = q_table_from_dqn(env, agent)

    assert 0.0 <= result.success_rate <= 1.0
    assert len(result.paths) == 2
    assert q_table.shape == (env.size * env.size, env.action_space.n)
