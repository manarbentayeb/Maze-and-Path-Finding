import numpy as np

from src.agents import QLearningAgent
from src.env import MazeEnv
from src.training.evaluator import evaluate_qlearning
from src.training.trainer_qlearning import train_qlearning


def test_qlearning_update_changes_q_value():
    agent = QLearningAgent(n_states=4, n_actions=2, learning_rate=0.5, gamma=0.9, seed=0)

    td_error = agent.update(state=0, action=1, reward=1.0, next_state=2, done=True)

    assert td_error == 1.0
    assert agent.q_table[0, 1] == 0.5


def test_epsilon_decays_to_floor():
    agent = QLearningAgent(
        n_states=4,
        n_actions=2,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.1,
        seed=0,
    )

    agent.decay_epsilon()
    agent.decay_epsilon()

    assert agent.epsilon == 0.2


def test_short_training_loop_records_metrics():
    env = MazeEnv(size=10, seed=3)
    agent = QLearningAgent(n_states=env.size * env.size, n_actions=env.action_space.n, seed=3)

    result = train_qlearning(env, agent, episodes=5, log_every=0)

    assert len(result.tracker) == 5
    assert result.agent.q_table.shape == (100, 4)
    assert np.isfinite(result.tracker.rewards).all()
    assert "greedy" in result.paths


def test_qlearning_can_be_evaluated():
    env = MazeEnv(size=10, seed=4)
    agent = QLearningAgent(n_states=env.size * env.size, n_actions=env.action_space.n, seed=4)
    train_qlearning(env, agent, episodes=3, log_every=0)

    result = evaluate_qlearning(env, agent, episodes=2)

    assert 0.0 <= result.success_rate <= 1.0
    assert len(result.paths) == 2
