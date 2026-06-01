# Maze-and-Path-Finding

Research project for reinforcement learning in static maze environments.

The current working phase is **Phase 1: Direct MDP with tabular Q-learning**.
Other algorithms such as DQN, PPO, and SAC are part of the wider research plan,
but this repository is currently set up to run and test the Q-learning baseline
first.

## Project Goal

The goal is to train reinforcement learning agents to solve generated maze
navigation tasks and compare learning quality using reproducible metrics.

For Phase 1, the maze is treated as a fully observable Markov Decision Process:

- State: agent position `(row, col)`
- Q-learning state index: `row * size + col`
- Actions: up, down, left, right
- Transitions: deterministic
- Episode ends when the goal is reached or the max step limit is exceeded

## Current Status

Implemented:

- Static maze environment
- Obstacle-block maze generator with difficulty levels
- Sparse reward with step and wall penalties
- Tabular Q-learning agent
- Training loop
- Greedy evaluation
- Metrics tracking
- CSV and JSON logging
- Q-table and metric saving
- Q-value heatmap, policy map, learning curve, success curve helpers
- Unit tests for the environment and Q-learning path

Not implemented yet:

- DQN training pipeline
- PPO pipeline
- SAC/discrete SAC pipeline
- Full multi-algorithm comparison experiments

## Environment

The environment lives in `src/env/maze_env.py`.

Maze properties:

- Grid type: 2D discrete maze
- Default level: `small`
- Named levels: `tiny=8x8`, `small=10x10`, `medium=20x20`, `large=50x50`
- Difficulty levels: `easy`, `medium`, `hard`
- Cell values in the maze array:
  - `0`: free cell
  - `1`: wall
- Start: top-left cell `(0, 0)`
- Goal: bottom-right cell `(size - 1, size - 1)`
- Max steps: `4 * size^2`

Observation encoding:

- Flat vector of length `size * size`
- `0.0`: wall
- `1.0`: free cell
- `2.0`: agent position
- `3.0`: goal position

Actions:

| Action | Meaning |
| --- | --- |
| `0` | Up |
| `1` | Down |
| `2` | Left |
| `3` | Right |

## Reward Function

Phase 1 uses the sparse baseline reward in `src/env/rewards.py`.

| Event | Reward |
| --- | ---: |
| Goal reached | `+1.0` |
| Wall or boundary hit | `-0.5` |
| Normal step | `-0.01` |

This reward encourages the agent to reach the goal while avoiding walls and
using shorter paths.

## Q-learning

The Q-learning agent lives in `src/agents/qlearning_agent.py`.

Default hyperparameters:

| Hyperparameter | Value |
| --- | ---: |
| Learning rate `alpha` | `0.1` |
| Discount `gamma` | `0.99` |
| Epsilon start | `1.0` |
| Epsilon end | `0.01` |
| Epsilon decay | `0.995` |
| Episodes | `5000` |

The Q-table shape is:

```text
(size * size, 4)
```

The update rule is:

```text
Q[s, a] = Q[s, a] + alpha * (reward + gamma * max(Q[s_next]) - Q[s, a])
```

## Project Structure

```text
configs/
  env.yaml                 Environment configuration
  qlearning.yaml           Q-learning configuration
  dqn.yaml                 DQN configuration

src/
  agents/
    base_agent.py          Common agent interface
    qlearning_agent.py     Tabular Q-learning implementation
    dqn_agent.py           DQN implementation
    networks.py            Neural network helpers
    replay_buffer.py       Replay buffer helper

  env/
    maze_env.py            Maze environment
    maze_generator.py      Obstacle-block maze generation
    rewards.py             Reward functions

  experiments/
    run_qlearning.py       Main Q-learning experiment runner
    run_dqn.py             DQN experiment runner
    compare_algorithms.py  Future comparison script

  training/
    trainer_qlearning.py   Q-learning training loop
    evaluator.py           Greedy Q-learning evaluation
    trainer_dqn.py         Future DQN trainer

  utils/
    io.py                  Save/load helpers
    logger.py              CSV/JSON/console logging
    metrics.py             Metrics tracking
    seed.py                Reproducibility helpers

  visualization/
    plots.py               Learning and success curves
    heatmaps.py            Q-value heatmap
    policy_maps.py         Policy arrow map
    animations.py          Episode path animation helper

tests/
  test_env.py              Environment tests
  test_qlearning.py        Q-learning tests
```

## Installation

Recommended Python version: `3.10` or `3.11`.

Install dependencies:

```bash
pip install -r requirements.txt
```

The core Q-learning path needs only lightweight packages such as NumPy and
pytest. The environment can use Gymnasium when installed, but also includes a
small fallback so Q-learning tests can run without Gymnasium.

## Running Q-learning

Train the default 10x10 Q-learning baseline:

```bash
python -m src.experiments.run_qlearning --size 10 --episodes 5000 --seed 42
```

Train by named level and difficulty:

```bash
python -m src.experiments.run_qlearning --level small --difficulty hard --episodes 5000 --seed 42
```

Run a quick smoke test:

```bash
python -m src.experiments.run_qlearning --level tiny --difficulty medium --episodes 20 --seed 42 --no-plots
```

Useful options:

```bash
python -m src.experiments.run_qlearning \
  --size 10 \
  --difficulty medium \
  --episodes 5000 \
  --seed 42 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon-start 1.0 \
  --epsilon-end 0.01 \
  --epsilon-decay 0.995
```

Use `--no-plots` if your local matplotlib installation is not working.

## Outputs

Training outputs are saved automatically.

```text
models/qlearning/
  qlearning_<level-or-size>_<difficulty>_seed<seed>.npy

results/metrics/
  qlearning_<level-or-size>_<difficulty>_seed<seed>_metrics.npz
  qlearning_<level-or-size>_<difficulty>_seed<seed>_paths.json

results/logs/
  qlearning_<level-or-size>_<difficulty>_seed<seed>.csv
  qlearning_<level-or-size>_<difficulty>_seed<seed>_summary.json

results/plots/
  qlearning_<level-or-size>_<difficulty>_seed<seed>_learning_curve.png
  qlearning_<level-or-size>_<difficulty>_seed<seed>_success_curve.png
  qlearning_<level-or-size>_<difficulty>_seed<seed>_q_heatmap.png
  qlearning_<level-or-size>_<difficulty>_seed<seed>_policy_map.png
```

## Metrics

The project tracks the Phase 1 research metrics:

| Metric | Meaning |
| --- | --- |
| Success rate | Fraction of episodes that reach the goal |
| Mean return | Average total episode reward |
| Steps to goal | Mean steps in successful episodes |
| Convergence episode | First episode where rolling success reaches 90% |
| Sample efficiency | Estimated steps required to reach convergence |
| Path optimality | Shortest path length divided by learned path length |

Metrics are implemented in `src/utils/metrics.py`.

## Visualizations

Available Q-learning visualizations:

- Learning curve: rolling average return
- Success curve: rolling success rate with 90% threshold
- Q-value heatmap: max Q-value per maze cell
- Policy arrow map: best action per free cell
- Path animation helper: animate a recorded path through the maze

Visualization code lives in `src/visualization/`.

Note: if matplotlib is incompatible with your installed NumPy version, training
still runs and plots are skipped with a warning.

## Testing

Run the full test suite:

```bash
python -m pytest
```

Current expected result:

```text
8 passed
```

The tests verify:

- Maze size and observation shape
- Maze solvability
- Wall penalty behavior
- Observation encoding
- Q-learning update rule
- Epsilon decay
- Training loop metric recording
- Greedy evaluation output

## Reproducibility

Use seeds for repeatable experiments:

```bash
python -m src.experiments.run_qlearning --size 10 --episodes 5000 --seed 42
```

The project seeds:

- Python random
- NumPy
- PyTorch if installed
- The environment action space when available

## Recommended Next Step

Run Q-learning on the 10x10 maze with several seeds and compare the summary
files in `results/logs/`.

Example:

```bash
python -m src.experiments.run_qlearning --size 10 --episodes 5000 --seed 1
python -m src.experiments.run_qlearning --size 10 --episodes 5000 --seed 2
python -m src.experiments.run_qlearning --size 10 --episodes 5000 --seed 3
python -m src.experiments.run_qlearning --size 10 --episodes 5000 --seed 4
python -m src.experiments.run_qlearning --size 10 --episodes 5000 --seed 5
```

This gives the first clean baseline before adding DQN, PPO, or SAC.
