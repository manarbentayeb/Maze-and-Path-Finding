# Maze and Path Finding with Reinforcement Learning

This repository studies reinforcement learning agents for grid-based maze
navigation. It includes static mazes, door-key dependency mazes, weighted-cost
terrain, dynamic obstacles, tabular Q-learning, DQN, Double DQN-style Phase 2
experiments, PPO experiment hooks, metrics, tests, saved models, and generated
plots.

The project has moved beyond the original Phase 1 baseline. The active codebase
now supports:

- Static fully observable maze navigation.
- Door-key mazes with required subgoal ordering.
- Weighted-cost mazes with mud cells and extra terrain penalty.
- Dynamic obstacle mazes with solvability-preserving wall changes.
- Tabular Q-learning and DQN training pipelines.
- Phase 2 DQN, DDQN, and PPO experiment utilities.
- Reproducible outputs under `models/` and `results/`.

## Research Objective

The main objective is to compare how different RL methods learn shortest-path
navigation policies as the environment becomes more structured and less
stationary.

The experiments measure:

- Success rate.
- Mean return.
- Steps to goal.
- Convergence episode.
- Sample efficiency.
- Path optimality against a BFS shortest-path baseline.
- Door-key subgoal completion rates for Phase 2.

## Environments

### Static Maze

`src/env/maze_env.py`

The baseline environment is a Gymnasium-compatible grid world with fixed start
and goal positions.

- Start: `(0, 0)`.
- Goal: `(size - 1, size - 1)`.
- Actions: up, down, left, right.
- Observation: flat vector of length `size * size`.
- Cell encoding: wall `0.0`, free `1.0`, agent `2.0`, goal `3.0`.
- Max episode length: `4 * size^2`.

Named levels:

| Level | Size |
| --- | ---: |
| `tiny` | 8 x 8 |
| `small` | 10 x 10 |
| `medium` | 20 x 20 |
| `large` | 50 x 50 |

Difficulty controls obstacle density:

| Difficulty | Wall density |
| --- | ---: |
| `easy` | 0.18 |
| `medium` | 0.28 |
| `hard` | 0.36 |

### Door-Key Maze

`src/env/door_key_maze_env.py`

Phase 2 adds long-horizon dependency structure. The agent must collect a key,
open a locked door, and then reach the goal. The environment validates generated
layouts so the door acts as a chokepoint and the key cannot be skipped.

- Observation: `size * size + 2`.
- Extra flags: `has_key`, `door_open`.
- Cell encoding: free `0`, wall `1`, agent `2`, goal `3`, key `4`, locked door
  `5`, open door `6`.
- Reward modes: `sparse`, `subgoal`, and `potential`.
- Dynamic object layouts can change across resets.

### Weighted-Cost Maze

`src/env/weighted_maze_env.py`

Weighted mazes add mud cells, which are traversable but costly.

- Mud encoding: `4.0` in observations and `2` in the internal maze array.
- Default mud density: `0.2`.
- Default mud penalty: `0.05`.
- The learned policy should trade off geometric path length against terrain
  cost.

### Dynamic Obstacle Maze

`src/env/dynamic_maze_env.py`

Dynamic mazes randomly flip walls/free cells during an episode while preserving
start-goal solvability.

- Default dynamic probability: `0.05`.
- Protected cells: start, goal, and current agent position.
- The environment tracks dynamic changes in `info`.

## Reward Defaults

The static, weighted, and dynamic maze runners use
`SparseWithPenalty` from `src/env/rewards.py` by default:

| Event | Reward |
| --- | ---: |
| Goal reached | `+10.0` |
| Wall or boundary hit | `-0.3` |
| Normal step | `-0.001` |

The weighted maze subtracts an additional mud penalty when the agent steps on a
mud cell. The door-key environment defines its own sparse, subgoal, and
potential reward modes in `src/env/door_key_maze_env.py`.

## Algorithms

### Tabular Q-learning

`src/agents/qlearning_agent.py`

Q-learning uses scalar state indices and an epsilon-greedy policy. For the base
maze, the state index is:

```text
state = row * size + col
```

For door-key mazes, the state index also includes `has_key` and `door_open`
flags.

Default hyperparameters:

| Parameter | Value |
| --- | ---: |
| Learning rate | 0.1 |
| Discount factor | 0.99 |
| Epsilon start | 1.0 |
| Epsilon end | 0.01 |
| Epsilon decay | 0.995 |

### DQN

`src/agents/dqn_agent.py`

DQN uses a multilayer perceptron, replay buffer, target network, and epsilon
decay. The default hidden architecture is `[128, 128]` for Phase 1/custom maze
experiments and `[256, 128]` in the Phase 2 config.

### Phase 2 Deep RL

`src/training/trainer_phase2.py`

Phase 2 includes DQN/DDQN training utilities and PPO integration through
Stable-Baselines3. Saved results include quick DQN, DDQN, and PPO runs over
sparse, subgoal, and potential rewards.

## Installation

Recommended Python version: 3.10 or 3.11.

```bash
pip install -r requirements.txt
```

Install the package in editable mode when running modules from outside the
repository root:

```bash
pip install -e .
```

## Running Experiments

Static Q-learning:

```bash
python -m src.experiments.run_qlearning --level small --difficulty hard --episodes 5000 --seed 42
```

Static DQN:

```bash
python -m src.experiments.run_dqn --level small --difficulty hard --episodes 2000 --seed 42
```

Weighted maze:

```bash
python -m src.experiments.run_weighted --algo qlearning --level small --difficulty medium --episodes 2000 --seed 42
python -m src.experiments.run_weighted --algo dqn --level small --difficulty medium --episodes 500 --seed 42
```

Dynamic maze:

```bash
python -m src.experiments.run_dynamic --algo qlearning --level small --difficulty medium --episodes 2000 --seed 42
python -m src.experiments.run_dynamic --algo dqn --size 5 --difficulty medium --episodes 5 --seed 42
```

Custom maze runner:

```bash
python -m src.experiments.run_custom_mazes --env weighted --algo qlearning
```

Use `--no-plots` on experiment commands when you only want models and metrics.

## Outputs

The repository stores experiment artifacts in predictable folders:

```text
models/
  qlearning/              Saved Q-tables
  dqn/                    Saved PyTorch models

results/
  logs/                   CSV histories and JSON summaries
  metrics/                NumPy metrics and path JSON files
  plots/                  Learning curves, success curves, heatmaps, policies
  videos/                 Rollout animations
```

Typical plot types:

- Maze layout.
- Greedy path.
- Learning curve.
- Success curve.
- Q-value heatmap.
- Policy map.
- Phase 2 rolling subgoal rates and reward ablations.

## Saved Result Highlights

These are read from the existing `results/logs/*_summary.json` files.

| Run | Episodes | Success | Mean return | Mean steps | Path optimality |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qlearning_small_hard_seed7` | 5000 | 1.00 | 9.9798 | 18.17 | 0.9906 |
| `dqn_small_hard_seed42` | 2000 | 1.00 | 9.6993 | 32.62 | 0.8584 |
| `weighted_qlearning_small_medium_seed42` | 2000 | 1.00 | 9.7528 | 24.25 | 0.9072 |
| `weighted_dqn_small_medium_seed42` | 500 | 0.32 | -6.7437 | 206.25 | 0.1067 |
| `dynamic_qlearning_small_medium_seed42` | 2000 | 0.00 | -0.7708 | n/a | n/a |
| `p2_quick_dqn_subgoal_small_medium_seed0` | 1000 | 0.93 | -0.1269 | 75.74 | 0.7290 |
| `p2_quick_ddqn_subgoal_small_medium_seed0` | 1000 | 0.77 | -0.7963 | 60.51 | 0.7622 |

Interpretation:

- Tabular Q-learning is very strong in small discrete static and weighted mazes.
- DQN solves the static small hard maze but needs more samples than Q-learning.
- Weighted terrain is harder for DQN under the saved 500-episode budget.
- The saved dynamic Q-learning run did not solve the dynamic maze; dynamic wall
  changes make the transition model substantially harder.
- Phase 2 subgoal rewards greatly improve DQN/DDQN behavior in the door-key
  task by making key and door progress learnable before the terminal reward.

## Project Structure

```text
configs/                  YAML experiment settings
notebooks/                Exploration, analysis, and visualization notebooks
src/
  agents/                 Q-learning, DQN, networks, replay buffer
  env/                    Static, door-key, weighted, and dynamic mazes
  experiments/            Command-line experiment runners
  training/               Training loops and evaluators
  utils/                  Metrics, logging, IO, seeds
  visualization/          Plots, heatmaps, policy maps, animations
tests/                    Pytest coverage for environments and agents
models/                   Saved Q-tables and DQN checkpoints
results/                  Logs, metrics, plots, and videos
```

## Testing

Run the test suite:

```bash
python -m pytest
```

The tests cover:

- Maze generation and solvability.
- Environment observations and rewards.
- Q-learning updates and epsilon decay.
- Training loop metrics.
- Door-key layout validity and subgoal sequence.
- Weighted mud penalties.
- Dynamic obstacle solvability and change tracking.

## Report

A LaTeX research-style report is included in:

```text
report.tex
```

It references the existing figures in `results/plots/`, so compile it from the
repository root with:

```bash
pdflatex report.tex
```
