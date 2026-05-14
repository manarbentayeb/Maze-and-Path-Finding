"""Run the Phase 1 Q-learning baseline.

Example:
    python -m src.experiments.run_qlearning --size 10 --episodes 5000 --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

from src.agents.qlearning_agent import QLearningAgent
from src.env.maze_env import MazeEnv
from src.env.maze_generator import LEVEL_SIZES
from src.training.evaluator import evaluate_qlearning
from src.training.trainer_qlearning import train_qlearning
from src.utils.io import DIRS, ensure_dirs, figure_path, save_metrics, save_qtable
from src.utils.logger import TrainingLogger
from src.utils.seed import set_global_seed
from src.visualization.heatmaps import plot_q_heatmap
from src.visualization.plots import plot_learning_curve, plot_success_curve
from src.visualization.policy_maps import plot_policy_arrows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular Q-learning on a static maze.")
    parser.add_argument("--size", type=int, default=10, help="Maze side length.")
    parser.add_argument("--level", choices=list(LEVEL_SIZES), default=None, help="Named maze level.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Q-learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-plots", action="store_true", help="Skip saving matplotlib figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_global_seed(args.seed)

    actual_size = LEVEL_SIZES[args.level] if args.level else args.size
    level_part = args.level or f"{actual_size}x{actual_size}"
    run_id = f"qlearning_{level_part}_{args.difficulty}_seed{args.seed}"
    env = MazeEnv(size=actual_size, level=args.level, difficulty=args.difficulty, seed=args.seed)
    env.action_space.seed(args.seed)

    agent = QLearningAgent(
        n_states=env.size * env.size,
        n_actions=env.action_space.n,
        learning_rate=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    logger = TrainingLogger(run_id=run_id, output_dir=str(DIRS["logs"]), log_every=args.log_every)
    try:
        result = train_qlearning(env=env, agent=agent, episodes=args.episodes, logger=logger)
        eval_result = evaluate_qlearning(env, agent, episodes=100)
        summary = result.tracker.compute_summary()
        summary.update(
            {
                "eval_success_rate": round(eval_result.success_rate, 4),
                "eval_mean_return": round(eval_result.mean_return, 4),
                "eval_mean_steps": round(eval_result.mean_steps, 2),
                "maze_size": env.size,
                "level": env.level,
                "difficulty": env.difficulty,
                "max_steps": env.max_steps,
            }
        )
        logger.log_summary(summary)
    finally:
        logger.close()

    save_qtable(agent.q_table, run_id)
    save_metrics(
        {
            "episodes": result.tracker.episodes_arr,
            "rewards": result.tracker.rewards,
            "steps": result.tracker.steps_arr,
            "success": result.tracker.success_arr,
        },
        run_id,
    )
    _save_path_json(result.paths, DIRS["metrics"] / f"{run_id}_paths.json")

    if not args.no_plots:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _save_plots(run_id, env, agent, result)
        except Exception as exc:
            print(f"[warning] plots skipped: {exc}")

    print(json.dumps(summary, indent=2))


def _save_path_json(paths: dict[str, list[tuple[int, int]]], path: Path) -> None:
    serializable = {name: [[int(r), int(c)] for r, c in points] for name, points in paths.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _save_plots(run_id: str, env: MazeEnv, agent: QLearningAgent, result) -> None:
    figures = {
        "learning_curve": plot_learning_curve(result.tracker.rewards),
        "success_curve": plot_success_curve(result.tracker.success_arr),
        "q_heatmap": plot_q_heatmap(agent.q_table, env.maze),
        "policy_map": plot_policy_arrows(agent.q_table, env.maze),
    }
    for name, fig in figures.items():
        fig.savefig(figure_path(run_id, name), dpi=150)


if __name__ == "__main__":
    main()
