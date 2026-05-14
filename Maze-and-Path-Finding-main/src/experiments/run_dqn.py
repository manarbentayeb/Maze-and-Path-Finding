"""Run the Phase 1 DQN baseline.

Example:
    python -m src.experiments.run_dqn --level small --episodes 1000 --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

from src.agents.dqn_agent import DQNAgent
from src.env.maze_env import MazeEnv
from src.env.maze_generator import LEVEL_SIZES
from src.training.evaluator import evaluate_dqn
from src.training.trainer_dqn import q_table_from_dqn, train_dqn
from src.utils.io import DIRS, ensure_dirs, figure_path, save_dqn, save_metrics
from src.utils.logger import TrainingLogger
from src.utils.seed import set_global_seed
from src.visualization.heatmaps import plot_q_heatmap
from src.visualization.plots import plot_learning_curve, plot_success_curve
from src.visualization.policy_maps import plot_policy_arrows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on a static maze.")
    parser.add_argument("--size", type=int, default=10, help="Maze side length.")
    parser.add_argument("--level", choices=list(LEVEL_SIZES), default=None, help="Named maze level.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-every", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-plots", action="store_true", help="Skip saving matplotlib figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_global_seed(args.seed)

    actual_size = LEVEL_SIZES[args.level] if args.level else args.size
    level_part = args.level or f"{actual_size}x{actual_size}"
    run_id = f"dqn_{level_part}_{args.difficulty}_seed{args.seed}"
    env = MazeEnv(size=actual_size, level=args.level, difficulty=args.difficulty, seed=args.seed)
    env.action_space.seed(args.seed)

    agent = DQNAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_every=args.target_update_every,
        warmup_steps=args.warmup_steps,
        hidden_dims=tuple(args.hidden_dims),
        seed=args.seed,
        device=args.device,
    )

    logger = TrainingLogger(run_id=run_id, output_dir=str(DIRS["logs"]), log_every=args.log_every)
    try:
        result = train_dqn(env=env, agent=agent, episodes=args.episodes, logger=logger)
        eval_result = evaluate_dqn(env, agent, episodes=100)
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
                "loss_count": len(result.losses),
            }
        )
        logger.log_summary(summary)
    finally:
        logger.close()

    save_dqn(agent.q_network, run_id)
    save_metrics(
        {
            "episodes": result.tracker.episodes_arr,
            "rewards": result.tracker.rewards,
            "steps": result.tracker.steps_arr,
            "success": result.tracker.success_arr,
            "losses": result.losses,
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


def _save_plots(run_id: str, env: MazeEnv, agent: DQNAgent, result) -> None:
    q_table = q_table_from_dqn(env, agent)
    figures = {
        "learning_curve": plot_learning_curve(result.tracker.rewards, title="DQN return"),
        "success_curve": plot_success_curve(result.tracker.success_arr, title="DQN success rate"),
        "q_heatmap": plot_q_heatmap(q_table, env.maze, title="DQN Q-value heatmap"),
        "policy_map": plot_policy_arrows(q_table, env.maze, title="DQN greedy policy"),
    }
    for name, fig in figures.items():
        fig.savefig(figure_path(run_id, name), dpi=150)


if __name__ == "__main__":
    main()
