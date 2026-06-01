"""Run Dynamic Obstacle Maze training.

Example:
    python -m src.experiments.run_dynamic --algo dqn --level small --episodes 1000 --seed 42
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

from src.agents.qlearning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.env.dynamic_maze_env import DynamicMazeEnv
from src.env.maze_generator import LEVEL_SIZES
from src.training.evaluator import evaluate_qlearning, evaluate_dqn
from src.training.trainer_qlearning import train_qlearning
from src.training.trainer_dqn import train_dqn, q_table_from_dqn
from src.utils.io import DIRS, ensure_dirs, figure_path, save_qtable, save_dqn, save_metrics
from src.utils.logger import TrainingLogger
from src.utils.seed import set_global_seed
from src.visualization.plots import plot_learning_curve, plot_success_curve
from src.visualization.heatmaps import plot_q_heatmap
from src.visualization.policy_maps import plot_policy_arrows
from src.visualization.maze_viz import plot_dynamic_maze


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train agent on Dynamic Obstacle maze.")
    parser.add_argument("--algo", choices=["qlearning", "dqn"], required=True)
    parser.add_argument("--size", type=int, default=10, help="Maze side length.")
    parser.add_argument("--level", choices=list(LEVEL_SIZES), default=None, help="Named maze level.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--episodes", type=int, default=2000, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # Env params
    parser.add_argument("--dynamic-prob", type=float, default=0.05)
    
    # Q-Learning params
    parser.add_argument("--alpha", type=float, default=0.1)
    
    # Shared / DQN Params
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update-every", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--device", default="cpu")
    
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_global_seed(args.seed)

    actual_size = LEVEL_SIZES[args.level] if args.level else args.size
    level_part = args.level or f"{actual_size}x{actual_size}"
    run_id = f"dynamic_{args.algo}_{level_part}_{args.difficulty}_seed{args.seed}"
    
    env = DynamicMazeEnv(
        size=actual_size, 
        level=args.level, 
        difficulty=args.difficulty, 
        seed=args.seed,
        dynamic_prob=args.dynamic_prob
    )
    env.action_space.seed(args.seed)

    logger = TrainingLogger(run_id=run_id, output_dir=str(DIRS["logs"]), log_every=args.log_every)
    
    try:
        if args.algo == "qlearning":
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
            result = train_qlearning(env=env, agent=agent, episodes=args.episodes, logger=logger)
            eval_result = evaluate_qlearning(env, agent, episodes=100)
            save_qtable(agent.q_table, run_id)
        else:
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
            result = train_dqn(env=env, agent=agent, episodes=args.episodes, logger=logger)
            eval_result = evaluate_dqn(env, agent, episodes=100)
            save_dqn(agent.q_network, run_id)
            
        summary = result.tracker.compute_summary()
        summary.update({
            "env_type": "dynamic",
            "eval_success_rate": round(eval_result.success_rate, 4),
            "eval_mean_return": round(eval_result.mean_return, 4),
            "eval_mean_steps": round(eval_result.mean_steps, 2),
            "maze_size": env.size,
            "max_steps": env.max_steps,
        })
        if args.algo == "dqn":
            summary["loss_count"] = len(result.losses)
            
        logger.log_summary(summary)
    finally:
        logger.close()

    metrics_data = {
        "episodes": result.tracker.episodes_arr,
        "rewards": result.tracker.rewards,
        "steps": result.tracker.steps_arr,
        "success": result.tracker.success_arr,
    }
    if args.algo == "dqn":
        metrics_data["losses"] = result.losses
        
    save_metrics(metrics_data, run_id)
    _save_path_json(result.paths, DIRS["metrics"] / f"{run_id}_paths.json")

    if not args.no_plots:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _save_plots(run_id, env, agent, result, args.algo)
        except Exception as exc:
            print(f"[warning] plots skipped: {exc}")

    print(json.dumps(summary, indent=2))


def _save_path_json(paths: dict[str, list[tuple[int, int]]], path: Path) -> None:
    serializable = {name: [[int(r), int(c)] for r, c in points] for name, points in paths.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _save_plots(run_id: str, env: DynamicMazeEnv, agent, result, algo: str) -> None:
    if algo == "qlearning":
        q_table = agent.q_table
    else:
        q_table = q_table_from_dqn(env, agent)
        
    # Take a snapshot of the initial layout
    env.reset()
    
    figures = {
        "learning_curve": plot_learning_curve(result.tracker.rewards, title=f"{algo.upper()} return"),
        "success_curve": plot_success_curve(result.tracker.success_arr, title=f"{algo.upper()} success rate"),
        "q_heatmap": plot_q_heatmap(q_table, env.maze, title=f"{algo.upper()} Q-value heatmap"),
        "policy_map": plot_policy_arrows(q_table, env.maze, title=f"{algo.upper()} greedy policy"),
    }
    
    # Add maze layout plot
    fig, _ = plot_dynamic_maze(env, title=f"Dynamic Maze Layout ({run_id})")
    figures["maze_layout"] = fig
    
    for name, fig in figures.items():
        fig.savefig(figure_path(run_id, name), dpi=150)


if __name__ == "__main__":
    main()
