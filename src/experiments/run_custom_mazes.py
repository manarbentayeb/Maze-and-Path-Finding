"""Run tests & training on Weighted-Cost and Dynamic Obstacle custom mazes.

Example:
    python -m src.experiments.run_custom_mazes --env weighted --episodes 1000
    python -m src.experiments.run_custom_mazes --env dynamic --episodes 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.dqn_agent import DQNAgent
from src.env.weighted_maze_env import WeightedMazeEnv
from src.env.dynamic_maze_env import DynamicMazeEnv
from src.env.maze_generator import LEVEL_SIZES
from src.training.evaluator import evaluate_dqn
from src.training.trainer_dqn import train_dqn
from src.utils.io import DIRS, ensure_dirs, save_dqn, save_metrics
from src.utils.logger import TrainingLogger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on custom mazes.")
    parser.add_argument("--env", type=str, choices=["weighted", "dynamic"], required=True, help="Which env to run.")
    parser.add_argument("--size", type=int, default=10, help="Maze side length.")
    parser.add_argument("--level", choices=list(LEVEL_SIZES), default=None, help="Named maze level.")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # Custom Env specific params
    parser.add_argument("--mud-density", type=float, default=0.2, help="Percentage of free cells turned to mud (weighted only).")
    parser.add_argument("--mud-penalty", type=float, default=0.05, help="Extra step penalty for mud (weighted only).")
    parser.add_argument("--dynamic-prob", type=float, default=0.05, help="Probability of a cell flipping per step (dynamic only).")
    
    # DQN Params
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update-every", type=int, default=250)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=50)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_global_seed(args.seed)

    actual_size = LEVEL_SIZES[args.level] if args.level else args.size
    level_part = args.level or f"{actual_size}x{actual_size}"
    
    print(f"Initializing {args.env.capitalize()} Maze Environment...")
    if args.env == "weighted":
        run_id = f"dqn_weighted_{level_part}_{args.difficulty}_seed{args.seed}"
        env = WeightedMazeEnv(
            size=actual_size, 
            level=args.level, 
            difficulty=args.difficulty, 
            seed=args.seed,
            mud_density=args.mud_density,
            mud_penalty=args.mud_penalty
        )
    elif args.env == "dynamic":
        run_id = f"dqn_dynamic_{level_part}_{args.difficulty}_seed{args.seed}"
        env = DynamicMazeEnv(
            size=actual_size, 
            level=args.level, 
            difficulty=args.difficulty, 
            seed=args.seed,
            dynamic_prob=args.dynamic_prob
        )
        
    env.action_space.seed(args.seed)

    print(f"Goal position: {env.goal_pos}, Start position: {env.start_pos}, Grid Size: {env.size}x{env.size}")

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
        seed=args.seed,
        device=args.device,
    )

    logger = TrainingLogger(run_id=run_id, output_dir=str(DIRS["logs"]), log_every=args.log_every)
    print("Beginning training...")
    try:
        result = train_dqn(env=env, agent=agent, episodes=args.episodes, logger=logger)
        print("Training complete. Evaluating...")
        eval_result = evaluate_dqn(env, agent, episodes=100)
        
        summary = result.tracker.compute_summary()
        summary.update(
            {
                "env_type": args.env,
                "eval_success_rate": round(eval_result.success_rate, 4),
                "eval_mean_return": round(eval_result.mean_return, 4),
                "eval_mean_steps": round(eval_result.mean_steps, 2),
                "maze_size": env.size,
                "max_steps": env.max_steps,
            }
        )
        logger.log_summary(summary)
        print(f"Evaluation Results -> Success Rate: {eval_result.success_rate:.2f} | Mean Return: {eval_result.mean_return:.2f}")
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
    print(f"Successfully saved model and metrics under run ID: {run_id}")


if __name__ == "__main__":
    main()