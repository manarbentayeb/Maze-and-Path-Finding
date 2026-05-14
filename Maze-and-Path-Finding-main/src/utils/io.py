"""
io.py
-----
Save and load utilities for models and results.

Keeps all file paths consistent across the project.

Convention
----------
  models/qlearning/<run_id>.npy        — Q-table (numpy array)
  models/dqn/<run_id>.pt               — DQN weights (torch state_dict)
  results/metrics/<run_id>_metrics.npz — raw arrays (rewards, steps, etc.)
  results/logs/<run_id>.csv            — per-episode CSV (written by logger)
  results/plots/<run_id>_*.png         — figures
"""

from pathlib import Path
from typing  import Any, Dict
import numpy as np


# ──────────────────────── paths ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]   # project root

DIRS = {
    "models_q"  : ROOT / "models" / "qlearning",
    "models_dqn": ROOT / "models" / "dqn",
    "metrics"   : ROOT / "results" / "metrics",
    "logs"      : ROOT / "results" / "logs",
    "plots"     : ROOT / "results" / "plots",
    "videos"    : ROOT / "results" / "videos",
}


def ensure_dirs() -> None:
    """Create all result directories if they don't exist."""
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


# ──────────────────────── Q-table ─────────────────────────────────────────

def save_qtable(Q: np.ndarray, run_id: str) -> Path:
    """Save a Q-table as a .npy file. Returns the path."""
    DIRS["models_q"].mkdir(parents=True, exist_ok=True)
    path = DIRS["models_q"] / f"{run_id}.npy"
    np.save(path, Q)
    print(f"  [io] Q-table saved -> {path}")
    return path


def load_qtable(run_id: str) -> np.ndarray:
    """Load a previously saved Q-table."""
    path = DIRS["models_q"] / f"{run_id}.npy"
    Q = np.load(path)
    print(f"  [io] Q-table loaded <- {path}  shape={Q.shape}")
    return Q


# ──────────────────────── DQN ──────────────────────────────────────────────

def save_dqn(model: Any, run_id: str) -> Path:
    """
    Save a PyTorch model's state_dict.
    `model` can be any nn.Module.
    """
    import torch
    DIRS["models_dqn"].mkdir(parents=True, exist_ok=True)
    path = DIRS["models_dqn"] / f"{run_id}.pt"
    torch.save(model.state_dict(), path)
    print(f"  [io] DQN model saved -> {path}")
    return path


def load_dqn(model: Any, run_id: str) -> Any:
    """Load weights into an existing model instance. Returns the model."""
    import torch
    path = DIRS["models_dqn"] / f"{run_id}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"  [io] DQN model loaded <- {path}")
    return model


# ──────────────────────── metrics arrays ──────────────────────────────────

def save_metrics(data: Dict[str, np.ndarray], run_id: str) -> Path:
    """
    Save raw metric arrays (rewards, steps, success) to a .npz file.

    Example
    -------
        save_metrics({
            "rewards": tracker.rewards,
            "steps"  : tracker.steps_arr,
            "success": tracker.success_arr,
        }, "qlearning_10x10_seed42")
    """
    DIRS["metrics"].mkdir(parents=True, exist_ok=True)
    path = DIRS["metrics"] / f"{run_id}_metrics.npz"
    np.savez(path, **data)
    print(f"  [io] metrics saved -> {path}")
    return path


def load_metrics(run_id: str) -> Dict[str, np.ndarray]:
    """Load metric arrays back as a dict of numpy arrays."""
    path = DIRS["metrics"] / f"{run_id}_metrics.npz"
    npz  = np.load(path)
    data = {k: npz[k] for k in npz.files}
    print(f"  [io] metrics loaded <- {path}  keys={list(data.keys())}")
    return data


# ──────────────────────── figures ─────────────────────────────────────────

def figure_path(run_id: str, name: str, ext: str = "png") -> Path:
    """
    Returns the canonical path for a plot file.

    Usage:
        fig.savefig(figure_path("qlearning_10x10_seed42", "learning_curve"))
    """
    DIRS["plots"].mkdir(parents=True, exist_ok=True)
    return DIRS["plots"] / f"{run_id}_{name}.{ext}"
