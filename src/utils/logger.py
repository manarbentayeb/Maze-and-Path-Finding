"""Structured ASCII-safe logging for training runs."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict


class TrainingLogger:
    """Log training progress to console, CSV, and summary JSON."""

    def __init__(
        self,
        run_id: str = "run",
        output_dir: str = "results/logs",
        log_every: int = 100,
    ):
        self.run_id = run_id
        self.log_every = log_every
        self._start_time = time.time()
        self._episode_count = 0

        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._dir / f"{run_id}.csv"
        self._summary_path = self._dir / f"{run_id}_summary.json"

        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None
        self._csv_headers_written = False

        self._print_header()

    def log_episode(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        success: bool,
        **extra,
    ) -> None:
        self._episode_count += 1
        row = {
            "episode": episode,
            "total_reward": round(total_reward, 5),
            "steps": steps,
            "success": int(success),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in extra.items()},
        }

        if not self._csv_headers_written:
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=list(row.keys()),
                extrasaction="ignore",
            )
            self._csv_writer.writeheader()
            self._csv_headers_written = True

        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self.log_every and episode % self.log_every == 0:
            self._print_progress(episode, total_reward, steps, success, extra)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        summary["run_id"] = self.run_id
        summary["elapsed_sec"] = round(time.time() - self._start_time, 1)

        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        self._print_summary(summary)

    def close(self) -> None:
        if not self._csv_file.closed:
            self._csv_file.close()

    def _print_header(self) -> None:
        line = "-" * 70
        print(f"\n{line}")
        print(f"Training run: {self.run_id}")
        print(f"CSV log:      {self._csv_path}")
        print(line)
        print(f"{'Episode':>10}  {'Reward':>10}  {'Steps':>8}  {'Success':>8}  Extra")
        print(f"{'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 20}")

    def _print_progress(
        self,
        episode: int,
        total_reward: float,
        steps: int,
        success: bool,
        extra: dict,
    ) -> None:
        extra_s = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in extra.items()
        )
        elapsed = time.time() - self._start_time
        print(
            f"{episode:>10,}  {total_reward:>10.4f}  {steps:>8,}  "
            f"{int(success):>8}  {extra_s}  [{elapsed:.0f}s]"
        )

    def _print_summary(self, summary: dict) -> None:
        line = "-" * 70
        print(f"\n{line}")
        print(f"Summary - {self.run_id}")
        print(line)
        for k, v in summary.items():
            if k == "run_id":
                continue
            print(f"{k:<30} {v}")
        print(f"{line}\n")
