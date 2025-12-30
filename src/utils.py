"""Utility functions for training."""

import json
import os
from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Logger for training metrics using TensorBoard."""

    def __init__(self, log_dir: str, algorithm: str = "PPO"):
        """Initialize logger.

        Args:
            log_dir: Directory to save logs
            algorithm: Algorithm name for logging
        """
        self.log_dir = os.path.join(log_dir, algorithm)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.metrics_file = os.path.join(self.log_dir, "metrics.jsonl")

        print(f"Logging to: {self.log_dir}")

    def log(self, data: Dict[str, Any], step: int):
        """Log metrics.

        Args:
            data: Dictionary of metrics to log
            step: Global step number
        """
        # Log to TensorBoard
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

        # Log to JSON lines file
        data["step"] = step
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(data) + "\n")

    def close(self):
        """Close the logger."""
        self.writer.close()


def make_env(env_name: str, seed: int = None):
    """Create a single environment.

    Args:
        env_name: Name of the environment
        seed: Random seed

    Returns:
        Environment instance
    """
    import gymnasium as gym
    from gymnasium.wrappers import RecordEpisodeStatistics

    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)

    if seed is not None:
        env.reset(seed=seed)

    return env
