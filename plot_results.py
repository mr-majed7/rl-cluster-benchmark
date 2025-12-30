#!/usr/bin/env python3
"""Plot training results from logs."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument(
        "--log-file", type=str, required=True, help="Path to metrics.jsonl file"
    )
    parser.add_argument(
        "--output", type=str, default="training_curves.png", help="Output file path"
    )
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size")
    return parser.parse_args()


def smooth(data, window_size):
    """Apply moving average smoothing."""
    if len(data) < window_size:
        return data
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode="valid")


def load_metrics(log_file):
    """Load metrics from JSONL file."""
    metrics = []
    with open(log_file, "r") as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def plot_training_curves(metrics, output_path, smooth_window=10):
    """Plot training curves."""
    # Extract data
    steps = [m["step"] for m in metrics if "rollout/ep_reward_mean" in m]
    rewards = [
        m["rollout/ep_reward_mean"] for m in metrics if "rollout/ep_reward_mean" in m
    ]

    policy_losses = [m.get("train/policy_loss", 0) for m in metrics]
    value_losses = [m.get("train/value_loss", 0) for m in metrics]
    entropy_losses = [m.get("train/entropy_loss", 0) for m in metrics]

    all_steps = [m["step"] for m in metrics]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("PPO Training Curves", fontsize=16)

    # Plot rewards
    ax = axes[0, 0]
    if steps and rewards:
        ax.plot(steps, rewards, alpha=0.3, color="blue", label="Raw")
        if len(rewards) > smooth_window:
            smoothed_rewards = smooth(rewards, smooth_window)
            smoothed_steps = steps[smooth_window - 1 :]
            ax.plot(
                smoothed_steps,
                smoothed_rewards,
                color="blue",
                linewidth=2,
                label=f"Smoothed (window={smooth_window})",
            )
        ax.set_xlabel("Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Episode Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot policy loss
    ax = axes[0, 1]
    ax.plot(all_steps, policy_losses, alpha=0.6, color="red")
    if len(policy_losses) > smooth_window:
        smoothed = smooth(policy_losses, smooth_window)
        ax.plot(all_steps[smooth_window - 1 :], smoothed, color="red", linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Policy Loss")
    ax.grid(True, alpha=0.3)

    # Plot value loss
    ax = axes[1, 0]
    ax.plot(all_steps, value_losses, alpha=0.6, color="green")
    if len(value_losses) > smooth_window:
        smoothed = smooth(value_losses, smooth_window)
        ax.plot(all_steps[smooth_window - 1 :], smoothed, color="green", linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Value Loss")
    ax.set_title("Value Loss")
    ax.grid(True, alpha=0.3)

    # Plot entropy loss
    ax = axes[1, 1]
    ax.plot(all_steps, entropy_losses, alpha=0.6, color="purple")
    if len(entropy_losses) > smooth_window:
        smoothed = smooth(entropy_losses, smooth_window)
        ax.plot(all_steps[smooth_window - 1 :], smoothed, color="purple", linewidth=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Entropy Loss")
    ax.set_title("Entropy Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Print summary statistics
    if rewards:
        print("\nTraining Summary:")
        print(f"  Total steps: {steps[-1]:,}")
        print(f"  Initial reward: {rewards[0]:.2f}")
        print(f"  Final reward: {rewards[-1]:.2f}")
        print(f"  Max reward: {max(rewards):.2f}")
        print(f"  Mean reward (last 20%): {np.mean(rewards[-len(rewards)//5:]):.2f}")


def main():
    """Main function."""
    args = parse_args()

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        print("\nAvailable log files:")
        log_dir = Path("logs")
        if log_dir.exists():
            for f in log_dir.rglob("metrics.jsonl"):
                print(f"  {f}")
        return

    print(f"Loading metrics from: {log_file}")
    metrics = load_metrics(log_file)
    print(f"Loaded {len(metrics)} data points")

    if not metrics:
        print("No metrics found in log file!")
        return

    plot_training_curves(metrics, args.output, args.smooth)


if __name__ == "__main__":
    main()
