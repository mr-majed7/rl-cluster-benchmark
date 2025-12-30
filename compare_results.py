#!/usr/bin/env python3
"""Compare sequential vs parallel training results."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_benchmark_metrics(benchmark_dir: str) -> Dict:
    """Load benchmark metrics from directory."""
    metrics_path = os.path.join(benchmark_dir, "benchmark_metrics.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, "r") as f:
        return json.load(f)


def load_training_logs(log_dir: str) -> List[Dict]:
    """Load training logs from metrics.jsonl file."""
    # Try different possible paths
    possible_paths = [
        Path(log_dir) / "logs" / "PPO_Sequential" / "metrics.jsonl",
        Path(log_dir) / "logs" / "PPO_Parallel" / "metrics.jsonl",
        Path(log_dir) / "PPO_Sequential" / "metrics.jsonl",
        Path(log_dir) / "PPO_Parallel" / "metrics.jsonl",
    ]

    for metrics_file in possible_paths:
        if metrics_file.exists():
            logs = []
            with open(metrics_file, "r") as f:
                for line in f:
                    logs.append(json.loads(line))
            return logs

    return []


def print_comparison(seq_metrics: Dict, par_metrics: Dict = None):
    """Print comparison table of metrics."""
    print("\n" + "=" * 80)
    print("TRAINING COMPARISON")
    print("=" * 80)

    if par_metrics:
        print(f"{'Metric':<40} {'Sequential':<20} {'Parallel':<20}")
        print("-" * 80)

        # Time metrics
        print(
            f"{'Duration (hours)':<40} {seq_metrics['duration_hours']:<20.2f} {par_metrics['duration_hours']:<20.2f}"
        )
        print(
            f"{'Total Timesteps':<40} {seq_metrics['total_timesteps']:<20,} {par_metrics['total_timesteps']:<20,}"
        )
        print(
            f"{'Total Updates':<40} {seq_metrics['total_updates']:<20,} {par_metrics['total_updates']:<20,}"
        )

        # Performance metrics
        print(f"\n{'Performance Metrics':<40}")
        print("-" * 80)
        print(
            f"{'Average FPS':<40} {seq_metrics['average_fps']:<20.0f} {par_metrics['average_fps']:<20.0f}"
        )
        speedup = (
            par_metrics["average_fps"] / seq_metrics["average_fps"]
            if seq_metrics["average_fps"] > 0
            else 0
        )
        print(f"{'Speedup (Parallel/Sequential)':<40} {'-':<20} {speedup:<20.2f}x")

        print(
            f"{'Final FPS':<40} {seq_metrics['final_fps']:<20.0f} {par_metrics['final_fps']:<20.0f}"
        )

        # Reward metrics
        if (
            seq_metrics["performance"]["final_reward"]
            and par_metrics["performance"]["final_reward"]
        ):
            print(f"\n{'Reward Metrics':<40}")
            print("-" * 80)
            print(
                f"{'Initial Reward':<40} {seq_metrics['performance']['initial_reward']:<20.2f} {par_metrics['performance']['initial_reward']:<20.2f}"
            )
            print(
                f"{'Final Reward':<40} {seq_metrics['performance']['final_reward']:<20.2f} {par_metrics['performance']['final_reward']:<20.2f}"
            )
            print(
                f"{'Best Reward':<40} {seq_metrics['performance']['best_reward']:<20.2f} {par_metrics['performance']['best_reward']:<20.2f}"
            )

        # Resource metrics
        print(f"\n{'Resource Usage':<40}")
        print("-" * 80)
        print(
            f"{'Peak Memory (GB)':<40} {seq_metrics['resource_usage']['peak_memory_gb']:<20.2f} {par_metrics['resource_usage']['peak_memory_gb']:<20.2f}"
        )

        # Efficiency metrics
        print(f"\n{'Efficiency Metrics':<40}")
        print("-" * 80)
        timesteps_per_hour_seq = (
            seq_metrics["total_timesteps"] / seq_metrics["duration_hours"]
        )
        timesteps_per_hour_par = (
            par_metrics["total_timesteps"] / par_metrics["duration_hours"]
        )
        print(
            f"{'Timesteps/Hour':<40} {timesteps_per_hour_seq:<20,.0f} {timesteps_per_hour_par:<20,.0f}"
        )
        efficiency_gain = (timesteps_per_hour_par / timesteps_per_hour_seq - 1) * 100
        print(f"{'Efficiency Gain (%)':<40} {'-':<20} {efficiency_gain:<20.1f}%")

    else:
        # Print sequential only
        print(f"{'Metric':<40} {'Value':<20}")
        print("-" * 80)

        print(f"{'Duration (hours)':<40} {seq_metrics['duration_hours']:<20.2f}")
        print(f"{'Total Timesteps':<40} {seq_metrics['total_timesteps']:<20,}")
        print(f"{'Total Updates':<40} {seq_metrics['total_updates']:<20,}")
        print(f"{'Average FPS':<40} {seq_metrics['average_fps']:<20.0f}")
        print(f"{'Final FPS':<40} {seq_metrics['final_fps']:<20.0f}")

        if seq_metrics["performance"]["final_reward"]:
            print(f"\n{'Reward Metrics':<40}")
            print("-" * 80)
            print(
                f"{'Initial Reward':<40} {seq_metrics['performance']['initial_reward']:<20.2f}"
            )
            print(
                f"{'Final Reward':<40} {seq_metrics['performance']['final_reward']:<20.2f}"
            )
            print(
                f"{'Best Reward':<40} {seq_metrics['performance']['best_reward']:<20.2f}"
            )

        print(f"\n{'Resource Usage':<40}")
        print("-" * 80)
        print(
            f"{'Peak Memory (GB)':<40} {seq_metrics['resource_usage']['peak_memory_gb']:<20.2f}"
        )

        timesteps_per_hour = (
            seq_metrics["total_timesteps"] / seq_metrics["duration_hours"]
        )
        print(f"\n{'Efficiency':<40}")
        print("-" * 80)
        print(f"{'Timesteps/Hour':<40} {timesteps_per_hour:<20,.0f}")

    print("=" * 80 + "\n")


def plot_comparison(
    seq_dir: str, par_dir: str = None, output_path: str = "comparison.png"
):
    """Plot comparison charts."""
    seq_logs = load_training_logs(seq_dir)

    if not seq_logs:
        print("Warning: No training logs found for sequential training")
        return

    # Extract data
    seq_steps = [log.get("step", 0) for log in seq_logs]
    seq_rewards = [
        log.get("rollout/ep_reward_mean")
        for log in seq_logs
        if "rollout/ep_reward_mean" in log
    ]
    seq_fps = [log.get("time/fps") for log in seq_logs if "time/fps" in log]
    seq_policy_loss = [
        log.get("train/policy_loss") for log in seq_logs if "train/policy_loss" in log
    ]

    seq_reward_steps = [
        log.get("step", 0) for log in seq_logs if "rollout/ep_reward_mean" in log
    ]

    # Create figure
    if par_dir:
        par_logs = load_training_logs(par_dir)
        par_steps = [log.get("step", 0) for log in par_logs]
        par_rewards = [
            log.get("rollout/ep_reward_mean")
            for log in par_logs
            if "rollout/ep_reward_mean" in log
        ]
        par_fps = [log.get("time/fps") for log in par_logs if "time/fps" in log]
        par_policy_loss = [
            log.get("train/policy_loss")
            for log in par_logs
            if "train/policy_loss" in log
        ]

        par_reward_steps = [
            log.get("step", 0) for log in par_logs if "rollout/ep_reward_mean" in log
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Sequential vs Parallel Training Comparison", fontsize=16)

        # Rewards
        ax = axes[0, 0]
        ax.plot(seq_reward_steps, seq_rewards, label="Sequential", alpha=0.7)
        ax.plot(par_reward_steps, par_rewards, label="Parallel", alpha=0.7)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Episode Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # FPS
        ax = axes[0, 1]
        ax.plot(seq_steps[: len(seq_fps)], seq_fps, label="Sequential", alpha=0.7)
        ax.plot(par_steps[: len(par_fps)], par_fps, label="Parallel", alpha=0.7)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("FPS")
        ax.set_title("Training Speed (FPS)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Policy Loss
        ax = axes[1, 0]
        ax.plot(
            seq_steps[: len(seq_policy_loss)],
            seq_policy_loss,
            label="Sequential",
            alpha=0.7,
        )
        ax.plot(
            par_steps[: len(par_policy_loss)],
            par_policy_loss,
            label="Parallel",
            alpha=0.7,
        )
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative timesteps over time
        ax = axes[1, 1]
        seq_times = [
            log.get("time/elapsed_time", 0) / 3600 for log in seq_logs
        ]  # Convert to hours
        par_times = [log.get("time/elapsed_time", 0) / 3600 for log in par_logs]
        ax.plot(seq_times, seq_steps, label="Sequential", alpha=0.7)
        ax.plot(par_times, par_steps, label="Parallel", alpha=0.7)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Total Timesteps")
        ax.set_title("Training Progress Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    else:
        # Sequential only
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Sequential Training Results", fontsize=16)

        # Rewards
        ax = axes[0, 0]
        ax.plot(seq_reward_steps, seq_rewards, alpha=0.7, color="blue")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Episode Rewards")
        ax.grid(True, alpha=0.3)

        # FPS
        ax = axes[0, 1]
        ax.plot(seq_steps[: len(seq_fps)], seq_fps, alpha=0.7, color="green")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("FPS")
        ax.set_title("Training Speed (FPS)")
        ax.grid(True, alpha=0.3)

        # Policy Loss
        ax = axes[1, 0]
        ax.plot(
            seq_steps[: len(seq_policy_loss)], seq_policy_loss, alpha=0.7, color="red"
        )
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Loss")
        ax.grid(True, alpha=0.3)

        # Timesteps over time
        ax = axes[1, 1]
        seq_times = [log.get("time/elapsed_time", 0) / 3600 for log in seq_logs]
        ax.plot(seq_times, seq_steps, alpha=0.7, color="purple")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Total Timesteps")
        ax.set_title("Training Progress Over Time")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare sequential vs parallel training results"
    )

    parser.add_argument(
        "--sequential",
        type=str,
        required=True,
        help="Path to sequential benchmark directory",
    )
    parser.add_argument(
        "--parallel",
        type=str,
        default=None,
        help="Path to parallel benchmark directory (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison.png",
        help="Output path for comparison plot",
    )

    args = parser.parse_args()

    # Load metrics
    print(f"\nLoading sequential metrics from: {args.sequential}")
    seq_metrics = load_benchmark_metrics(args.sequential)

    par_metrics = None
    if args.parallel:
        print(f"Loading parallel metrics from: {args.parallel}")
        par_metrics = load_benchmark_metrics(args.parallel)

    # Print comparison
    print_comparison(seq_metrics, par_metrics)

    # Plot comparison
    plot_comparison(args.sequential, args.parallel, args.output)

    # Generate summary report
    report_path = "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SEQUENTIAL VS PARALLEL TRAINING COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Sequential Directory: {args.sequential}\n")
        if par_metrics:
            f.write(f"Parallel Directory: {args.parallel}\n")
        f.write("\n")

        f.write("SEQUENTIAL TRAINING\n")
        f.write("-" * 80 + "\n")
        f.write(f"Duration: {seq_metrics['duration_hours']:.2f} hours\n")
        f.write(f"Total Timesteps: {seq_metrics['total_timesteps']:,}\n")
        f.write(f"Average FPS: {seq_metrics['average_fps']:.0f}\n")
        f.write(
            f"Peak Memory: {seq_metrics['resource_usage']['peak_memory_gb']:.2f} GB\n"
        )

        if par_metrics:
            f.write("\n")
            f.write("PARALLEL TRAINING\n")
            f.write("-" * 80 + "\n")
            f.write(f"Duration: {par_metrics['duration_hours']:.2f} hours\n")
            f.write(f"Total Timesteps: {par_metrics['total_timesteps']:,}\n")
            f.write(f"Average FPS: {par_metrics['average_fps']:.0f}\n")
            f.write(
                f"Peak Memory: {par_metrics['resource_usage']['peak_memory_gb']:.2f} GB\n"
            )

            f.write("\n")
            f.write("COMPARISON\n")
            f.write("-" * 80 + "\n")
            speedup = par_metrics["average_fps"] / seq_metrics["average_fps"]
            f.write(f"Speedup (FPS): {speedup:.2f}x\n")

            timesteps_per_hour_seq = (
                seq_metrics["total_timesteps"] / seq_metrics["duration_hours"]
            )
            timesteps_per_hour_par = (
                par_metrics["total_timesteps"] / par_metrics["duration_hours"]
            )
            efficiency_gain = (
                timesteps_per_hour_par / timesteps_per_hour_seq - 1
            ) * 100
            f.write(f"Efficiency Gain: {efficiency_gain:.1f}%\n")

    print(f"\nComparison report saved to: {report_path}")


if __name__ == "__main__":
    main()
