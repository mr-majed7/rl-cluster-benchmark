#!/usr/bin/env python3
"""Timed parallel training script for benchmarking parallel vs sequential."""
import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from datetime import datetime

import torch

from src.config import load_config, save_config
from src.parallel_trainer import ParallelTrainer


class TimedParallelTrainer:
    """Wrapper for timed parallel training with metrics collection."""

    def __init__(self, trainer, max_duration_hours, output_dir):
        self.trainer = trainer
        self.max_duration_seconds = max_duration_hours * 3600
        self.output_dir = output_dir
        self.start_time = None
        self.interrupted = False

        # Metrics to collect
        self.metrics = {
            "training_mode": "parallel",
            "num_workers": trainer.num_workers,
            "total_envs": trainer.total_envs,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            "duration_hours": 0,
            "total_timesteps": 0,
            "total_updates": 0,
            "final_fps": 0,
            "average_fps": 0,
            "config": {},
            "performance": {
                "initial_reward": None,
                "final_reward": None,
                "best_reward": None,
            },
            "resource_usage": {
                "peak_memory_gb": 0,
            },
        }

        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print("\n\n⚠️  Training interrupted! Saving results...")
        self.interrupted = True
        self.trainer.stop_event.set()

    def train(self):
        """Run timed training."""
        self.start_time = time.time()
        self.metrics["start_time"] = datetime.now().isoformat()

        print("\n" + "=" * 70)
        print("TIMED TRAINING - Parallel Mode")
        print("=" * 70)
        print(f"Duration: {self.max_duration_seconds / 3600:.2f} hours")
        print(f"Workers: {self.trainer.num_workers}")
        print(f"Total environments: {self.trainer.total_envs}")
        print(f"Output directory: {self.output_dir}")
        print(f"Start time: {self.metrics['start_time']}")
        print("=" * 70 + "\n")

        # Start workers
        self.trainer.start_workers()

        # Training loop with time limit
        num_updates = self.trainer.total_timesteps // (
            self.trainer.n_steps * self.trainer.total_envs
        )
        start_time = time.time()

        from tqdm import tqdm

        try:
            for update_idx in tqdm(range(num_updates), desc="Training"):
                # Check time limit
                elapsed = time.time() - start_time
                if elapsed >= self.max_duration_seconds or self.interrupted:
                    print(f"\n⏱️  Time limit reached: {elapsed / 3600:.2f} hours")
                    break

                update_start_time = time.time()

                # Collect rollouts from workers
                rollout_data, rollout_stats = self.trainer.collect_rollouts()

                # Create buffer
                from src.buffer import RolloutBuffer

                buffer = RolloutBuffer(
                    buffer_size=self.trainer.n_steps,
                    observation_shape=self.trainer.observation_shape,
                    num_envs=self.trainer.total_envs,
                    device=torch.device(self.trainer.device),
                )

                # Fill buffer
                for t in range(self.trainer.n_steps):
                    buffer.add(
                        rollout_data["observations"][t],
                        torch.from_numpy(rollout_data["actions"][t]),
                        rollout_data["rewards"][t],
                        rollout_data["dones"][t],
                        torch.from_numpy(rollout_data["values"][t]),
                        torch.from_numpy(rollout_data["log_probs"][t]),
                    )

                # Compute returns
                buffer.compute_returns_and_advantages(
                    torch.from_numpy(rollout_data["last_values"]),
                    self.trainer.agent.gamma,
                    self.trainer.agent.gae_lambda,
                )

                # Train
                train_stats = self.trainer.agent.train_step(
                    buffer, self.trainer.batch_size, self.trainer.n_epochs
                )

                # Update counters
                self.trainer.update_step += 1
                update_time = time.time() - update_start_time

                # Logging
                if self.trainer.update_step % self.trainer.log_interval == 0:
                    fps = (self.trainer.n_steps * self.trainer.total_envs) / update_time
                    elapsed_time = time.time() - start_time

                    log_data = {
                        "time/fps": fps,
                        "time/total_timesteps": self.trainer.global_step,
                        "time/elapsed_time": elapsed_time,
                        "train/policy_loss": train_stats["policy_loss"],
                        "train/value_loss": train_stats["value_loss"],
                        "train/entropy_loss": train_stats["entropy_loss"],
                        "train/total_loss": train_stats["total_loss"],
                        "train/clip_fraction": train_stats["clip_fraction"],
                        "train/approx_kl": train_stats["approx_kl"],
                    }

                    # Memory stats
                    if self.trainer.memory_monitor:
                        self.trainer.memory_monitor.update()
                        mem_stats = self.trainer.memory_monitor.get_stats()
                        log_data["system/memory_gb"] = mem_stats["current_memory_gb"]
                        log_data["system/peak_memory_gb"] = mem_stats["peak_memory_gb"]

                    if rollout_stats:
                        log_data.update(
                            {
                                "rollout/ep_reward_mean": rollout_stats[
                                    "episode_reward_mean"
                                ],
                                "rollout/ep_reward_std": rollout_stats[
                                    "episode_reward_std"
                                ],
                                "rollout/ep_length_mean": rollout_stats[
                                    "episode_length_mean"
                                ],
                            }
                        )

                        tqdm.write(
                            f"Update {self.trainer.update_step}/{num_updates} | "
                            f"Step {self.trainer.global_step:,} | "
                            f"FPS: {fps:.0f} | "
                            f"Time: {elapsed_time / 3600:.2f}h | "
                            f"Reward: {rollout_stats['episode_reward_mean']:.2f} ± {rollout_stats['episode_reward_std']:.2f}"
                        )

                    self.trainer.logger.log(log_data, self.trainer.global_step)

                # Save checkpoint
                if self.trainer.update_step % self.trainer.save_interval == 0:
                    os.makedirs(self.trainer.checkpoint_dir, exist_ok=True)
                    checkpoint_path = f"{self.trainer.checkpoint_dir}/ppo_parallel_step_{self.trainer.global_step}.pt"
                    self.trainer.agent.save(checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")

            # Final save
            os.makedirs(self.trainer.checkpoint_dir, exist_ok=True)
            final_path = f"{self.trainer.checkpoint_dir}/ppo_parallel_final.pt"
            self.trainer.agent.save(final_path)
            print(f"\nTraining complete! Final model saved: {final_path}")
            print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user!")
        finally:
            self.trainer.stop_workers()
            self.trainer.logger.close()
            self._collect_final_metrics()
            self._save_results()

    def _collect_final_metrics(self):
        """Collect final metrics."""
        end_time = time.time()
        duration = end_time - self.start_time

        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["duration_seconds"] = duration
        self.metrics["duration_hours"] = duration / 3600
        self.metrics["total_timesteps"] = self.trainer.global_step
        self.metrics["total_updates"] = self.trainer.update_step

        # Read metrics from log file
        metrics_file = os.path.join(self.trainer.logger.log_dir, "metrics.jsonl")
        if os.path.exists(metrics_file):
            fps_values = []
            rewards = []

            with open(metrics_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if "time/fps" in data:
                        fps_values.append(data["time/fps"])
                    if "rollout/ep_reward_mean" in data:
                        rewards.append(data["rollout/ep_reward_mean"])

            if fps_values:
                self.metrics["final_fps"] = fps_values[-1]
                self.metrics["average_fps"] = sum(fps_values) / len(fps_values)

            if rewards:
                self.metrics["performance"]["initial_reward"] = rewards[0]
                self.metrics["performance"]["final_reward"] = rewards[-1]
                self.metrics["performance"]["best_reward"] = max(rewards)

        # Get memory stats
        if self.trainer.memory_monitor:
            mem_stats = self.trainer.memory_monitor.get_stats()
            self.metrics["resource_usage"]["peak_memory_gb"] = mem_stats[
                "peak_memory_gb"
            ]

    def _save_results(self):
        """Save results."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save metrics JSON
        metrics_path = os.path.join(self.output_dir, "benchmark_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Duration: {self.metrics['duration_hours']:.2f} hours")
        print(f"Workers: {self.metrics['num_workers']}")
        print(f"Total Environments: {self.metrics['total_envs']}")
        print(f"Total Timesteps: {self.metrics['total_timesteps']:,}")
        print(f"Total Updates: {self.metrics['total_updates']}")
        print(f"Average FPS: {self.metrics['average_fps']:.0f}")
        print(f"Final FPS: {self.metrics['final_fps']:.0f}")

        if self.metrics["performance"]["final_reward"]:
            print("\nPerformance:")
            print(
                f"  Initial Reward: {self.metrics['performance']['initial_reward']:.2f}"
            )
            print(f"  Final Reward: {self.metrics['performance']['final_reward']:.2f}")
            print(f"  Best Reward: {self.metrics['performance']['best_reward']:.2f}")

        print("\nResource Usage:")
        print(
            f"  Peak Memory: {self.metrics['resource_usage']['peak_memory_gb']:.2f} GB"
        )

        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Logs: {self.trainer.logger.log_dir}")
        print(f"  - Checkpoints: {self.trainer.checkpoint_dir}")
        print("=" * 70 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Timed parallel training for benchmarking"
    )

    # Time limit
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Training duration in hours",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmarks/parallel",
        help="Output directory",
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="config/ppo_parallel.yaml",
        help="Config file",
    )

    # Environment
    parser.add_argument("--env", type=str, help="Environment name")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Parallel settings
    parser.add_argument("--num-workers", type=int, help="Number of worker processes")
    parser.add_argument(
        "--num-envs-per-worker", type=int, help="Environments per worker"
    )

    # Training
    parser.add_argument("--n-steps", type=int, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, help="Batch size")

    # Hardware
    parser.add_argument("--num-threads", type=int, help="CPU threads for main process")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {}

    # Override with command line args
    if args.env:
        config.setdefault("env", {})["name"] = args.env
    if args.seed is not None:
        config.setdefault("env", {})["seed"] = args.seed
    if args.num_workers:
        config.setdefault("parallel", {})["num_workers"] = args.num_workers
    if args.num_envs_per_worker:
        config.setdefault("parallel", {})[
            "num_envs_per_worker"
        ] = args.num_envs_per_worker
    if args.n_steps:
        config.setdefault("training", {})["n_steps"] = args.n_steps
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.num_threads:
        config.setdefault("hardware", {})["num_threads"] = args.num_threads

    # Set device to CPU
    config.setdefault("hardware", {})["device"] = "cpu"

    # Extract parameters
    env_config = config.get("env", {})
    training_config = config.get("training", {})
    ppo_config = config.get("ppo", {})
    logging_config = config.get("logging", {})
    hardware_config = config.get("hardware", {})
    parallel_config = config.get("parallel", {})

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Update paths
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    log_dir = os.path.join(args.output_dir, "logs")

    # Save config
    save_config(config, os.path.join(args.output_dir, "config.yaml"))

    # Set very large timesteps (stopped by time)
    total_timesteps = 1_000_000_000

    print("\n" + "=" * 70)
    print("PPO Parallel Training - Timed Benchmark")
    print("=" * 70)
    print(f"Environment: {env_config.get('name', 'procgen-coinrun-v0')}")
    print(f"Workers: {parallel_config.get('num_workers', 4)}")
    print(f"Envs per worker: {parallel_config.get('num_envs_per_worker', 8)}")
    print(f"Duration: {args.duration} hours")
    print("Device: cpu")
    print(f"Output: {args.output_dir}")
    print("=" * 70 + "\n")

    # Initialize trainer
    trainer = ParallelTrainer(
        env_name=env_config.get("name", "procgen-coinrun-v0"),
        num_workers=parallel_config.get("num_workers", 4),
        num_envs_per_worker=parallel_config.get("num_envs_per_worker", 8),
        n_steps=training_config.get("n_steps", 128),
        total_timesteps=total_timesteps,
        batch_size=training_config.get("batch_size", 2048),
        n_epochs=training_config.get("n_epochs", 4),
        learning_rate=ppo_config.get("learning_rate", 5e-4),
        gamma=ppo_config.get("gamma", 0.999),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        log_interval=1,
        save_interval=logging_config.get("save_interval", 50),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        seed=env_config.get("seed"),
        device="cpu",
        num_threads=hardware_config.get("num_threads"),
    )

    # Create timed trainer
    timed_trainer = TimedParallelTrainer(trainer, args.duration, args.output_dir)

    # Store config
    timed_trainer.metrics["config"] = config

    # Start training
    timed_trainer.train()


if __name__ == "__main__":
    # Required for multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
