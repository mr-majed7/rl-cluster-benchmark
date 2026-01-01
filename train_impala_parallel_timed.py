#!/usr/bin/env python3
"""
Timed training script for IMPALA (parallel).
Runs training for a specified duration and collects benchmark metrics.
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from src.impala_parallel_trainer import ParallelIMPALATrainer


class TimedIMPALAParallelTrainer:
    """Wrapper for timed IMPALA parallel training with benchmarking."""

    def __init__(
        self,
        duration_hours: float,
        output_dir: str,
        config_path: str = "config/impala_parallel.yaml",
    ):
        """
        Initialize timed trainer.

        Args:
            duration_hours: Training duration in hours
            output_dir: Output directory for results
            config_path: Path to config file
        """
        self.duration_hours = duration_hours
        self.duration_seconds = duration_hours * 3600
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Override checkpoint and log directories
        self.config["logging"]["checkpoint_dir"] = str(self.output_dir / "checkpoints")
        self.config["logging"]["log_dir"] = str(self.output_dir / "logs")

        # Create trainer
        self.trainer = ParallelIMPALATrainer(
            env_name=self.config["env"]["name"],
            num_actors=self.config["parallel"]["num_actors"],
            num_envs_per_actor=self.config["parallel"]["num_envs_per_actor"],
            n_steps=self.config["training"]["n_steps"],
            total_timesteps=999_999_999,  # Set very high
            learning_rate=self.config["training"]["learning_rate"],
            gamma=self.config["training"]["gamma"],
            vtrace_clip_rho_threshold=self.config["impala"][
                "vtrace_clip_rho_threshold"
            ],
            vtrace_clip_pg_rho_threshold=self.config["impala"][
                "vtrace_clip_pg_rho_threshold"
            ],
            entropy_coef=self.config["impala"]["entropy_coef"],
            value_coef=self.config["impala"]["value_coef"],
            max_grad_norm=self.config["impala"]["max_grad_norm"],
            policy_update_frequency=self.config["parallel"]["policy_update_frequency"],
            log_interval=self.config["logging"]["log_interval"],
            save_interval=self.config["logging"]["save_interval"],
            checkpoint_dir=self.config["logging"]["checkpoint_dir"],
            log_dir=self.config["logging"]["log_dir"],
            device=self.config["hardware"]["device"],
            num_threads=self.config["hardware"]["num_threads"],
            seed=self.config["env"].get("seed"),
        )

        # Ensure checkpoint directory exists
        os.makedirs(self.trainer.checkpoint_dir, exist_ok=True)

        # Metrics
        self.metrics = {
            "duration_hours": duration_hours,
            "start_time": None,
            "end_time": None,
            "total_timesteps": 0,
            "total_updates": 0,
            "fps_history": [],
            "loss_history": [],
        }

    def train(self):
        """Run timed training."""
        print("\n" + "=" * 70)
        print("IMPALA Parallel Training - Timed Benchmark")
        print("=" * 70)
        print(f"Environment: {self.config['env']['name']}")
        print(f"Actors: {self.config['parallel']['num_actors']}")
        print(f"Envs per actor: {self.config['parallel']['num_envs_per_actor']}")
        print(f"Total environments: {self.trainer.total_envs}")
        print(f"Duration: {self.duration_hours} hours")
        print(f"Device: {self.config['hardware']['device']}")
        print(f"Output: {self.output_dir}")
        print("=" * 70 + "\n")

        self.metrics["start_time"] = datetime.now().isoformat()
        start_time = time.time()

        # Start actors
        self.trainer.start_actors()
        self.trainer.broadcast_policy()

        print("\n" + "=" * 70)
        print("TIMED TRAINING - Parallel Mode")
        print("=" * 70)
        print(f"Duration: {self.duration_hours:.2f} hours")
        print(f"Actors: {self.trainer.num_actors}")
        print(f"Total environments: {self.trainer.total_envs}")
        print(f"Output directory: {self.output_dir}")
        print(f"Start time: {self.metrics['start_time']}")
        print("=" * 70 + "\n")

        try:
            # Main training loop with time limit
            while (time.time() - start_time) < self.duration_seconds:
                # Get experience
                try:
                    experience = self.trainer.experience_queue.get(timeout=10.0)
                except:
                    continue

                # Train
                metrics = self.trainer.agent.train_step(
                    observations=experience["observations"],
                    actions=experience["actions"],
                    rewards=experience["rewards"],
                    dones=experience["dones"],
                    behavior_logits=experience["behavior_logits"],
                )

                self.trainer.update_step += 1

                # Update global step
                batch_timesteps = (
                    experience["actions"].shape[0] * experience["actions"].shape[1]
                )
                self.trainer.global_step += batch_timesteps

                # Update memory monitor
                if self.trainer.memory_monitor:
                    self.trainer.memory_monitor.update()

                # Broadcast policy
                if self.trainer.update_step % self.trainer.policy_update_frequency == 0:
                    self.trainer.broadcast_policy()

                # Record metrics
                elapsed = time.time() - start_time
                fps = self.trainer.global_step / elapsed if elapsed > 0 else 0
                self.metrics["fps_history"].append(fps)
                self.metrics["loss_history"].append(metrics["policy_loss"])

                # Logging
                if self.trainer.update_step % self.trainer.log_interval == 0:
                    remaining = self.duration_seconds - elapsed
                    remaining_str = str(timedelta(seconds=int(remaining)))

                    print(
                        f"[{elapsed/3600:.2f}h/{self.duration_hours:.2f}h] "
                        f"Step: {self.trainer.global_step:,} | "
                        f"Updates: {self.trainer.update_step} | "
                        f"FPS: {fps:.0f} | "
                        f"Loss: {metrics['policy_loss']:.4f} | "
                        f"Remaining: {remaining_str}"
                    )

                # Periodic checkpoint
                if self.trainer.update_step % 500 == 0:
                    checkpoint_path = (
                        self.trainer.checkpoint_dir
                        / f"impala_parallel_step_{self.trainer.global_step}.pt"
                    )
                    self.trainer.agent.save(str(checkpoint_path))

        finally:
            # Stop actors
            self.trainer.stop_actors()

        # Training finished
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["total_timesteps"] = self.trainer.global_step
        self.metrics["total_updates"] = self.trainer.update_step

        # Save final checkpoint
        final_path = self.trainer.checkpoint_dir / "impala_parallel_final.pt"
        self.trainer.agent.save(str(final_path))

        # Calculate final statistics
        elapsed = time.time() - start_time
        avg_fps = self.trainer.global_step / elapsed if elapsed > 0 else 0
        final_fps = (
            self.metrics["fps_history"][-1] if self.metrics["fps_history"] else 0
        )

        # Get memory stats
        peak_memory_gb = 0
        if self.trainer.memory_monitor:
            mem_stats = self.trainer.memory_monitor.get_stats()
            peak_memory_gb = mem_stats["peak_memory_gb"]

        # Save metrics
        summary = {
            "algorithm": "IMPALA",
            "mode": "parallel",
            "config": self.config,
            "duration_hours": self.duration_hours,
            "duration_seconds": elapsed,
            "total_timesteps": self.trainer.global_step,
            "total_updates": self.trainer.update_step,
            "avg_fps": avg_fps,
            "final_fps": final_fps,
            "peak_memory_gb": peak_memory_gb,
            "num_actors": self.trainer.num_actors,
            "num_envs_per_actor": self.trainer.num_envs_per_actor,
            "total_envs": self.trainer.total_envs,
            "start_time": self.metrics["start_time"],
            "end_time": self.metrics["end_time"],
        }

        # Save summary
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save detailed metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Print summary
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Duration: {self.duration_hours:.2f} hours")
        print(f"Actors: {self.trainer.num_actors}")
        print(f"Total Environments: {self.trainer.total_envs}")
        print(f"Total Timesteps: {self.trainer.global_step:,}")
        print(f"Total Updates: {self.trainer.update_step:,}")
        print(f"Average FPS: {avg_fps:.0f}")
        print(f"Final FPS: {final_fps:.0f}")

        print("\nResource Usage:")
        print(f"  Peak Memory: {peak_memory_gb:.2f} GB")

        print(f"\nResults saved to: {self.output_dir}")
        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run IMPALA parallel training for a specified duration"
    )
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Training duration in hours",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/impala_parallel.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Create timed trainer
    trainer = TimedIMPALAParallelTrainer(
        duration_hours=args.duration,
        output_dir=args.output_dir,
        config_path=args.config,
    )

    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
