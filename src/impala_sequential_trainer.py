"""
Sequential IMPALA Trainer for CPU-based training.
"""

import time
from pathlib import Path
from typing import Dict, Optional

import gym  # Old gym for procgen
import gymnasium
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

# Import procgen to register environments
try:
    import procgen  # noqa: F401
except ImportError:
    print("Warning: procgen not installed")

from .cpu_utils import CPUMemoryMonitor, print_cpu_info, setup_cpu_optimization
from .impala import IMPALA
from .utils import Logger


class SequentialIMPALATrainer:
    """
    Sequential IMPALA trainer that mimics actor-learner structure
    but runs in a single process.

    Maintains a replay buffer to simulate off-policy learning.
    """

    def __init__(
        self,
        env_name: str = "procgen-coinrun-v0",
        num_envs: int = 32,
        n_steps: int = 128,
        total_timesteps: int = 10_000_000,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        vtrace_clip_rho_threshold: float = 1.0,
        vtrace_clip_pg_rho_threshold: float = 1.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 40.0,
        log_interval: int = 10,
        save_interval: int = 100,
        checkpoint_dir: str = "./checkpoints/impala_sequential",
        log_dir: str = "./logs",
        device: str = "cpu",
        num_threads: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Sequential IMPALA trainer.

        Args:
            env_name: Procgen environment name
            num_envs: Number of parallel environments
            n_steps: Number of steps to collect per rollout
            total_timesteps: Total timesteps to train for
            learning_rate: Learning rate
            gamma: Discount factor
            vtrace_clip_rho_threshold: V-trace ρ clipping
            vtrace_clip_pg_rho_threshold: V-trace c clipping
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Max gradient norm
            log_interval: Logging frequency (in updates)
            save_interval: Checkpoint save frequency (in updates)
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            device: Device to use
            num_threads: Number of CPU threads
            seed: Random seed
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device

        # Setup CPU optimization
        if device == "cpu":
            if num_threads is None:
                num_threads = setup_cpu_optimization()
            else:
                setup_cpu_optimization(num_threads)

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create environments
        print(f"Creating {num_envs} environments: {env_name}")

        def make_env():
            # Use old gym for procgen, then wrap for gymnasium compatibility
            if "procgen" in env_name:
                old_env = gym.make(env_name)
                new_env = GymV21CompatibilityV0(env=old_env)
            else:
                new_env = gymnasium.make(env_name)
            return RecordEpisodeStatistics(new_env)

        self.envs = gymnasium.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

        # Get observation and action space info
        obs_shape = self.envs.single_observation_space.shape
        # Convert from (H, W, C) to (C, H, W) for PyTorch
        observation_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        num_actions = self.envs.single_action_space.n

        print(f"Observation shape: {observation_shape}")
        print(f"Number of actions: {num_actions}")

        # Initialize IMPALA agent
        self.agent = IMPALA(
            observation_shape=observation_shape,
            num_actions=num_actions,
            num_envs=num_envs,
            learning_rate=learning_rate,
            gamma=gamma,
            vtrace_clip_rho_threshold=vtrace_clip_rho_threshold,
            vtrace_clip_pg_rho_threshold=vtrace_clip_pg_rho_threshold,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            num_threads=num_threads,
        )

        # Logger
        self.logger = Logger(log_dir, "IMPALA_Sequential")

        # Statistics
        self.global_step = 0
        self.update_step = 0

        # CPU memory monitoring
        self.memory_monitor = CPUMemoryMonitor() if device == "cpu" else None

        # Print CPU info
        if device == "cpu":
            print_cpu_info()

    def collect_rollout(self) -> Dict:
        """
        Collect a rollout of experiences.

        Returns:
            Dictionary containing rollout data
        """
        # Storage for rollout
        observations = []
        actions = []
        rewards = []
        dones = []
        behavior_logits = []

        # Current observation (get from _observations attribute)
        obs = self.envs._observations

        # Collect n_steps of experience
        for _ in range(self.n_steps):
            # Transpose observation from (B, H, W, C) to (B, C, H, W) for PyTorch
            obs_transposed = np.transpose(obs, (0, 3, 1, 2))
            observations.append(obs_transposed)

            # Get action from behavior policy (current policy)
            action, logits, _ = self.agent.get_action(obs_transposed)

            actions.append(action)
            behavior_logits.append(logits)

            # Step environment
            obs, reward, terminated, truncated, info = self.envs.step(action)
            done = terminated | truncated

            rewards.append(reward)
            dones.append(done)

            self.global_step += self.num_envs

        # Add final observation (also transposed)
        obs_transposed = np.transpose(obs, (0, 3, 1, 2))
        observations.append(obs_transposed)

        # Convert lists to arrays
        rollout = {
            "observations": np.array(observations),  # [T+1, B, C, H, W]
            "actions": np.array(actions),  # [T, B]
            "rewards": np.array(rewards),  # [T, B]
            "dones": np.array(dones),  # [T, B]
            "behavior_logits": np.array(behavior_logits),  # [T, B, A]
        }

        return rollout

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Sequential IMPALA Training")
        print("=" * 60)
        print(f"Total timesteps: {self.total_timesteps:,}")
        print(f"Environments: {self.num_envs}")
        print(f"Rollout length: {self.n_steps}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")

        # Reset environments
        obs, _ = self.envs.reset()

        # Training metrics
        episode_rewards = []
        episode_lengths = []
        start_time = time.time()

        # Main training loop
        while self.global_step < self.total_timesteps:
            rollout_start = time.time()

            # Collect rollout
            rollout = self.collect_rollout()

            rollout_time = time.time() - rollout_start

            # Train on rollout
            train_start = time.time()

            metrics = self.agent.train_step(
                observations=rollout["observations"],
                actions=rollout["actions"],
                rewards=rollout["rewards"],
                dones=rollout["dones"],
                behavior_logits=rollout["behavior_logits"],
            )

            train_time = time.time() - train_start

            self.update_step += 1

            # Update memory monitor
            if self.memory_monitor:
                self.memory_monitor.update()

            # Logging
            if self.update_step % self.log_interval == 0:
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = self.global_step / elapsed if elapsed > 0 else 0

                # Log to TensorBoard
                self.logger.log(
                    "train/policy_loss", metrics["policy_loss"], self.global_step
                )
                self.logger.log(
                    "train/value_loss", metrics["value_loss"], self.global_step
                )
                self.logger.log("train/entropy", metrics["entropy"], self.global_step)
                self.logger.log(
                    "train/grad_norm", metrics["grad_norm"], self.global_step
                )
                self.logger.log(
                    "train/mean_advantage", metrics["mean_advantage"], self.global_step
                )
                self.logger.log(
                    "train/mean_value", metrics["mean_value"], self.global_step
                )
                self.logger.log("time/fps", fps, self.global_step)
                self.logger.log("time/rollout_time", rollout_time, self.global_step)
                self.logger.log("time/train_time", train_time, self.global_step)

                # Get episode statistics from info
                if hasattr(self.envs, "return_queue"):
                    while not self.envs.return_queue.empty():
                        ep_info = self.envs.return_queue.get()
                        episode_rewards.append(ep_info["r"])
                        episode_lengths.append(ep_info["l"])

                # Log episode statistics
                if episode_rewards:
                    mean_reward = np.mean(episode_rewards[-100:])
                    mean_length = np.mean(episode_lengths[-100:])

                    self.logger.log(
                        "rollout/ep_reward_mean", mean_reward, self.global_step
                    )
                    self.logger.log(
                        "rollout/ep_length_mean", mean_length, self.global_step
                    )

                # Memory usage
                if self.memory_monitor:
                    mem_stats = self.memory_monitor.get_stats()
                    self.logger.log(
                        "system/memory_gb", mem_stats["current_gb"], self.global_step
                    )

                # Console output
                print(
                    f"Step: {self.global_step:,} | "
                    f"Update: {self.update_step} | "
                    f"FPS: {fps:.0f} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f}"
                )

                if episode_rewards:
                    print(
                        f"  Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.0f}"
                    )

            # Save checkpoint
            if self.update_step % self.save_interval == 0:
                checkpoint_path = (
                    self.checkpoint_dir
                    / f"impala_sequential_step_{self.global_step}.pt"
                )
                self.agent.save(str(checkpoint_path))
                print(f"Saved checkpoint: {checkpoint_path}")

        # Save final checkpoint
        final_path = self.checkpoint_dir / "impala_sequential_final.pt"
        self.agent.save(str(final_path))
        print(f"\nTraining complete! Final checkpoint saved: {final_path}")

        # Print final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"Total timesteps: {self.global_step:,}")
        print(f"Total updates: {self.update_step}")
        print(f"Time elapsed: {elapsed:.2f}s ({elapsed/3600:.2f}h)")
        print(f"Average FPS: {self.global_step/elapsed:.0f}")

        if episode_rewards:
            print(
                f"Final mean reward (last 100 eps): {np.mean(episode_rewards[-100:]):.2f}"
            )
            print(f"Best reward: {max(episode_rewards):.2f}")

        if self.memory_monitor:
            mem_stats = self.memory_monitor.get_stats()
            print(f"Peak memory usage: {mem_stats['peak_gb']:.2f} GB")

        print("=" * 60)

        # Close environments
        self.envs.close()
