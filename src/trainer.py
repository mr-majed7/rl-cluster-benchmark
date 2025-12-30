"""Sequential training pipeline for PPO on Procgen."""

import time
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm

from .buffer import RolloutBuffer
from .ppo import PPO
from .utils import Logger


class SequentialTrainer:
    """Sequential training pipeline for PPO."""

    def __init__(
        self,
        env_name: str = "procgen:procgen-coinrun-v0",
        num_envs: int = 64,
        n_steps: int = 256,
        total_timesteps: int = 25_000_000,
        batch_size: int = 2048,
        n_epochs: int = 3,
        learning_rate: float = 5e-4,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        log_interval: int = 1,
        save_interval: int = 100,
        eval_episodes: int = 10,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        seed: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the sequential trainer.

        Args:
            env_name: Name of the Procgen environment
            num_envs: Number of parallel environments
            n_steps: Number of steps to collect before update
            total_timesteps: Total number of environment steps
            batch_size: Batch size for training
            n_epochs: Number of epochs per update
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm
            log_interval: Logging interval (in updates)
            save_interval: Checkpoint saving interval (in updates)
            eval_episodes: Number of episodes for evaluation
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            seed: Random seed
            device: Device to use
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.total_timesteps = total_timesteps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_episodes = eval_episodes
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        # Set random seeds
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Create environments
        print(f"Creating {num_envs} environments: {env_name}")
        self.envs = gym.vector.SyncVectorEnv(
            [
                lambda: RecordEpisodeStatistics(gym.make(env_name))
                for _ in range(num_envs)
            ]
        )

        # Get observation shape and action space
        obs_shape = self.envs.single_observation_space.shape
        self.observation_shape = (
            obs_shape[2],
            obs_shape[0],
            obs_shape[1],
        )  # Convert to (C, H, W)
        self.num_actions = self.envs.single_action_space.n

        print(
            f"Observation shape: {self.observation_shape}, Number of actions: {self.num_actions}"
        )

        # Initialize PPO agent
        self.agent = PPO(
            observation_shape=self.observation_shape,
            num_actions=self.num_actions,
            num_envs=num_envs,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        # Initialize rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=n_steps,
            observation_shape=self.observation_shape,
            num_envs=num_envs,
            device=torch.device(device),
        )

        # Initialize logger
        self.logger = Logger(log_dir, algorithm="PPO_Sequential")

        # Training state
        self.global_step = 0
        self.update_step = 0

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Preprocess observations: normalize and transpose to (N, C, H, W)."""
        obs = obs.astype(np.float32) / 255.0
        obs = np.transpose(obs, (0, 3, 1, 2))  # (N, H, W, C) -> (N, C, H, W)
        return obs

    def collect_rollouts(self) -> Dict[str, float]:
        """Collect rollouts using current policy.

        Returns:
            Dictionary with episode statistics
        """
        episode_rewards = []
        episode_lengths = []

        obs, _ = self.envs.reset()
        obs = self._preprocess_obs(obs)

        for step in range(self.n_steps):
            # Select action
            action, value, log_prob = self.agent.predict(obs)

            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.envs.step(
                action.numpy()
            )
            done = np.logical_or(terminated, truncated)

            # Store transition
            self.buffer.add(obs, action, reward, done, value, log_prob)

            # Update observation
            obs = self._preprocess_obs(next_obs)

            # Update global step
            self.global_step += self.num_envs

            # Collect episode statistics
            if "final_info" in info:
                for final_info in info["final_info"]:
                    if final_info is not None and "episode" in final_info:
                        episode_rewards.append(final_info["episode"]["r"])
                        episode_lengths.append(final_info["episode"]["l"])

        # Compute returns and advantages
        with torch.no_grad():
            _, last_value, _ = self.agent.predict(obs)
        self.buffer.compute_returns_and_advantages(
            last_value, self.agent.gamma, self.agent.gae_lambda
        )

        stats = {}
        if episode_rewards:
            stats["episode_reward_mean"] = np.mean(episode_rewards)
            stats["episode_reward_std"] = np.std(episode_rewards)
            stats["episode_length_mean"] = np.mean(episode_lengths)
            stats["num_episodes"] = len(episode_rewards)

        return stats

    def train(self):
        """Run the training loop."""
        print(f"\nStarting training for {self.total_timesteps:,} timesteps")
        print(f"Device: {self.device}")
        print(f"Updates per rollout: {self.n_epochs}")
        print(f"Steps per rollout: {self.n_steps}")
        print(
            f"Total updates: {self.total_timesteps // (self.n_steps * self.num_envs)}"
        )
        print("-" * 60)

        num_updates = self.total_timesteps // (self.n_steps * self.num_envs)
        start_time = time.time()

        for update in tqdm(range(num_updates), desc="Training"):
            update_start_time = time.time()

            # Collect rollouts
            rollout_stats = self.collect_rollouts()

            # Train on collected data
            train_stats = self.agent.train_step(
                self.buffer, self.batch_size, self.n_epochs
            )

            # Reset buffer
            self.buffer.reset()

            # Update counter
            self.update_step += 1
            update_time = time.time() - update_start_time

            # Logging
            if self.update_step % self.log_interval == 0:
                fps = (self.n_steps * self.num_envs) / update_time
                elapsed_time = time.time() - start_time

                log_data = {
                    "time/fps": fps,
                    "time/total_timesteps": self.global_step,
                    "time/elapsed_time": elapsed_time,
                    "train/policy_loss": train_stats["policy_loss"],
                    "train/value_loss": train_stats["value_loss"],
                    "train/entropy_loss": train_stats["entropy_loss"],
                    "train/total_loss": train_stats["total_loss"],
                    "train/clip_fraction": train_stats["clip_fraction"],
                    "train/approx_kl": train_stats["approx_kl"],
                }

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
                        f"Update {self.update_step}/{num_updates} | "
                        f"Step {self.global_step:,}/{self.total_timesteps:,} | "
                        f"FPS: {fps:.0f} | "
                        f"Reward: {rollout_stats['episode_reward_mean']:.2f} ± {rollout_stats['episode_reward_std']:.2f}"
                    )

                self.logger.log(log_data, self.global_step)

            # Save checkpoint
            if self.update_step % self.save_interval == 0:
                checkpoint_path = (
                    f"{self.checkpoint_dir}/ppo_sequential_step_{self.global_step}.pt"
                )
                self.agent.save(checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")

        # Final save
        final_path = f"{self.checkpoint_dir}/ppo_sequential_final.pt"
        self.agent.save(final_path)
        print(f"\nTraining complete! Final model saved: {final_path}")
        print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")

        self.envs.close()
        self.logger.close()
