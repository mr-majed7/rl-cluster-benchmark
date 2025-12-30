"""PPO algorithm implementation."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .buffer import RolloutBuffer
from .cpu_utils import setup_cpu_optimization
from .models import CNNActorCritic


class PPO:
    """Proximal Policy Optimization algorithm."""

    def __init__(
        self,
        observation_shape: tuple,
        num_actions: int,
        num_envs: int = 1,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        num_threads: int = None,
    ):
        """Initialize PPO agent.

        Args:
            observation_shape: Shape of observations (C, H, W)
            num_actions: Number of discrete actions
            num_envs: Number of parallel environments
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            clip_range_vf: Value function clipping parameter
            normalize_advantage: Whether to normalize advantages
            ent_coef: Entropy coefficient
            vf_coef: Value function loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on (default: cpu)
            num_threads: Number of threads for CPU training (None = auto)
        """
        self.device = torch.device(device)

        # Set CPU threading for optimal performance
        # (Skip if already configured by parent)
        if device == "cpu" and torch.get_num_threads() == 0:
            configured_threads = setup_cpu_optimization(num_threads)
            print(f"CPU optimized with {configured_threads} threads")

        self.num_actions = num_actions
        self.num_envs = num_envs

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Initialize network
        self.policy = CNNActorCritic(observation_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=learning_rate, eps=1e-5
        )

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> tuple:
        """Predict action for given observation.

        Args:
            obs: Observation array
            deterministic: If True, use argmax instead of sampling

        Returns:
            action: Selected action
            value: Value estimate
            log_prob: Log probability of action
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)

        with torch.no_grad():
            if deterministic:
                logits, value = self.policy(obs_tensor)
                action = torch.argmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(action)
            else:
                action, log_prob, _, value = self.policy.get_action_and_value(
                    obs_tensor
                )

        return action.cpu(), value.cpu(), log_prob.cpu()

    def train_step(
        self, buffer: RolloutBuffer, batch_size: int, n_epochs: int
    ) -> Dict[str, float]:
        """Perform a training step using data from the buffer.

        Args:
            buffer: Rollout buffer containing experiences
            batch_size: Batch size for training
            n_epochs: Number of epochs to train on the buffer

        Returns:
            Dictionary of training metrics
        """
        # Track metrics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        clip_fractions = []
        approx_kls = []

        for _ in range(n_epochs):
            for batch in buffer.get(batch_size):
                obs, actions, old_log_probs, advantages, returns, old_values = batch

                # Normalize advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Get current policy outputs
                _, new_log_probs, entropy, new_values = (
                    self.policy.get_action_and_value(obs, actions)
                )

                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                policy_loss_1 = -advantages * ratio
                policy_loss_2 = -advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Value loss
                if self.clip_range_vf is not None:
                    value_pred_clipped = old_values + torch.clamp(
                        new_values - old_values, -self.clip_range_vf, self.clip_range_vf
                    )
                    value_loss_1 = (new_values - returns) ** 2
                    value_loss_2 = (value_pred_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * ((new_values - returns) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())

                with torch.no_grad():
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1) > self.clip_range).float()
                    ).item()
                    clip_fractions.append(clip_fraction)
                    approx_kl = torch.mean(old_log_probs - new_log_probs).item()
                    approx_kls.append(approx_kl)

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "total_loss": np.mean(total_losses),
            "clip_fraction": np.mean(clip_fractions),
            "approx_kl": np.mean(approx_kls),
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
