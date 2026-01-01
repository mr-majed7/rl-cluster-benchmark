"""
IMPALA (Importance Weighted Actor-Learner Architecture) Implementation
CPU-optimized for sequential and parallel training

Reference: https://arxiv.org/abs/1802.01561
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class IMPALA:
    """
    IMPALA algorithm with V-trace off-policy correction.

    Key differences from PPO:
    - Uses V-trace for off-policy correction
    - Separate behavior and learner policies
    - Experience replay buffer for off-policy learning
    """

    def __init__(
        self,
        observation_shape: tuple,
        num_actions: int,
        num_envs: int = 1,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        vtrace_clip_rho_threshold: float = 1.0,
        vtrace_clip_pg_rho_threshold: float = 1.0,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 40.0,
        device: str = "cpu",
        num_threads: Optional[int] = None,
    ):
        """
        Initialize IMPALA algorithm.

        Args:
            observation_shape: Shape of observations
            num_actions: Number of discrete actions
            num_envs: Number of parallel environments
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            vtrace_clip_rho_threshold: Clipping threshold for importance sampling (ρ)
            vtrace_clip_pg_rho_threshold: Clipping threshold for policy gradient (c)
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use (cpu/cuda)
            num_threads: Number of CPU threads (None = auto-detect)
        """
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.num_envs = num_envs

        # Set CPU threading for optimal performance
        if device == "cpu" and num_threads is not None:
            torch.set_num_threads(num_threads)
        elif device == "cpu":
            import os

            num_threads = os.cpu_count() or 1
            torch.set_num_threads(num_threads)
            print(f"IMPALA using {num_threads} CPU threads")

        # Hyperparameters
        self.gamma = gamma
        self.rho_bar = vtrace_clip_rho_threshold
        self.c_bar = vtrace_clip_pg_rho_threshold
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Import here to avoid circular dependency
        from .models import CNNActorCritic

        # Policy network (actor-critic)
        self.policy = CNNActorCritic(observation_shape, num_actions).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=learning_rate,
            eps=1e-5,
            alpha=0.99,
        )

        # Training statistics
        self.update_count = 0

    def compute_vtrace(
        self,
        behavior_logits: torch.Tensor,
        target_logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute V-trace targets for off-policy correction.

        Args:
            behavior_logits: Log probabilities from behavior policy [T, B, A]
            target_logits: Log probabilities from target policy [T, B, A]
            actions: Actions taken [T, B]
            rewards: Rewards received [T, B]
            values: Value estimates [T+1, B]
            dones: Done flags [T, B]

        Returns:
            vs: V-trace value targets [T, B]
            pg_advantages: Policy gradient advantages [T, B]
        """
        T, B = actions.shape

        # Convert logits to log probabilities
        behavior_log_probs = F.log_softmax(behavior_logits, dim=-1)
        target_log_probs = F.log_softmax(target_logits, dim=-1)

        # Get log probabilities for taken actions
        behavior_log_probs_actions = behavior_log_probs.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        target_log_probs_actions = target_log_probs.gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)

        # Importance sampling ratios
        log_rhos = target_log_probs_actions - behavior_log_probs_actions
        rhos = torch.exp(log_rhos)

        # Clipped importance sampling ratios
        clipped_rhos = torch.clamp(rhos, max=self.rho_bar)
        clipped_cs = torch.clamp(rhos, max=self.c_bar)

        # Compute temporal difference
        deltas = rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]

        # V-trace recursive computation (backward pass)
        vs_minus_v = torch.zeros_like(values[:-1])
        vs_minus_v[-1] = clipped_rhos[-1] * deltas[-1]

        for t in reversed(range(T - 1)):
            vs_minus_v[t] = (
                clipped_rhos[t] * deltas[t]
                + self.gamma * clipped_cs[t] * (1 - dones[t]) * vs_minus_v[t + 1]
            )

        # V-trace targets
        vs = values[:-1] + vs_minus_v

        # Policy gradient advantages (using clipped_rhos)
        pg_advantages = clipped_rhos * (
            rewards + self.gamma * values[1:] * (1 - dones) - values[:-1]
        )

        return vs, pg_advantages

    def train_step(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        behavior_logits: np.ndarray,
    ) -> Dict[str, float]:
        """
        Perform one IMPALA training step with V-trace.

        Args:
            observations: Observations [T+1, B, C, H, W]
            actions: Actions taken [T, B]
            rewards: Rewards received [T, B]
            dones: Done flags [T, B]
            behavior_logits: Logits from behavior policy [T, B, A]

        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        obs = torch.FloatTensor(observations).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        behavior_logits_t = torch.FloatTensor(behavior_logits).to(self.device)

        T, B = actions.shape

        # Forward pass through current policy
        with torch.no_grad():
            # Get values for all timesteps
            all_logits = []
            all_values = []

            for t in range(T + 1):
                logits, value = self.policy(obs[t])
                all_logits.append(logits)
                all_values.append(value.squeeze(-1))  # Squeeze [B, 1] -> [B]

            target_logits = torch.stack(all_logits[:-1])  # [T, B, A]
            values = torch.stack(all_values)  # [T+1, B]

        # Compute V-trace targets
        vs, pg_advantages = self.compute_vtrace(
            behavior_logits_t,
            target_logits,
            actions_t,
            rewards_t,
            values,
            dones_t,
        )

        # Now compute losses with gradients
        # Forward pass for training
        logits_list = []
        values_list = []

        for t in range(T):
            logits, value = self.policy(obs[t])
            logits_list.append(logits)
            values_list.append(value.squeeze(-1))  # Squeeze [B, 1] -> [B]

        logits_train = torch.stack(logits_list)  # [T, B, A]
        values_train = torch.stack(values_list)  # [T, B]

        # Policy loss (using V-trace advantages)
        log_probs = F.log_softmax(logits_train, dim=-1)
        log_probs_actions = log_probs.gather(-1, actions_t.unsqueeze(-1)).squeeze(-1)

        policy_loss = -(log_probs_actions * pg_advantages.detach()).mean()

        # Value loss (V-trace targets)
        value_loss = F.mse_loss(values_train, vs.detach())

        # Entropy bonus
        probs = F.softmax(logits_train, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )

        self.optimizer.step()

        self.update_count += 1

        # Return metrics
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "grad_norm": grad_norm.item(),
            "mean_advantage": pg_advantages.mean().item(),
            "mean_value": values_train.mean().item(),
        }

    def get_action(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get action from policy.

        Args:
            observation: Current observation [B, C, H, W]
            deterministic: If True, use greedy action selection

        Returns:
            actions: Selected actions [B]
            logits: Action logits [B, A]
            values: Value estimates [B]
        """
        with torch.no_grad():
            obs = torch.FloatTensor(observation).to(self.device)
            logits, values = self.policy(obs)

            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(-1)

            return (
                actions.cpu().numpy(),
                logits.cpu().numpy(),
                values.cpu().numpy(),
            )

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_count": self.update_count,
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)
