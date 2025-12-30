"""Experience replay buffer for PPO."""
import numpy as np
import torch
from typing import Generator, Tuple


class RolloutBuffer:
    """Buffer for storing rollout experiences for PPO training."""
    
    def __init__(self, buffer_size: int, observation_shape: Tuple[int, ...], 
                 num_envs: int, device: torch.device):
        """Initialize the rollout buffer.
        
        Args:
            buffer_size: Number of steps to collect before update
            observation_shape: Shape of observations
            num_envs: Number of parallel environments
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        
        # Allocate memory
        self.observations = torch.zeros((buffer_size, num_envs) + observation_shape, dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, num_envs), dtype=torch.long)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        
        # Computed during finalize
        self.advantages = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size, num_envs), dtype=torch.float32)
        
        self.pos = 0
        
    def add(self, obs: np.ndarray, action: torch.Tensor, reward: np.ndarray, 
            done: np.ndarray, value: torch.Tensor, log_prob: torch.Tensor):
        """Add a step of experience.
        
        Args:
            obs: Observations
            action: Actions taken
            reward: Rewards received
            done: Done flags
            value: Value estimates
            log_prob: Log probabilities of actions
        """
        self.observations[self.pos] = torch.from_numpy(obs).float()
        self.actions[self.pos] = action
        self.rewards[self.pos] = torch.from_numpy(reward).float()
        self.dones[self.pos] = torch.from_numpy(done).float()
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        
    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate for the last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.advantages = advantages
        self.returns = advantages + self.values
        
    def get(self, batch_size: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """Generate batches of experiences.
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Batches of (obs, actions, old_log_probs, advantages, returns, old_values)
        """
        # Flatten the buffers
        obs = self.observations.reshape(-1, *self.observations.shape[2:])
        actions = self.actions.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        values = self.values.reshape(-1)
        
        # Move to device
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        log_probs = log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        values = values.to(self.device)
        
        # Generate random indices
        indices = np.arange(len(obs))
        np.random.shuffle(indices)
        
        # Generate batches
        for start in range(0, len(obs), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield (
                obs[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
                values[batch_indices]
            )
    
    def reset(self):
        """Reset the buffer."""
        self.pos = 0

