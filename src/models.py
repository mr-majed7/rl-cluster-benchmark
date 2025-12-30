"""Neural network models for RL agents."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNActorCritic(nn.Module):
    """Actor-Critic network with CNN encoder for image observations."""
    
    def __init__(self, observation_shape: Tuple[int, int, int], num_actions: int, hidden_size: int = 512):
        super().__init__()
        
        # CNN encoder for procgen (64x64x3 images)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the size of the flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape)
            dummy_output = self.encoder(dummy_input)
            encoder_output_size = dummy_output.shape[1]
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(encoder_output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value estimate.
        
        Args:
            x: Observation tensor of shape (batch, C, H, W)
            
        Returns:
            action_logits: Tensor of shape (batch, num_actions)
            value: Tensor of shape (batch, 1)
        """
        features = self.encoder(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value.
        
        Args:
            x: Observation tensor
            action: Optional action tensor for evaluation
            
        Returns:
            action: Sampled or provided action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: Value estimate
        """
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)

