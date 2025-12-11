"""
PPO (Proximal Policy Optimization) - Starting Point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Combined actor-critic network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # TODO: Shared backbone
        # TODO: Actor head (policy)
        # TODO: Critic head (value)
        pass
    
    def forward(self, state: torch.Tensor):
        """Return (action_logits, value)."""
        pass
    
    def get_action_and_value(self, state: torch.Tensor, action: torch.Tensor = None):
        """Get action, log_prob, entropy, and value."""
        # TODO: Compute policy distribution
        # TODO: Sample action if not provided
        # TODO: Compute log probability of action
        # TODO: Compute entropy
        # TODO: Compute value
        pass


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                next_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
    """
    Compute Generalized Advantage Estimation.
    
    GAE(γ,λ) = Σ (γλ)^t * δ_t
    where δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    """
    # TODO: Compute TD errors (deltas)
    # TODO: Compute GAE by accumulating weighted deltas
    pass


def ppo_loss(old_log_probs: torch.Tensor, log_probs: torch.Tensor,
             advantages: torch.Tensor, clip_epsilon: float = 0.2):
    """
    Clipped surrogate objective.
    
    L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    """
    # TODO: Compute probability ratio
    # TODO: Compute clipped ratio
    # TODO: Return loss
    pass


class PPOAgent:
    """PPO agent."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95, clip_epsilon: float = 0.2,
                 k_epochs: int = 4, entropy_coef: float = 0.01, value_coef: float = 0.5):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # TODO: Create actor-critic network
        # TODO: Create optimizer
        pass
    
    def select_action(self, state: torch.Tensor):
        """Select action and return (action, log_prob, value)."""
        pass
    
    def update(self, rollout_buffer):
        """Update policy using collected experience."""
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    agent = PPOAgent(state_dim=4, action_dim=2)
    
    # Dummy rollout
    states = torch.randn(64, 4)
    actions = torch.randint(0, 2, (64,))
    rewards = torch.randn(64)
    dones = torch.zeros(64)
    
    action, log_prob, value = agent.select_action(states[0])
    print(f"Action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
