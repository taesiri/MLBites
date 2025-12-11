"""
DQN (Deep Q-Network) - Starting Point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque


class QNetwork(nn.Module):
    """Q-value network."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # TODO: Create MLP
        pass
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        pass


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int):
        # TODO: Initialize buffer
        pass
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        pass
    
    def sample(self, batch_size: int):
        """Sample random batch of experiences."""
        pass
    
    def __len__(self):
        pass


class DQNAgent:
    """DQN agent."""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, target_update_freq: int = 100):
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # TODO: Create Q-network and target network
        # TODO: Create optimizer
        pass
    
    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        """Select action using ε-greedy policy."""
        # TODO: With probability ε, select random action
        # TODO: Otherwise, select argmax_a Q(s, a)
        pass
    
    def train_step(self, batch) -> float:
        """Perform one training step."""
        # TODO: Unpack batch
        # TODO: Compute current Q values
        # TODO: Compute target Q values (with target network)
        # TODO: Compute loss
        # TODO: Update network
        # TODO: Periodically update target network
        pass
    
    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        pass


def train_dqn(env, agent, buffer, episodes=500, batch_size=64, 
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """Training loop."""
    epsilon = epsilon_start
    
    for episode in range(episodes):
        # TODO: Reset environment
        # TODO: Episode loop
        # TODO: Collect experience
        # TODO: Train agent
        # TODO: Decay epsilon
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Simple test
    agent = DQNAgent(state_dim=4, action_dim=2)
    buffer = ReplayBuffer(10000)
    
    # Dummy experience
    for _ in range(100):
        state = torch.randn(4)
        action = random.randint(0, 1)
        reward = random.random()
        next_state = torch.randn(4)
        done = random.random() > 0.9
        buffer.push(state, action, reward, next_state, done)
    
    # Train step
    batch = buffer.sample(32)
    loss = agent.train_step(batch)
    print(f"Loss: {loss:.4f}")
