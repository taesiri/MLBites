"""
DQN (Deep Q-Network) - Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        return self.net(state)


class DuelingQNetwork(nn.Module):
    """Dueling DQN variant: separate value and advantage streams."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        x = self.features(state)
        value = self.value(x)
        advantage = self.advantage(x)
        # Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states]),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in next_states]),
            torch.tensor(dones, dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                 gamma: float = 0.99, target_update_freq: int = 100,
                 double_dqn: bool = True):
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.steps = 0
        
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
    
    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax(dim=-1).item()
    
    def train_step(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch
        
        # Current Q values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select action
                next_actions = self.q_network(next_states).argmax(dim=-1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_network(next_states).max(dim=-1)[0]
            
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss
        loss = F.smooth_l1_loss(q_values, targets)  # Huber loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


if __name__ == "__main__":
    torch.manual_seed(42)
    
    agent = DQNAgent(state_dim=4, action_dim=2)
    buffer = ReplayBuffer(10000)
    
    for _ in range(100):
        state = torch.randn(4)
        action = random.randint(0, 1)
        reward = random.random()
        next_state = torch.randn(4)
        done = random.random() > 0.9
        buffer.push(state, action, reward, next_state, done)
    
    for i in range(10):
        batch = buffer.sample(32)
        loss = agent.train_step(batch)
        print(f"Step {i}, Loss: {loss:.4f}")
