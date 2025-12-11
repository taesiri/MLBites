"""
PPO (Proximal Policy Optimization) - Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = self.shared(state)
        return self.actor(x), self.critic(x)
    
    def get_action_and_value(self, state, action=None):
        logits, value = self(state)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    
    returns = advantages + values
    return advantages, returns


def ppo_clip_loss(old_log_probs, log_probs, advantages, clip_epsilon=0.2):
    """Clipped surrogate objective."""
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    return loss


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.tensor(self.rewards),
            torch.stack(self.values),
            torch.tensor(self.dones, dtype=torch.float32),
        )
    
    def clear(self):
        self.__init__()


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95, clip_epsilon: float = 0.2,
                 k_epochs: int = 4, entropy_coef: float = 0.01, value_coef: float = 0.5):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state)
        return action.item(), log_prob, value
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.k_epochs):
            _, log_probs, entropy, values = self.network.get_action_and_value(states, actions)
            
            # Policy loss (clipped)
            policy_loss = ppo_clip_loss(old_log_probs, log_probs, advantages, self.clip_epsilon)
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    agent = PPOAgent(state_dim=4, action_dim=2)
    buffer = RolloutBuffer()
    
    # Simulate rollout
    for _ in range(64):
        state = torch.randn(4)
        action, log_prob, value = agent.select_action(state)
        buffer.add(state, torch.tensor(action), log_prob, np.random.randn(), value, False)
    
    states, actions, old_log_probs, rewards, values, dones = buffer.get()
    
    # Compute advantages
    advantages, returns = compute_gae(rewards, values, dones, values[-1], 0.99, 0.95)
    
    # Update
    policy_loss, value_loss = agent.update(states, actions, old_log_probs, returns, advantages)
    print(f"Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}")
