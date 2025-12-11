# DQN (Deep Q-Network)

## Problem Statement

Implement **DQN (Deep Q-Network)** for reinforcement learning. DQN uses a neural network to approximate the Q-function and pioneered deep RL by achieving human-level performance on Atari games.

Your task is to:

1. Implement the Q-network
2. Create a replay buffer for experience replay
3. Implement ε-greedy action selection
4. Use a target network for stable training

## DQN Key Components

| Component | Purpose |
|-----------|---------|
| Q-Network | Estimates Q(s, a) for all actions |
| Target Network | Stable Q-targets (updated periodically) |
| Replay Buffer | Store and sample past experiences |
| ε-greedy | Balance exploration vs exploitation |

## Function Signature

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        pass
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        pass

class ReplayBuffer:
    def __init__(self, capacity: int):
        pass
    
    def push(self, state, action, reward, next_state, done):
        pass
    
    def sample(self, batch_size: int) -> tuple:
        pass

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3):
        pass
    
    def select_action(self, state: torch.Tensor, epsilon: float) -> int:
        pass
    
    def train_step(self, batch) -> float:
        pass
```

## DQN Loss

```
L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))²]
```

## Example

```python
agent = DQNAgent(state_dim=4, action_dim=2)  # CartPole
buffer = ReplayBuffer(10000)

for episode in range(1000):
    state = env.reset()
    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        
        if len(buffer) > batch_size:
            agent.train_step(buffer.sample(batch_size))
```

## Hints

- Target network updated every N steps (not every step)
- Decay ε from 1.0 to 0.01 over training
- Huber loss more stable than MSE
- Double DQN: use online network to select action, target to evaluate
