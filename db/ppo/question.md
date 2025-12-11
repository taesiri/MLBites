# PPO (Proximal Policy Optimization)

## Problem Statement

Implement **PPO (Proximal Policy Optimization)**, a popular policy gradient algorithm that achieves stable training through clipped objectives. PPO is used for training ChatGPT (RLHF).

Your task is to:

1. Implement actor-critic network
2. Compute GAE (Generalized Advantage Estimation)
3. Implement clipped surrogate objective
4. Train with multiple epochs per batch

## PPO Components

| Component | Purpose |
|-----------|---------|
| Actor | Outputs action probabilities π(a\|s) |
| Critic | Estimates value V(s) |
| GAE | Computes advantages with bias-variance tradeoff |
| Clipped Loss | Prevents too large policy updates |

## Function Signature

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        pass
    
    def forward(self, state: torch.Tensor) -> tuple:
        """Return (action_probs, value)."""
        pass

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    pass

def ppo_loss(old_log_probs, log_probs, advantages, clip_epsilon=0.2):
    """Clipped surrogate objective."""
    pass
```

## PPO Objective

```python
ratio = exp(log_prob - old_log_prob)
clipped = clip(ratio, 1-ε, 1+ε)
L_policy = -min(ratio * A, clipped * A)
L_value = (V - V_target)²
L = L_policy + c1 * L_value - c2 * entropy
```

## Example

```python
agent = PPOAgent(state_dim=4, action_dim=2)

# Collect trajectory
states, actions, rewards, log_probs = collect_trajectory(env, agent)
advantages = compute_gae(rewards, values, dones)

# Train for multiple epochs on same batch
for _ in range(K_epochs):
    agent.update(states, actions, advantages, old_log_probs)
```

## Hints

- Normalize advantages (subtract mean, divide by std)
- Use separate networks for actor/critic or shared backbone
- Entropy bonus encourages exploration
- Typical clip_epsilon: 0.1-0.2, K_epochs: 3-10
