# Adam Optimizer from Scratch

## Problem Statement

Implement the **Adam (Adaptive Moment Estimation)** optimizer from scratch. Adam is one of the most popular optimizers in deep learning, combining the benefits of AdaGrad and RMSprop with momentum.

Adam maintains:
- First moment (mean of gradients) - like momentum
- Second moment (uncentered variance of gradients) - like RMSprop
- Bias correction for both moments

Your task is to:

1. Implement the Adam update rule with bias correction
2. Handle per-parameter state (first and second moments)
3. Support weight decay (AdamW variant)

## Requirements

- Do **NOT** use `torch.optim.Adam` or `torch.optim.AdamW`
- Implement bias correction for the moments
- Support the standard Adam hyperparameters (lr, betas, eps)
- Optionally implement AdamW (decoupled weight decay)

## Function Signature

```python
class Adam:
    def __init__(
        self,
        params,
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """Initialize Adam optimizer."""
        pass
    
    def step(self):
        """Perform a single optimization step."""
        pass
    
    def zero_grad(self):
        """Zero out gradients."""
        pass
```

## Example

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    output = model(x)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Adam Update Rules

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t           # First moment
v_t = β2 * v_{t-1} + (1 - β2) * g_t²          # Second moment

m̂_t = m_t / (1 - β1^t)                        # Bias-corrected first moment
v̂_t = v_t / (1 - β2^t)                        # Bias-corrected second moment

θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)        # Parameter update
```

## Hints

- Store state per parameter (m, v, step count)
- Use `param.data` and `param.grad.data` for updates
- Bias correction is crucial in early steps
- For AdamW, apply weight decay directly to parameters, not gradients
