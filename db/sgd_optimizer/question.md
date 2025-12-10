# SGD Optimizer from Scratch

## Problem Statement

Implement **Stochastic Gradient Descent (SGD)** with momentum from scratch. SGD is the foundational optimization algorithm in deep learning, and understanding its implementation is essential.

Your task is to:

1. Implement basic SGD update rule
2. Add momentum support
3. Add Nesterov momentum variant
4. Support weight decay (L2 regularization)

## Requirements

- Do **NOT** use `torch.optim.SGD`
- Implement momentum with velocity tracking
- Support Nesterov accelerated gradient
- Support weight decay

## Function Signature

```python
class SGD:
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        """Initialize SGD optimizer."""
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
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(100):
    output = model(x)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Update Rules

**Basic SGD:**
```
θ = θ - lr * ∇L(θ)
```

**SGD with Momentum:**
```
v = μ * v + ∇L(θ)
θ = θ - lr * v
```

**Nesterov Momentum:**
```
v = μ * v + ∇L(θ + μ * v)  # Look-ahead gradient
θ = θ - lr * v
```

## Hints

- Store velocity per parameter for momentum
- Nesterov can be implemented by modifying how gradients are applied
- Weight decay adds `weight_decay * param` to the gradient
