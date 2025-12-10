# Custom Loss Functions (Huber Loss)

## Problem Statement

Implement **custom loss functions** from scratch, including the Huber Loss. Understanding how to create custom losses is essential for adapting models to specific problems.

Your task is to:

1. Implement Huber Loss (Smooth L1 Loss)
2. Implement Focal Loss for imbalanced classification
3. Create loss functions as both functions and nn.Module classes

## Requirements

- Implement without using built-in loss functions
- Support reduction modes: 'none', 'mean', 'sum'
- Handle edge cases properly
- Make losses differentiable for backpropagation

## Function Signature

```python
def huber_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    delta: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Huber loss (Smooth L1 Loss).
    
    Combines L1 and L2 loss:
    - L2 for small errors (|error| < delta)
    - L1 for large errors (|error| >= delta)
    """
    pass

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        pass
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass
```

## Example

```python
import torch

pred = torch.randn(10, 1)
target = torch.randn(10, 1)

# Huber loss
loss = huber_loss(pred, target, delta=1.0)
print(f"Huber loss: {loss.item():.4f}")

# Focal loss
focal = FocalLoss(gamma=2.0)
logits = torch.randn(10, 5)  # 10 samples, 5 classes
labels = torch.randint(0, 5, (10,))
loss = focal(logits, labels)
```

## Loss Formulas

**Huber Loss:**
```
L = 0.5 * (y - ŷ)²           if |y - ŷ| < δ
L = δ * |y - ŷ| - 0.5 * δ²   if |y - ŷ| >= δ
```

**Focal Loss:**
```
FL = -α * (1 - p_t)^γ * log(p_t)
where p_t = p if y=1 else (1-p)
```

## Hints

- Huber loss is less sensitive to outliers than MSE
- Use `torch.where` for conditional operations
- Focal loss down-weights easy examples (high p_t)
