# Softmax from Scratch

## Problem Statement

Implement the **Softmax** function from scratch using PyTorch. Softmax is a crucial activation function that converts raw logits into a probability distribution, and is used in classification tasks and attention mechanisms.

Your task is to:

1. Implement a numerically stable softmax function
2. Handle the temperature parameter for controlling the distribution
3. Support softmax along different dimensions

## Requirements

- Do **NOT** use `F.softmax`, `torch.softmax`, or `nn.Softmax`
- Handle numerical stability (prevent overflow/underflow)
- Support softmax along any specified dimension
- Optionally support temperature scaling

## Function Signature

```python
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute numerically stable softmax.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax
        
    Returns:
        Softmax probabilities (same shape as input)
    """
    pass

def softmax_with_temperature(
    x: torch.Tensor, 
    temperature: float = 1.0, 
    dim: int = -1
) -> torch.Tensor:
    """Compute softmax with temperature scaling.
    
    Higher temperature -> more uniform distribution
    Lower temperature -> more peaked distribution
    
    Args:
        x: Input tensor
        temperature: Temperature parameter (default 1.0)
        dim: Dimension along which to compute softmax
        
    Returns:
        Softmax probabilities
    """
    pass
```

## Example

```python
import torch

# Simple logits
logits = torch.tensor([2.0, 1.0, 0.1])

# Compute softmax
probs = softmax(logits)
print(f"Probabilities: {probs}")
print(f"Sum: {probs.sum()}")  # Should be 1.0

# With temperature
hot_probs = softmax_with_temperature(logits, temperature=0.5)  # More peaked
cold_probs = softmax_with_temperature(logits, temperature=2.0)  # More uniform

# Batch processing
batch_logits = torch.randn(4, 10)  # 4 samples, 10 classes
batch_probs = softmax(batch_logits, dim=-1)
print(f"Each row sums to 1: {batch_probs.sum(dim=-1)}")
```

## Hints

- The naive formula `exp(x) / sum(exp(x))` can overflow for large values
- For numerical stability, subtract the maximum before exponentiating
- `softmax(x) = softmax(x - max(x))` (invariant to constant shift)
- Remember to keep dimensions aligned when subtracting max and dividing
