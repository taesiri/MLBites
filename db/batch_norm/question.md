# Batch Normalization from Scratch

## Problem Statement

Implement **Batch Normalization** from scratch using PyTorch. Batch Normalization is a technique that normalizes activations across the batch dimension, significantly improving training stability and speed in deep networks.

Key concepts:
- During training: compute batch statistics and update running averages
- During inference: use running averages instead of batch statistics
- Learn scale (gamma) and shift (beta) parameters

Your task is to:

1. Implement BatchNorm for 2D inputs (fully connected layers)
2. Implement BatchNorm for 4D inputs (convolutional layers)
3. Handle training vs. evaluation mode correctly

## Requirements

- Do **NOT** use `nn.BatchNorm1d`, `nn.BatchNorm2d`, or `F.batch_norm`
- Maintain running mean and variance for inference
- Support learnable affine parameters (gamma and beta)
- Correctly handle the momentum parameter for running statistics

## Function Signature

```python
class BatchNorm1d(nn.Module):
    def __init__(
        self, 
        num_features: int, 
        eps: float = 1e-5, 
        momentum: float = 0.1
    ):
        """Initialize Batch Normalization for 1D inputs.
        
        Args:
            num_features: Number of features/channels
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization.
        
        Args:
            x: Input tensor of shape (batch, features) or (batch, features, length)
            
        Returns:
            Normalized tensor
        """
        pass
```

## Example

```python
import torch

# Create BatchNorm layer
bn = BatchNorm1d(64)

# Training mode
bn.train()
x = torch.randn(32, 64)  # batch=32, features=64
output = bn(x)

# Each feature should be normalized across the batch
print(f"Mean (should be ~0): {output.mean(dim=0)[0]:.6f}")
print(f"Std (should be ~1): {output.std(dim=0)[0]:.6f}")

# Evaluation mode uses running statistics
bn.eval()
x_eval = torch.randn(32, 64)
output_eval = bn(x_eval)
```

## Hints

- Training: `x_norm = (x - batch_mean) / sqrt(batch_var + eps)`
- Running stats update: `running = (1 - momentum) * running + momentum * batch_stat`
- For 2D BatchNorm, normalize over (N, H, W), keeping C independent
- Use `self.training` to check if in training mode
- Register buffers for running statistics with `self.register_buffer`
