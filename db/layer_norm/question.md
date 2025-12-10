# Layer Normalization from Scratch

## Problem Statement

Implement **Layer Normalization** from scratch using PyTorch. Layer Normalization is a technique used extensively in Transformers and RNNs to stabilize training by normalizing activations across the feature dimension.

Unlike Batch Normalization, Layer Normalization:
- Normalizes across features (not batch)
- Works the same during training and inference
- Is independent of batch size

Your task is to:

1. Implement Layer Normalization with learnable parameters (gamma and beta)
2. Compute mean and variance across the normalized dimensions
3. Apply the normalization formula with numerical stability

## Requirements

- Do **NOT** use `nn.LayerNorm` or `F.layer_norm`
- Support learnable scale (gamma) and shift (beta) parameters
- Handle the epsilon parameter for numerical stability
- Support normalization over configurable dimensions

## Function Signature

```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple, eps: float = 1e-5):
        """Initialize Layer Normalization.
        
        Args:
            normalized_shape: Shape of the features to normalize
            eps: Small constant for numerical stability
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor (same shape as input)
        """
        pass
```

## Example

```python
import torch

# Create Layer Norm for embedding dim 512
layer_norm = LayerNorm(512)

# Input: batch of sequences
x = torch.randn(4, 10, 512)  # (batch, seq_len, embed_dim)

# Apply layer normalization
output = layer_norm(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Check that each position is normalized
print(f"Mean (should be ~0): {output[0, 0].mean():.6f}")
print(f"Std (should be ~1): {output[0, 0].std():.6f}")
```

## Hints

- For input of shape (batch, seq, features), normalize over the features dimension
- Mean: `μ = mean(x, dim=-1, keepdim=True)`
- Variance: `σ² = var(x, dim=-1, keepdim=True, unbiased=False)`
- Normalization: `x_norm = (x - μ) / sqrt(σ² + ε)`
- Output: `y = γ * x_norm + β`
