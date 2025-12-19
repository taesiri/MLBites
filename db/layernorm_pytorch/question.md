# Layer Normalization

## Problem
Layer Normalization (LayerNorm) is a normalization technique commonly used in Transformers and RNNs. Unlike Batch Normalization which normalizes across the batch dimension, Layer Normalization normalizes across the feature dimensions for each individual sample. This makes it particularly suitable for sequence models where batch statistics can be unreliable.

## Task
Implement a `LayerNorm` class in PyTorch from scratch. The class should normalize the input over the last dimension, then apply learnable scale (weight) and shift (bias) parameters.

Your implementation should match the behavior of `torch.nn.LayerNorm` when given a single normalized dimension.

## Function Signature

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

## Inputs and Outputs

### `__init__`
- **Inputs**:
  - `normalized_shape`: Size of the last dimension to normalize over (integer)
  - `eps`: Small constant for numerical stability (default 1e-5)
- The weight and bias should be initialized to ones and zeros respectively

### `forward`
- **Inputs**:
  - `x`: Input tensor of shape `(..., normalized_shape)` where `...` means any number of leading dimensions
- **Outputs**:
  - Normalized tensor of the same shape as input

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: focus on correctness, not edge cases.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- Do not call `torch.nn.LayerNorm` or `F.layer_norm` — implement the math yourself.

## Examples

### Example 1 (basic normalization)
```python
import torch

layer_norm = LayerNorm(4)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

out = layer_norm(x)
# Each row is normalized to have mean ≈ 0, std ≈ 1
# out ≈ [[-1.3416, -0.4472, 0.4472, 1.3416]]
```

### Example 2 (with learned parameters)
```python
import torch

layer_norm = LayerNorm(3)
# Manually set weight and bias
layer_norm.weight.data = torch.tensor([2.0, 1.0, 0.5])
layer_norm.bias.data = torch.tensor([1.0, 0.0, -1.0])

x = torch.tensor([[1.0, 2.0, 3.0]])
out = layer_norm(x)
# Normalized then scaled/shifted
# out ≈ [[-1.4495, 0.0, -0.6376]]
```

### Example 3 (3D input)
```python
import torch

layer_norm = LayerNorm(4)
x = torch.randn(2, 3, 4)  # (batch, sequence, features)

out = layer_norm(x)
# out has shape (2, 3, 4)
# Each of the 6 vectors of length 4 is independently normalized
```

