# Leaky ReLU Activation Function

## Problem
Leaky ReLU (Leaky Rectified Linear Unit) is a variation of the standard ReLU activation function that addresses the "dying ReLU" problem. While standard ReLU outputs zero for all negative inputs (which can cause neurons to become inactive and stop learning), Leaky ReLU allows a small, non-zero gradient for negative inputs. This small slope for negative values helps keep neurons alive and learning.

## Task
Implement a `LeakyReLU` class that inherits from `nn.Module` and computes the Leaky ReLU activation function. The class should:
- Accept `negative_slope` as a constructor parameter (default: 0.01)
- Implement a `forward` method that applies the Leaky ReLU activation element-wise
- Return positive inputs unchanged
- Multiply negative inputs by `negative_slope`

## Class Signature

```python
class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs
- **constructor**:
  - `negative_slope`: `float` coefficient for negative inputs (default: 0.01)
- **forward inputs**:
  - `x`: `torch.Tensor` of any shape containing input values
- **forward outputs**:
  - A `torch.Tensor` of the same shape as `x` containing the Leaky ReLU activations

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal boilerplate.
- Assume inputs satisfy the documented contract (valid tensors).
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1 (positive inputs)
```python
x = torch.tensor([1.0, 2.0, 3.0])
leaky_relu = LeakyReLU(negative_slope=0.01)
# All positive, output unchanged
output = leaky_relu(x)
# expected: tensor([1.0, 2.0, 3.0])
```

### Example 2 (negative inputs)
```python
x = torch.tensor([-1.0, -2.0, -3.0])
leaky_relu = LeakyReLU(negative_slope=0.01)
# All negative: multiply by 0.01
output = leaky_relu(x)
# expected: tensor([-0.01, -0.02, -0.03])
```

### Example 3 (mixed inputs with custom slope)
```python
x = torch.tensor([-2.0, 0.0, 2.0])
leaky_relu = LeakyReLU(negative_slope=0.1)
# x[0] < 0: -2.0 * 0.1 = -0.2
# x[1] == 0: treated as non-positive, 0.0 * 0.1 = 0.0
# x[2] > 0: unchanged = 2.0
output = leaky_relu(x)
# expected: tensor([-0.2, 0.0, 2.0])
```



