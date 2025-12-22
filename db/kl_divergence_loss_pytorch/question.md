# KL Divergence Loss

## Problem
Implement a KL Divergence loss module for PyTorch.

## Task
Implement a `KLDivLoss` class that inherits from `nn.Module` and computes the KL Divergence loss between two probability distributions.

## Class Signature

```python
class KLDivLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None: ...

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor: ...
```

## Inputs and Outputs
- **constructor**:
  - `reduction`: `str` specifying how to reduce the output (`"mean"`, `"batchmean"`, `"sum"`, or `"none"`; default: `"mean"`)
- **forward inputs**:
  - `input`: `torch.Tensor` of shape `(N, C)` or `(N, C, ...)` containing log-probabilities
  - `target`: `torch.Tensor` of the same shape as `input` containing probabilities
- **forward outputs**:
  - A `torch.Tensor` containing the KL divergence loss (scalar if reduced, same shape as input if `reduction="none"`)

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal boilerplate.
- Assume inputs satisfy the documented contract (same shapes, valid probability distributions).
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1
```python
input = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25]]))
target = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
loss_fn = KLDivLoss(reduction="batchmean")
loss = loss_fn(input, target)
# expected: tensor(≈0.0719)
```

### Example 2
```python
p = torch.tensor([[0.2, 0.3, 0.5]])
input = torch.log(p)
target = p
loss_fn = KLDivLoss(reduction="batchmean")
loss = loss_fn(input, target)
# expected: tensor(0.0)
```




