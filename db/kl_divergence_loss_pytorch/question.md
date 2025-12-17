# Implement KL Divergence Loss

## Problem
Kullback-Leibler (KL) Divergence is a measure of how one probability distribution differs from a reference probability distribution. It is widely used in machine learning for tasks such as variational autoencoders (VAEs), knowledge distillation, and regularization. KL divergence is asymmetric: \( D_{KL}(P \| Q) \neq D_{KL}(Q \| P) \).

## Task
Implement a `KLDivLoss` class that inherits from `nn.Module` and computes the KL Divergence loss. The class should:
- Accept `reduction` as a constructor parameter (one of `"mean"`, `"batchmean"`, `"sum"`, or `"none"`)
- Implement a `forward` method that computes KL divergence
- Follow PyTorch's convention: `input` contains **log-probabilities** (log Q), `target` contains **probabilities** (P)
- Use the formula: \( D_{KL}(P \| Q) = \sum P(x) \cdot (\log P(x) - \log Q(x)) \)

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
  - `input`: `torch.Tensor` of shape `(N, C)` or `(N, C, ...)` containing **log-probabilities** (log Q)
  - `target`: `torch.Tensor` of the same shape as `input` containing **probabilities** (P), should sum to 1 along the distribution dimension
- **forward outputs**:
  - A `torch.Tensor` containing the KL divergence loss (scalar if reduced, same shape as input if `reduction="none"`)

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal boilerplate.
- Assume inputs satisfy the documented contract (same shapes, valid probability distributions).
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1 (simple distributions, batchmean reduction)
```python
input = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25]]))  # log Q (uniform)
target = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # P
loss_fn = KLDivLoss(reduction="batchmean")
# KL = sum(P * (log P - log Q))
#    = 0.1*log(0.1/0.25) + 0.2*log(0.2/0.25) + 0.3*log(0.3/0.25) + 0.4*log(0.4/0.25)
#    ≈ 0.0719
loss = loss_fn(input, target)
# expected: tensor(≈0.0719)
```

### Example 2 (identical distributions)
```python
p = torch.tensor([[0.2, 0.3, 0.5]])
input = torch.log(p)  # log Q = log P
target = p  # P
loss_fn = KLDivLoss(reduction="batchmean")
# KL divergence of identical distributions is 0
loss = loss_fn(input, target)
# expected: tensor(0.0)
```

### Example 3 (no reduction)
```python
input = torch.log(torch.tensor([[0.5, 0.5], [0.25, 0.75]]))
target = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
loss_fn = KLDivLoss(reduction="none")
# Row 0: P=Q, so KL=0 for each element
# Row 1: 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75)
loss = loss_fn(input, target)
# expected: tensor([[0.0, 0.0], [0.3466, -0.2027]])
```

