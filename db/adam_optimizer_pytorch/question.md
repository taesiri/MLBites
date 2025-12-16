# Implement Adam Optimizer

## Problem
Adam is one of the most commonly used optimizers for training neural networks. In PyTorch, optimizers update parameters based on their gradients and keep running statistics (momentum + second moment).

## Task
Implement a minimal Adam optimizer in PyTorch as a small class that:
- takes a list of parameter tensors,
- reads gradients from `p.grad`,
- maintains `m` (first moment) and `v` (second moment) buffers for each parameter,
- performs the Adam parameter update with bias correction.

## Function Signature

```python
class Adam:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None: ...

    @torch.no_grad()
    def step(self) -> None: ...
```

## Inputs and Outputs
- **inputs**:
  - `params`: list of `torch.Tensor` parameters to update (each may have `p.grad` set by autograd)
  - `lr`: learning rate (float)
  - `betas`: `(beta1, beta2)` for the exponential moving averages
  - `eps`: small constant added to the denominator for numerical stability
- **outputs**:
  - `step()` returns `None` and updates the tensors in `params` **in-place**

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: avoid heavy boilerplate (no need to subclass `torch.optim.Optimizer`).
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1 (single update direction)
Assume a single parameter tensor `p = torch.tensor([1.0, 2.0])` and a gradient `g = torch.tensor([0.1, -0.2])`.
With `lr=0.01`, `betas=(0.9, 0.999)`, `eps=1e-8`, starting from `m=0`, `v=0`, after one `step()` the updated parameter is:
- expected `p ≈ tensor([0.99, 2.01])`

### Example 2 (zero gradient)
If `p = torch.tensor([3.0])` and `p.grad = torch.tensor([0.0])`, then after `step()`:
- expected `p == tensor([3.0])`


