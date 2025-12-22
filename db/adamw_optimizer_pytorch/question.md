# AdamW Optimizer

## Problem
AdamW is a variant of Adam that decouples weight decay from the gradient update. Unlike Adam (which adds weight decay to the gradient), AdamW applies weight decay directly to the parameters, which often leads to better generalization in practice.

## Task
Implement a minimal AdamW optimizer in PyTorch as a small class that:
- takes a list of parameter tensors,
- reads gradients from `p.grad`,
- maintains `m` (first moment) and `v` (second moment) buffers for each parameter,
- performs the AdamW parameter update with bias correction and decoupled weight decay.

## Function Signature

```python
class AdamW:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
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
  - `weight_decay`: weight decay coefficient (applied directly to parameters, not gradients)
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
With `lr=0.01`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, starting from `m=0`, `v=0`, after one `step()`:
- The parameter is updated using Adam-style adaptive learning rate
- Weight decay is applied directly: `p = p - lr * weight_decay * p` (separate from the Adam update)
- expected `p ≈ tensor([0.9899, 1.9801])` (approximate, accounting for both Adam update and weight decay)

### Example 2 (zero gradient)
If `p = torch.tensor([3.0])` and `p.grad = torch.tensor([0.0])`, then after `step()`:
- Adam update will be zero (no gradient)
- Weight decay still applies: `p = p - lr * weight_decay * p`
- expected `p ≈ tensor([2.9997])` (with `lr=0.01`, `weight_decay=0.01`)
