# Implement AdamW Optimizer

## Problem
AdamW is a variant of Adam that uses **decoupled weight decay**. Instead of adding an L2 term into the gradient (like classic weight decay), AdamW applies weight decay directly to the parameters as a separate step.

PyTorch provides this as `torch.optim.AdamW`.

## Task
Implement a minimal AdamW optimizer in PyTorch as a small class that:
- takes a list of parameter tensors,
- reads gradients from `p.grad`,
- maintains per-parameter state (`m` and `v`) like Adam,
- applies **decoupled weight decay** (when `weight_decay != 0`),
- performs the AdamW update with bias correction.

Your update should match `torch.optim.AdamW` for the supported features.

## Function Signature

```python
class AdamW:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None: ...

    @torch.no_grad()
    def step(self) -> None: ...
```

## Inputs and Outputs
- **inputs**:
  - `params`: list of `torch.Tensor` parameters to update in-place
  - `lr`: learning rate
  - `betas`: `(beta1, beta2)` coefficients for first/second moment estimates
  - `eps`: small constant for numerical stability
  - `weight_decay`: decoupled weight decay coefficient
- **outputs**:
  - `step()` returns `None` and updates tensors in `params` **in-place**

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: no need to subclass `torch.optim.Optimizer`.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1 (decoupled weight decay shrinks parameters even if gradient is small)
If `weight_decay > 0`, then each step should include an in-place shrink:
`p *= (1 - lr * weight_decay)` (applied only for parameters with gradients in this exercise).

### Example 2 (matches PyTorch)
For deterministic tensors and gradients, your `AdamW.step()` should match `torch.optim.AdamW` step-by-step.


