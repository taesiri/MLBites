# Implement SGD Optimizer

## Problem
Stochastic Gradient Descent (SGD) is a foundational optimizer. In PyTorch, `torch.optim.SGD` can optionally use momentum and weight decay (L2 regularization).

## Task
Implement a minimal SGD optimizer in PyTorch as a small class that:
- takes a list of parameter tensors,
- reads gradients from `p.grad`,
- optionally applies weight decay,
- optionally uses momentum,
- updates parameters in-place.

The update must match the behavior of `torch.optim.SGD` for the supported features.

## Function Signature

```python
class SGD:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None: ...

    @torch.no_grad()
    def step(self) -> None: ...
```

## Inputs and Outputs
- **inputs**:
  - `params`: list of `torch.Tensor` parameters to update in-place
  - `lr`: learning rate
  - `momentum`: momentum factor (0.0 means no momentum)
  - `weight_decay`: L2 penalty coefficient (0.0 means no weight decay)
- **outputs**:
  - `step()` returns `None` and updates tensors in `params` **in-place**

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: no need to subclass `torch.optim.Optimizer`.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1 (no momentum, no weight decay)
If `p = tensor([1.0, 2.0])`, `p.grad = tensor([0.1, -0.2])`, and `lr=0.1`, then one step does:
- expected `p == tensor([0.99, 2.02])`

### Example 2 (momentum speeds up in the same direction)
With momentum, a second step with the same gradient will usually move further than plain SGD.
In this exercise, tests will compare your updates directly against `torch.optim.SGD`.


