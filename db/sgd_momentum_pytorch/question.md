# SGD with Momentum

## Problem
Momentum is a simple yet powerful enhancement to vanilla Stochastic Gradient Descent. Instead of updating parameters using only the current gradient, momentum accumulates a "velocity" that smooths updates and helps escape shallow local minima.

The momentum update rule is:
- \( v_t = \mu \cdot v_{t-1} + g_t \)
- \( \theta_t = \theta_{t-1} - \alpha \cdot v_t \)

where \( \mu \) is the momentum coefficient, \( \alpha \) is the learning rate, \( g_t \) is the gradient, and \( v_t \) is the velocity buffer.

## Task
Implement a minimal SGD optimizer with momentum in PyTorch as a class that:
- Takes a list of parameter tensors and hyperparameters (lr, momentum)
- Reads gradients from `p.grad`
- Maintains a velocity buffer for each parameter
- Updates parameters in-place

Your implementation must match the behavior of `torch.optim.SGD` with `momentum > 0`.

## Function Signature

```python
class SGDMomentum:
    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-2,
        momentum: float = 0.9,
    ) -> None: ...

    @torch.no_grad()
    def step(self) -> None: ...
```

## Inputs and Outputs
- **inputs**:
  - `params`: list of `torch.Tensor` parameters to update in-place
  - `lr`: learning rate (default 0.01)
  - `momentum`: momentum coefficient (default 0.9)
- **outputs**:
  - `step()` returns `None` and updates tensors in `params` **in-place**

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: no need to subclass `torch.optim.Optimizer`.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1 (single step)
```python
p = torch.tensor([1.0, 2.0], requires_grad=True)
p.grad = torch.tensor([0.1, -0.2])
opt = SGDMomentum([p], lr=0.1, momentum=0.9)
opt.step()
# After step: p ≈ tensor([0.99, 2.02])
# velocity buffer = [0.1, -0.2]
```

### Example 2 (momentum accumulation)
With `momentum=0.9`, if the gradient stays the same direction over multiple steps, the velocity grows and the parameter moves faster:
```python
# Step 1: v = 0*0.9 + g = g, update = -lr * g
# Step 2: v = v*0.9 + g = 0.9g + g = 1.9g, update = -lr * 1.9g
# Step 3: v = 1.9g*0.9 + g = 2.71g, update = -lr * 2.71g
```
This acceleration effect helps converge faster in consistent gradient directions.


