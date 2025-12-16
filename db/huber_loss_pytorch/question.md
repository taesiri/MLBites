# Implement Huber Loss (PyTorch)

## Problem
Huber Loss is a loss function commonly used in regression that combines the best properties of Mean Squared Error (MSE) and Mean Absolute Error (MAE). For small errors, it behaves like MSE (quadratic), while for large errors it behaves like MAE (linear). This makes it more robust to outliers than MSE while remaining differentiable everywhere.

## Task
Implement a function that computes the Huber Loss between predictions and targets. The function should:
- Compute element-wise Huber loss
- Use the squared loss for residuals within the delta threshold
- Use the linear loss for residuals outside the delta threshold
- Return the mean loss over all elements

## Function Signature

```python
def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor: ...
```

## Inputs and Outputs
- **inputs**:
  - `predictions`: `torch.Tensor` of any shape containing model predictions
  - `targets`: `torch.Tensor` of the same shape as `predictions` containing ground truth values
  - `delta`: `float` threshold that determines where to switch from quadratic to linear loss (default: 1.0)
- **outputs**:
  - A scalar `torch.Tensor` containing the mean Huber loss

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: minimal boilerplate.
- Assume inputs satisfy the documented contract (same shapes, valid tensors).
- Allowed libs: PyTorch (`torch`) and Python standard library.

## Examples

### Example 1 (small residuals, within delta)
```python
predictions = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.2, 1.8, 3.1])
delta = 1.0
# residuals = [-0.2, 0.2, -0.1], all |residual| <= 1.0
# huber = 0.5 * residual^2 = [0.02, 0.02, 0.005]
# mean = 0.015
loss = huber_loss(predictions, targets, delta)
# expected: tensor(0.015)
```

### Example 2 (large residuals, outside delta)
```python
predictions = torch.tensor([0.0, 5.0])
targets = torch.tensor([3.0, 0.0])
delta = 1.0
# residuals = [-3.0, 5.0], all |residual| > 1.0
# huber = delta * (|residual| - 0.5 * delta) = [2.5, 4.5]
# mean = 3.5
loss = huber_loss(predictions, targets, delta)
# expected: tensor(3.5)
```

### Example 3 (mixed residuals)
```python
predictions = torch.tensor([1.0, 0.0])
targets = torch.tensor([1.5, 3.0])
delta = 1.0
# residual[0] = -0.5, |r| <= delta -> 0.5 * 0.25 = 0.125
# residual[1] = -3.0, |r| > delta -> 1.0 * (3.0 - 0.5) = 2.5
# mean = 1.3125
loss = huber_loss(predictions, targets, delta)
# expected: tensor(1.3125)
```

