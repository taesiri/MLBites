# Solution: Huber Loss

## Approach
- Compute the residual (difference) between predictions and targets: `residual = predictions - targets`.
- Compute the absolute residual: `abs_residual = |residual|`.
- For elements where `abs_residual <= delta`: use the quadratic formula `0.5 * residual^2`.
- For elements where `abs_residual > delta`: use the linear formula `delta * (abs_residual - 0.5 * delta)`.
- Use `torch.where()` to select between the two cases element-wise.
- Return the mean of all element-wise losses.

## Correctness
- The piecewise formula matches the standard Huber loss definition.
- For small errors (within delta), the quadratic term provides smooth gradients.
- For large errors (beyond delta), the linear term reduces sensitivity to outliers.
- The function is continuous and differentiable at `abs_residual = delta` since both branches evaluate to `0.5 * delta^2` at that point.
- Using `torch.where()` handles the conditional logic correctly while maintaining gradient flow.

## Complexity
- Time: \(O(n)\) where \(n\) is the number of elements in the input tensors. Each element is processed once.
- Space: \(O(n)\) for intermediate tensors (residuals, abs_residuals, element-wise losses).

## Common Pitfalls
- Forgetting the `0.5` coefficient in the quadratic term.
- Using `residual^2` without the `0.5` factor in the linear term offset (`0.5 * delta`).
- Confusing the condition: it should be `abs_residual <= delta` for quadratic, not `abs_residual < delta`.
- Applying reduction (mean) before computing the piecewise function instead of after.
- Using in-place operations that break autograd.

